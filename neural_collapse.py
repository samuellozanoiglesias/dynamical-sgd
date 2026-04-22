from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


EPS = 1e-12


@dataclass
class NCLayerSpec:
	name: str
	mode: str  # "input", "hook_input", or "hook_output"
	module: nn.Module | None


@dataclass
class NCEpochRaw:
	class_counts: np.ndarray
	margin_sum: float
	margin_sq_sum: float
	margin_count: int
	layer_mu_sqnorm: dict[str, np.ndarray]
	layer_feat_sqmean: dict[str, np.ndarray]
	layer_pair_dot: dict[str, np.ndarray]


def build_nc_class_pairs(num_classes: int) -> list[tuple[int, int]]:
	# Keep adjacency-first ordering: (0,1), (1,2), ... then remaining pairs.
	pairs: list[tuple[int, int]] = []
	for left in range(num_classes - 1):
		pairs.append((left, left + 1))
	for left in range(num_classes):
		for right in range(left + 2, num_classes):
			pairs.append((left, right))
	return pairs


def build_nc_layer_specs(model: nn.Module) -> list[NCLayerSpec]:
	specs: list[NCLayerSpec] = [NCLayerSpec(name="input", mode="input", module=None)]

	feature_extractor = getattr(model, "feature_extractor", None)
	if isinstance(feature_extractor, nn.Sequential):
		first_linear: nn.Module | None = None
		first_activation: nn.Module | None = None

		for module in feature_extractor:
			if first_linear is None and isinstance(module, nn.Linear):
				first_linear = module
				continue
			if first_linear is not None and first_activation is None and isinstance(module, nn.ReLU):
				first_activation = module
				break

		if first_linear is not None:
			specs.append(NCLayerSpec(name="first_linear_pre", mode="hook_output", module=first_linear))
		if first_activation is not None:
			specs.append(NCLayerSpec(name="first_activation_post", mode="hook_output", module=first_activation))

	classifier = getattr(model, "fc", None)
	if classifier is None:
		raise ValueError("Model does not expose classifier layer as 'fc'; cannot compute NC metrics.")
	specs.append(NCLayerSpec(name="pre_classifier", mode="hook_input", module=classifier))
	return specs


def _flatten_features(x: torch.Tensor) -> torch.Tensor:
	return x.reshape(x.shape[0], -1)


def _register_layer_hooks(specs: list[NCLayerSpec], captured: dict[str, torch.Tensor]) -> list[Any]:
	handles: list[Any] = []

	for spec in specs:
		if spec.mode == "input":
			continue
		assert spec.module is not None

		if spec.mode == "hook_input":
			def _hook_input(
				_module: nn.Module,
				hook_input: tuple[torch.Tensor, ...],
				_output: torch.Tensor,
				name: str = spec.name,
			) -> None:
				captured[name] = hook_input[0].detach()

			handles.append(spec.module.register_forward_hook(_hook_input))
		elif spec.mode == "hook_output":
			def _hook_output(
				_module: nn.Module,
				_hook_input: tuple[torch.Tensor, ...],
				hook_output: torch.Tensor,
				name: str = spec.name,
			) -> None:
				captured[name] = hook_output.detach()

			handles.append(spec.module.register_forward_hook(_hook_output))
		else:
			raise ValueError(f"Unsupported hook mode '{spec.mode}'.")

	return handles


@torch.no_grad()
def collect_nc_raw_epoch(
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
	num_classes: int,
	class_pairs: list[tuple[int, int]],
	layer_specs: list[NCLayerSpec],
) -> NCEpochRaw:
	layer_names = [spec.name for spec in layer_specs]
	class_counts = torch.zeros(num_classes, device=device, dtype=torch.float64)
	layer_sums: dict[str, torch.Tensor | None] = {name: None for name in layer_names}
	layer_sqnorm_sums: dict[str, torch.Tensor] = {
		name: torch.zeros(num_classes, device=device, dtype=torch.float64) for name in layer_names
	}

	captured: dict[str, torch.Tensor] = {}
	handles = _register_layer_hooks(layer_specs, captured)

	margin_sum = 0.0
	margin_sq_sum = 0.0
	margin_count = 0

	use_non_blocking = device.type == "cuda"
	was_training = model.training
	model.eval()

	try:
		for data, target in loader:
			data = data.to(device, non_blocking=use_non_blocking)
			target = target.to(device, non_blocking=use_non_blocking)

			captured.clear()
			logits = model(data)

			true_logit = logits.gather(1, target.unsqueeze(1)).squeeze(1)
			other_logits = logits.clone()
			other_logits.scatter_(1, target.unsqueeze(1), float("-inf"))
			max_other = other_logits.max(dim=1).values
			margin = (true_logit - max_other).to(dtype=torch.float64)

			margin_sum += float(margin.sum().item())
			margin_sq_sum += float((margin * margin).sum().item())
			margin_count += int(margin.numel())

			class_counts += torch.bincount(target, minlength=num_classes).to(dtype=torch.float64)

			for spec in layer_specs:
				if spec.mode == "input":
					features = _flatten_features(data).to(dtype=torch.float64)
				else:
					feat = captured.get(spec.name)
					if feat is None:
						raise RuntimeError(f"Failed to capture activations for layer '{spec.name}'.")
					features = _flatten_features(feat).to(dtype=torch.float64)

				if layer_sums[spec.name] is None:
					layer_sums[spec.name] = torch.zeros(
						(num_classes, features.shape[1]),
						device=device,
						dtype=torch.float64,
					)

				for class_id in range(num_classes):
					mask = target == class_id
					if torch.any(mask):
						class_feats = features[mask]
						layer_sums[spec.name][class_id] += class_feats.sum(dim=0)
						layer_sqnorm_sums[spec.name][class_id] += torch.sum(class_feats * class_feats).item()

		missing = torch.nonzero(class_counts <= 0, as_tuple=False).flatten().tolist()
		if missing:
			raise ValueError(f"Cannot compute class means for classes with no samples: {missing}")

		layer_mu_sqnorm: dict[str, np.ndarray] = {}
		layer_feat_sqmean: dict[str, np.ndarray] = {}
		layer_pair_dot: dict[str, np.ndarray] = {}

		for spec in layer_specs:
			sums = layer_sums[spec.name]
			if sums is None:
				raise ValueError(f"No activations found for layer '{spec.name}'.")
			means = sums / class_counts.unsqueeze(1)
			mu_sqnorm = torch.sum(means * means, dim=1)
			feat_sqmean = layer_sqnorm_sums[spec.name] / class_counts
			gram = means @ means.T

			pair_dot = torch.tensor(
				[gram[left, right] for left, right in class_pairs],
				device=device,
				dtype=torch.float64,
			)

			layer_mu_sqnorm[spec.name] = mu_sqnorm.detach().cpu().numpy().astype(np.float64)
			layer_feat_sqmean[spec.name] = feat_sqmean.detach().cpu().numpy().astype(np.float64)
			layer_pair_dot[spec.name] = pair_dot.detach().cpu().numpy().astype(np.float64)

		return NCEpochRaw(
			class_counts=class_counts.detach().cpu().numpy().astype(np.float64),
			margin_sum=margin_sum,
			margin_sq_sum=margin_sq_sum,
			margin_count=margin_count,
			layer_mu_sqnorm=layer_mu_sqnorm,
			layer_feat_sqmean=layer_feat_sqmean,
			layer_pair_dot=layer_pair_dot,
		)
	finally:
		for handle in handles:
			handle.remove()
		if was_training:
			model.train()


def initialize_nc_csv(
	nc_csv_path: Path,
	layer_names: list[str],
	num_classes: int,
	class_pairs: list[tuple[int, int]],
) -> None:
	header = ["epoch", "global_step", "margin_sum", "margin_sq_sum", "margin_count"]
	header.extend([f"count_{class_id}" for class_id in range(num_classes)])

	for layer_name in layer_names:
		header.extend([f"{layer_name}_mu_sqnorm_{class_id}" for class_id in range(num_classes)])
		header.extend([f"{layer_name}_feat_sqmean_{class_id}" for class_id in range(num_classes)])
		header.extend([f"{layer_name}_dot_{left}_{right}" for left, right in class_pairs])

	with open(nc_csv_path, "w", encoding="utf-8", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(header)


def append_nc_csv_row(
	nc_csv_path: Path,
	epoch: int,
	global_step: int,
	raw: NCEpochRaw,
	layer_names: list[str],
	num_classes: int,
) -> None:
	row: list[float | int] = [epoch, global_step, raw.margin_sum, raw.margin_sq_sum, raw.margin_count]
	row.extend(raw.class_counts[:num_classes].tolist())

	for layer_name in layer_names:
		row.extend(raw.layer_mu_sqnorm[layer_name].tolist())
		row.extend(raw.layer_feat_sqmean[layer_name].tolist())
		row.extend(raw.layer_pair_dot[layer_name].tolist())

	with open(nc_csv_path, "a", encoding="utf-8", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(row)


def _load_nc_columns(nc_csv_path: Path) -> list[dict[str, float]]:
	rows: list[dict[str, float]] = []
	with open(nc_csv_path, "r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		for row in reader:
			casted: dict[str, float] = {}
			for key, value in row.items():
				if key in {"epoch", "global_step", "margin_count"}:
					casted[key] = float(int(value))
				else:
					casted[key] = float(value)
			rows.append(casted)
	return rows


def _plot_nc_dashboard(
	steps: np.ndarray,
	inter_mean_by_layer: dict[str, np.ndarray],
	within_std_by_layer: dict[str, np.ndarray],
	ratio_by_layer: dict[str, np.ndarray],
	cosine_mean_by_layer: dict[str, np.ndarray],
	margin_mean: np.ndarray,
	margin_std: np.ndarray,
	pre_classifier_pair_dist: np.ndarray,
	class_pairs: list[tuple[int, int]],
	num_classes: int,
	output_path: Path,
	tpt_step: int,
) -> None:
	layer_names = list(inter_mean_by_layer.keys())

	fig, axes = plt.subplots(3, 2, figsize=(18, 14), sharex=True)

	ax = axes[0, 0]
	for layer_name in layer_names:
		ax.plot(steps, inter_mean_by_layer[layer_name], linewidth=1.8, label=layer_name)
	ax.axhline(0.0, linestyle="--", color="gray", alpha=0.6, label="floor 0")
	ax.set_title("Inter-class Distance Mean (higher is better)")
	ax.set_ylabel("Mean ||mu_i - mu_j||")
	ax.grid(True, alpha=0.3)
	ax.legend(fontsize=8)

	ax = axes[0, 1]
	for layer_name in layer_names:
		ax.plot(steps, within_std_by_layer[layer_name], linewidth=1.8, label=layer_name)
	ax.axhline(0.0, linestyle="--", color="gray", alpha=0.6, label="target 0 (down)")
	ax.set_title("Within-class Spread (lower is better)")
	ax.set_ylabel("Mean class std")
	ax.grid(True, alpha=0.3)
	ax.legend(fontsize=8)

	ax = axes[1, 0]
	for layer_name in layer_names:
		ax.plot(steps, ratio_by_layer[layer_name], linewidth=1.8, label=layer_name)
	ax.set_title("Separation Ratio = inter / within (higher is better)")
	ax.set_ylabel("Ratio")
	ax.grid(True, alpha=0.3)
	ax.legend(fontsize=8)

	ax = axes[1, 1]
	for layer_name in layer_names:
		ax.plot(steps, cosine_mean_by_layer[layer_name], linewidth=1.8, label=layer_name)
	if num_classes > 1:
		target_cos = -1.0 / float(num_classes - 1)
		ax.axhline(target_cos, linestyle="--", color="gray", alpha=0.7, label=f"ETF target {target_cos:.3f}")
	ax.set_title("Mean Pairwise Cosine Between Class Means")
	ax.set_ylabel("Mean cosine")
	ax.grid(True, alpha=0.3)
	ax.legend(fontsize=8)

	ax = axes[2, 0]
	ax.plot(steps, margin_mean, linewidth=1.8, color="tab:purple", label="mean margin")
	ax.fill_between(
		steps,
		margin_mean - margin_std,
		margin_mean + margin_std,
		color="tab:purple",
		alpha=0.2,
		label="±1 std",
	)
	ax.axhline(0.0, linestyle="--", color="gray", alpha=0.6, label="decision boundary")
	ax.set_title("Logit Margin (higher is better)")
	ax.set_xlabel("Global Step")
	ax.set_ylabel("Margin")
	ax.grid(True, alpha=0.3)
	ax.legend(fontsize=8)

	ax = axes[2, 1]
	for pair_idx, (left, right) in enumerate(class_pairs):
		ax.plot(
			steps,
			pre_classifier_pair_dist[:, pair_idx],
			linewidth=1.4,
			label=f"||mu_{left} - mu_{right}||",
		)
	ax.set_title("Pre-classifier Pairwise Distances")
	ax.set_xlabel("Global Step")
	ax.set_ylabel("Distance")
	ax.grid(True, alpha=0.3)
	if len(class_pairs) <= 16:
		ax.legend(fontsize=7)

	if tpt_step >= 0:
		for axis in axes.flat:
			axis.axvline(tpt_step, color="black", linestyle="-", linewidth=2.0)

	fig.suptitle("Neural Collapse Metrics", fontsize=16)
	plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
	output_path.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(output_path, dpi=180, bbox_inches="tight")
	plt.close(fig)


def finalize_nc_metrics(
	nc_csv_path: Path,
	output_path: Path,
	layer_names: list[str],
	num_classes: int,
	class_pairs: list[tuple[int, int]],
	tpt_step: int = -1,
) -> None:
	rows = _load_nc_columns(nc_csv_path)
	if not rows:
		raise ValueError(f"No NC rows found in {nc_csv_path}")

	steps = np.asarray([int(row["global_step"]) for row in rows], dtype=np.int64)
	margin_sum = np.asarray([row["margin_sum"] for row in rows], dtype=np.float64)
	margin_sq_sum = np.asarray([row["margin_sq_sum"] for row in rows], dtype=np.float64)
	margin_count = np.asarray([max(1.0, row["margin_count"]) for row in rows], dtype=np.float64)
	margin_mean = margin_sum / margin_count
	margin_var = np.clip(margin_sq_sum / margin_count - margin_mean * margin_mean, 0.0, None)
	margin_std = np.sqrt(margin_var)

	pair_left = np.asarray([left for left, _ in class_pairs], dtype=np.int64)
	pair_right = np.asarray([right for _, right in class_pairs], dtype=np.int64)

	inter_mean_by_layer: dict[str, np.ndarray] = {}
	within_std_by_layer: dict[str, np.ndarray] = {}
	ratio_by_layer: dict[str, np.ndarray] = {}
	cosine_mean_by_layer: dict[str, np.ndarray] = {}
	pre_classifier_pair_dist = np.zeros((len(rows), len(class_pairs)), dtype=np.float64)

	for layer_name in layer_names:
		mu_sqnorm = np.asarray(
			[[row[f"{layer_name}_mu_sqnorm_{class_id}"] for class_id in range(num_classes)] for row in rows],
			dtype=np.float64,
		)
		feat_sqmean = np.asarray(
			[[row[f"{layer_name}_feat_sqmean_{class_id}"] for class_id in range(num_classes)] for row in rows],
			dtype=np.float64,
		)
		pair_dot = np.asarray(
			[[row[f"{layer_name}_dot_{left}_{right}"] for left, right in class_pairs] for row in rows],
			dtype=np.float64,
		)

		pair_dist = np.sqrt(
			np.clip(mu_sqnorm[:, pair_left] + mu_sqnorm[:, pair_right] - 2.0 * pair_dot, 0.0, None)
		)
		denom = np.sqrt(np.clip(mu_sqnorm[:, pair_left] * mu_sqnorm[:, pair_right], EPS, None))
		pair_cos = pair_dot / denom

		within_var = np.clip(feat_sqmean - mu_sqnorm, 0.0, None)
		within_std = np.sqrt(within_var)

		inter_mean = np.mean(pair_dist, axis=1)
		within_mean = np.mean(within_std, axis=1)
		separation_ratio = inter_mean / (within_mean + EPS)
		cosine_mean = np.mean(pair_cos, axis=1)

		inter_mean_by_layer[layer_name] = inter_mean
		within_std_by_layer[layer_name] = within_mean
		ratio_by_layer[layer_name] = separation_ratio
		cosine_mean_by_layer[layer_name] = cosine_mean

		if layer_name == "pre_classifier":
			pre_classifier_pair_dist = pair_dist

	_plot_nc_dashboard(
		steps=steps,
		inter_mean_by_layer=inter_mean_by_layer,
		within_std_by_layer=within_std_by_layer,
		ratio_by_layer=ratio_by_layer,
		cosine_mean_by_layer=cosine_mean_by_layer,
		margin_mean=margin_mean,
		margin_std=margin_std,
		pre_classifier_pair_dist=pre_classifier_pair_dist,
		class_pairs=class_pairs,
		num_classes=num_classes,
		output_path=output_path,
		tpt_step=tpt_step,
	)
