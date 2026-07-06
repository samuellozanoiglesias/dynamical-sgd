"""
model_cnn.py

Model creation + training ported EXACTLY (math-wise) from neuralcollapse_(1).py's
model definition, hook, and train() function -- running on real PyTorch/torchvision.

Only the data-feeding is adapted: instead of a torchvision DataLoader over its
own downloaded MNIST, train_epoch_cnn() iterates the numpy arrays your
generate_dataset.py already produces (dataset_bundle.train_inputs / train_targets),
in fixed-size minibatches, matching the batching convention already used
elsewhere in training_runner.py (_iterate_minibatches).

Metrics are intentionally NOT reimplemented here. Instead, TorchModelAdapter
below duck-types the JAXModel interface (apply / classifier_weight_matrix /
classifier_bias_vector) so training_runner.py can pass this model straight
into collect_nc_raw_epoch, collect_sep_raw_epoch, collect_classifier_epoch,
the PCA/geo/hyperplane/proj_nc collectors, etc. with zero changes to those
files.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Exact model + hook from neuralcollapse_(1).py
# ---------------------------------------------------------------------------

class _FeatureCapture:
    """Equivalent of the `features` class in neuralcollapse_(1).py."""
    value: torch.Tensor | None = None


def _make_hook(capture: "_FeatureCapture"):
    def hook(self, input, output):
        # identical to: features.value = input[0].clone()
        capture.value = input[0].clone()
    return hook


def build_cnn_model(num_classes: int, input_ch: int, device: torch.device):
    """Unmodified from neuralcollapse_(1).py:

        model = models.resnet18(pretrained=False, num_classes=C)
        model.conv1 = nn.Conv2d(input_ch, model.conv1.weight.shape[0], 3, 1, 1, bias=False)
        model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        model = model.to(device)
        classifier = model.fc
        classifier.register_forward_hook(hook)
    """
    model = models.resnet18(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(input_ch, model.conv1.weight.shape[0], 3, 1, 1, bias=False)
    model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
    model = model.to(device)

    capture = _FeatureCapture()
    classifier = model.fc
    classifier.register_forward_hook(_make_hook(capture))
    return model, classifier, capture


def make_optimizer_and_scheduler(
    model: nn.Module, lr: float, momentum: float, weight_decay: float, epochs: int
):
    """Unmodified from neuralcollapse_(1).py:

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[e//3, e*2//3], gamma=0.1)
    """
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    milestones = [epochs // 3, epochs * 2 // 3]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    return optimizer, scheduler


def make_criterion(loss_name: str):
    if loss_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    if loss_name == "MSELoss":
        return nn.MSELoss()
    raise ValueError(f"Unsupported loss_name '{loss_name}'.")


def _iterate_minibatches(inputs: np.ndarray, targets: np.ndarray, batch_size: int):
    num_samples = int(inputs.shape[0])
    for start in range(0, num_samples, batch_size):
        end = min(num_samples, start + batch_size)
        yield inputs[start:end], targets[start:end]


# ---------------------------------------------------------------------------
# Bump sampling helpers -- copied verbatim (pure numpy, no jax dependency)
# from training_runner.py's _compute_focus_weight / _compute_class_probabilities /
# _sample_batch_by_class_counts, so the CNN path samples batches with the
# exact same bump dynamics as the JAX path. Not reimplemented -- identical math.
# ---------------------------------------------------------------------------

def _compute_focus_weight(period_step: int, period_length: int, w_max: float) -> tuple[float, float]:
    t = float(period_step)
    period = float(period_length)
    slope = 2.0 * (w_max - 1.0) / period
    if t < period / 2.0:
        focus_weight = 1.0 + t * slope
    else:
        focus_weight = 2.0 * w_max - t * slope - 1.0
    phase = t / period
    return focus_weight, phase


def _compute_class_probabilities(
    step: int,
    num_classes: int,
    period_length: int,
    w_max: float,
    bump_order: list[int] | None = None,
) -> tuple[np.ndarray, int, float, float]:
    if bump_order is None:
        focus_class = (step // period_length) % num_classes
    else:
        cycle_index = (step // period_length) % len(bump_order)
        focus_class = int(bump_order[cycle_index])
    period_step = step % period_length
    focus_weight, bump_phase = _compute_focus_weight(period_step, period_length, w_max)

    weights = np.ones(num_classes, dtype=np.float64)
    weights[focus_class] = focus_weight
    probabilities = weights / weights.sum()
    return probabilities, int(focus_class), float(focus_weight), float(bump_phase)


def _sample_batch_by_class_counts(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    class_to_indices: dict,
    class_counts: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    class_counts_int = np.asarray(class_counts, dtype=np.int64)
    sampled_index_parts: list[np.ndarray] = []
    for class_id, count in enumerate(class_counts_int):
        if count <= 0:
            continue
        class_indices = class_to_indices[class_id]
        chosen = rng.choice(class_indices, size=int(count), replace=True)
        sampled_index_parts.append(chosen.astype(np.int64, copy=False))

    if not sampled_index_parts:
        raise ValueError(
            "Deterministic class_counts produced an empty batch. "
            "Increase training.batch_size or adjust dynamics settings."
        )

    sampled_indices = np.concatenate(sampled_index_parts, axis=0)
    return train_inputs[sampled_indices], train_targets[sampled_indices], class_counts_int

class _EpochShuffler:
    """Mimics DataLoader(shuffle=True): reshuffles indices each epoch,
    draws sequential batches without replacement, drops the final partial batch."""
    def __init__(self, num_samples: int, batch_size: int, rng: np.random.Generator):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.rng = rng
        self._order = self.rng.permutation(num_samples)
        self._pos = 0

    def next_batch(self) -> np.ndarray:
        if self._pos + self.batch_size > self.num_samples:
            self._order = self.rng.permutation(self.num_samples)  # new epoch
            self._pos = 0
        idx = self._order[self._pos:self._pos + self.batch_size]
        self._pos += self.batch_size
        return idx

def train_epoch_cnn(
    model: nn.Module,
    criterion,
    device: torch.device,
    num_classes: int,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    class_to_indices: dict,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    steps_this_epoch: int,
    total_steps: int,
    bumps_enabled: bool,
    period_length: int,
    w_max: float,
    bump_order: list[int] | None,
    rng: np.random.Generator,
    shuffler: "_EpochShuffler",
) -> dict:
    """Same forward/backward/optimizer.step math as train() in
    neuralcollapse_(1).py. What changed vs. the previous version: instead of
    one pass over the dataset in fixed order, this runs `steps_this_epoch`
    steps, each pulling a batch via the same class-count bump sampling used
    by the JAX train_epoch -- so bumps-before/after-tpt behave identically
    on both paths. No freeze logic (not requested for this path)."""
    model.train()
    loss_sum = 0.0
    correct_sum = 0
    count_sum = 0
    sampled_distributions: list[np.ndarray] = []
    bump_state = {"active": False, "focus_class": -1, "focus_weight": 1.0, "phase": 0.0}

    for _ in range(steps_this_epoch):
        in_uniform_tail = global_step >= (0.95 * float(total_steps))
        if bumps_enabled and not in_uniform_tail:
            class_probs, focus_class, focus_weight, bump_phase = _compute_class_probabilities(
                step=global_step,
                num_classes=num_classes,
                period_length=period_length,
                w_max=w_max,
                bump_order=bump_order,
            )
            bump_state = {
                "active": True,
                "focus_class": focus_class,
                "focus_weight": focus_weight,
                "phase": bump_phase,
            }
            class_counts = (class_probs * float(batch_size)).astype(np.int64)
            batch_data, batch_target, sampled_class_counts = _sample_batch_by_class_counts(
                train_inputs=train_inputs,
                train_targets=train_targets,
                class_to_indices=class_to_indices,
                class_counts=class_counts,
                rng=rng,
            )
        else:
            idx = shuffler.next_batch()
            batch_data, batch_target = train_inputs[idx], train_targets[idx]
            sampled_class_counts = np.bincount(batch_target, minlength=num_classes)
        
        sampled_total = int(sampled_class_counts.sum())
        sampled_distributions.append(
            sampled_class_counts.astype(np.float64) / float(max(1, sampled_total))
        )

        data = torch.as_tensor(batch_data, dtype=torch.float32, device=device)
        target = torch.as_tensor(batch_target, dtype=torch.long, device=device)

        optimizer.zero_grad()
        out = model(data)
        if isinstance(criterion, nn.CrossEntropyLoss):
            loss = criterion(out, target)
        elif isinstance(criterion, nn.MSELoss):
            loss = criterion(out, F.one_hot(target, num_classes=num_classes).float())
        else:
            raise ValueError("Unsupported criterion.")

        loss.backward()
        optimizer.step()

        preds = torch.argmax(out, dim=1)
        correct_sum += int((preds == target).sum().item())
        loss_sum += float(loss.item()) * data.shape[0]
        count_sum += int(data.shape[0])
        global_step += 1

    if sampled_distributions:
        step_distributions = np.stack(sampled_distributions, axis=0)
    else:
        step_distributions = np.zeros((0, num_classes), dtype=np.float64)

    return {
        "loss": loss_sum / max(1, count_sum),
        "accuracy": correct_sum / max(1, count_sum),
        "global_step": global_step,
        "bump_state": bump_state,
        "step_distributions": step_distributions,
    }


# ---------------------------------------------------------------------------
# Adapter: makes the PyTorch model usable by your EXISTING metric functions
# ---------------------------------------------------------------------------

class TorchModelAdapter:
    """Duck-types the JAXModel interface used everywhere in training_runner.py
    and in neural_collapse.py / separability_measures.py / classifier_metrics.py /
    PCA_analysis.py / PCA_geometric_overlapping.py / hyperplanes.py /
    projection_PCA_analysis.py:

        model.apply(params, x, return_intermediates=True) -> (logits, {"pre_classifier": ...})
        model.classifier_weight_matrix(params) -> [C, D] array
        model.classifier_bias_vector(params)   -> [C] array or None

    `params` is accepted everywhere but ignored -- the torch model already
    owns its weights. This means every existing metrics call site keeps
    working with zero changes to those files; just pass params=None.
    """

    def __init__(
        self,
        torch_model: nn.Module,
        classifier: nn.Module,
        capture: _FeatureCapture,
        device: torch.device,
    ):
        self.torch_model = torch_model
        self.classifier = classifier
        self.capture = capture
        self.device = device

    def apply(self, params, x, *, return_intermediates: bool = False):
        self.torch_model.eval()
        x_np = np.asarray(x, dtype=np.float32)
        x_t = torch.as_tensor(x_np, device=self.device)
        with torch.no_grad():
            logits_t = self.torch_model(x_t)
        logits = jnp.asarray(logits_t.detach().cpu().numpy())

        if not return_intermediates:
            return logits

        pre_classifier_t = self.capture.value.reshape(x_t.shape[0], -1)
        pre_classifier = jnp.asarray(pre_classifier_t.detach().cpu().numpy())
        return logits, {"pre_classifier": pre_classifier}

    def classifier_weight_matrix(self, params):
        # nn.Linear.weight is already [C, D] (out_features, in_features) --
        # no transpose needed, unlike the JAX version which stores [D, C].
        return jnp.asarray(self.classifier.weight.detach().cpu().numpy())

    def classifier_bias_vector(self, params):
        if self.classifier.bias is None:
            return None
        return jnp.asarray(self.classifier.bias.detach().cpu().numpy())