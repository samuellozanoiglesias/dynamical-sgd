"""Diagnostics aimed at explaining *why* training gets stuck, not just *that*
it is stuck.

Everything here is self-contained (only needs jax/numpy/optax/matplotlib and
the JAXModel/ParamTree types from model.py) so it can be dropped next to
training_runner.py without touching any of the other analysis modules.

Four diagnostics, computed on a fixed (non-minibatched) subset of the data at
each recorded checkpoint:

1. sharpness_top_eig
   Dominant eigenvalue of the loss Hessian (full-batch, on the metric
   subset), estimated with a few steps of power iteration using
   Hessian-vector products (no Hessian is ever formed explicitly).
   Sign matters: strongly negative => you are sitting on a saddle (a bump
   only has to nudge you off it); small/near-zero positive => the region is
   genuinely flat, not a saddle.

2. class_grad_conflict_mean / class_grad_conflict_min
   Pairwise cosine similarity between the *per-class* gradients (the
   gradient of the loss restricted to each class's own points). Strongly
   negative values mean different classes are pulling the shared weights in
   opposite directions -- the small net gradient you'd see in a plain
   gradient-norm plot can be the sum of several large, cancelling forces
   rather than genuine flatness. This is the "tug of war" hypothesis.

3. grad_cosine_persistence
   Cosine similarity between the full-batch gradient at the start of the
   window and at the end of the window. Near 1: smooth, consistent descent
   (just slow). Near 0 or negative: the gradient direction is oscillating in
   place across the window, a signature of a ravine/saddle rather than slow
   progress toward a minimum.

4. gradient_noise_scale
   How much different minibatches (drawn the ordinary, un-bumped way)
   disagree with each other in gradient direction, relative to the
   magnitude of the full-batch gradient. Cross-reference with (1)-(3): a
   high noise scale means minibatch stochasticity alone could plausibly
   already be jostling the run around; a low noise scale means the
   stochastic-gradient noise floor is small and something else (a genuine
   saddle/flat region) is responsible for getting stuck.

5. displacement_norm / grad_displacement_cosine / effective_step_ratio
   How far the parameters actually moved over the window, and whether that
   motion was aligned with the (start-of-window) gradient direction. Adam's
   per-parameter adaptive scaling can make the *raw* gradient look
   healthy while the *actual* step taken stays tiny and/or drifts away from
   the gradient direction -- this isolates that possibility from genuine
   landscape flatness.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from model import JAXModel, ParamTree


# --------------------------------------------------------------------------
# Small pytree helpers (kept local so this file has no cross-module deps
# beyond model.py).
# --------------------------------------------------------------------------

def _tree_dot(a: ParamTree, b: ParamTree) -> jax.Array:
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    return sum(jnp.vdot(x, y) for x, y in zip(leaves_a, leaves_b))


def _tree_norm(a: ParamTree) -> jax.Array:
    return jnp.sqrt(jnp.clip(_tree_dot(a, a), 0.0))


def _tree_scale(a: ParamTree, s: jax.Array | float) -> ParamTree:
    return jax.tree_util.tree_map(lambda x: x * s, a)


def _tree_sub(a: ParamTree, b: ParamTree) -> ParamTree:
    return jax.tree_util.tree_map(lambda x, y: x - y, a, b)


def _flatten(a: ParamTree) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(a)
    return jnp.concatenate([jnp.reshape(x, (-1,)) for x in leaves])


def _cosine(a: ParamTree, b: ParamTree) -> float:
    denom = _tree_norm(a) * _tree_norm(b)
    cos = jnp.where(denom > 1e-12, _tree_dot(a, b) / denom, 0.0)
    return float(cos)


def _make_loss_fn(model: JAXModel, x: jax.Array, y: jax.Array):
    def loss_fn(params: ParamTree) -> jax.Array:
        logits = model.apply(params, x)
        per_sample = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        return jnp.mean(per_sample)

    return loss_fn


def _hvp(loss_fn, params: ParamTree, v: ParamTree) -> ParamTree:
    """Hessian-vector product H @ v via reverse-over-reverse autodiff."""
    return jax.grad(lambda p: _tree_dot(jax.grad(loss_fn)(p), v))(params)


# --------------------------------------------------------------------------
# Diagnostic 1: sharpness via power iteration.
# --------------------------------------------------------------------------

def estimate_top_eigenvalue(
    loss_fn,
    params: ParamTree,
    num_iters: int,
    key: jax.Array,
) -> float:
    v = jax.tree_util.tree_map(
        lambda x: jax.random.normal(key, x.shape, dtype=jnp.float32), params
    )
    v_norm = _tree_norm(v)
    v = _tree_scale(v, 1.0 / jnp.maximum(v_norm, 1e-12))

    eigenvalue = 0.0
    for _ in range(max(1, num_iters)):
        hv = _hvp(loss_fn, params, v)
        eigenvalue = float(_tree_dot(v, hv))
        hv_norm = _tree_norm(hv)
        if float(hv_norm) < 1e-12:
            break
        v = _tree_scale(hv, 1.0 / hv_norm)
    return eigenvalue


# --------------------------------------------------------------------------
# Diagnostic 2: per-class gradient conflict.
# --------------------------------------------------------------------------

def _per_class_gradients(
    model: JAXModel,
    params: ParamTree,
    inputs: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    device: jax.Device,
) -> list[ParamTree | None]:
    grads: list[ParamTree | None] = []
    for class_id in range(num_classes):
        mask = targets == class_id
        if not np.any(mask):
            grads.append(None)
            continue
        xc = jax.device_put(jnp.asarray(inputs[mask], dtype=jnp.float32), device=device)
        yc = jax.device_put(jnp.asarray(targets[mask], dtype=jnp.int32), device=device)
        loss_fn = _make_loss_fn(model, xc, yc)
        grads.append(jax.grad(loss_fn)(params))
    return grads


def _class_gradient_conflict(grads_per_class: list[ParamTree | None]) -> tuple[float, float]:
    flat = [_flatten(g) for g in grads_per_class if g is not None]
    if len(flat) < 2:
        return float("nan"), float("nan")
    cos_sims: list[float] = []
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            num = jnp.dot(flat[i], flat[j])
            den = jnp.linalg.norm(flat[i]) * jnp.linalg.norm(flat[j])
            cos = float(jnp.where(den > 1e-12, num / den, 0.0))
            cos_sims.append(cos)
    return float(np.mean(cos_sims)), float(np.min(cos_sims))


# --------------------------------------------------------------------------
# Diagnostic 4: gradient noise scale across ordinary (un-bumped) minibatches.
# --------------------------------------------------------------------------

def _estimate_gradient_noise_scale(
    model: JAXModel,
    params: ParamTree,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    batch_size: int,
    num_minibatches: int,
    rng: np.random.Generator,
    device: jax.Device,
) -> float:
    num_samples = train_inputs.shape[0]
    batch_grads: list[jax.Array] = []
    for _ in range(num_minibatches):
        idx = rng.choice(num_samples, size=min(batch_size, num_samples), replace=False)
        xb = jax.device_put(jnp.asarray(train_inputs[idx], dtype=jnp.float32), device=device)
        yb = jax.device_put(jnp.asarray(train_targets[idx], dtype=jnp.int32), device=device)
        loss_fn = _make_loss_fn(model, xb, yb)
        batch_grads.append(_flatten(jax.grad(loss_fn)(params)))

    grads_arr = jnp.stack(batch_grads, axis=0)  # [num_minibatches, num_params]
    mean_grad = jnp.mean(grads_arr, axis=0)
    var = jnp.mean(jnp.sum((grads_arr - mean_grad) ** 2, axis=1))
    denom = jnp.sum(mean_grad ** 2)
    noise_scale = jnp.where(denom > 1e-12, var / denom, 0.0)
    return float(noise_scale)


# --------------------------------------------------------------------------
# Public entry point: one diagnostics snapshot for a training window.
# --------------------------------------------------------------------------

def collect_deep_epoch(
    model: JAXModel,
    params_before: ParamTree,
    params_after: ParamTree,
    metric_inputs: np.ndarray,
    metric_targets: np.ndarray,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    num_classes: int,
    batch_size: int,
    rng: np.random.Generator,
    device: jax.Device,
    power_iters: int = 8,
    gns_minibatches: int = 5,
    hess_key: jax.Array | None = None,
) -> Dict[str, Any]:
    x_full = jax.device_put(jnp.asarray(metric_inputs, dtype=jnp.float32), device=device)
    y_full = jax.device_put(jnp.asarray(metric_targets, dtype=jnp.int32), device=device)
    loss_fn_before = _make_loss_fn(model, x_full, y_full)
    loss_fn_after = _make_loss_fn(model, x_full, y_full)

    grad_before = jax.grad(loss_fn_before)(params_before)
    grad_after = jax.grad(loss_fn_after)(params_after)

    grad_norm_full = float(_tree_norm(grad_before))
    grad_cosine_persistence = _cosine(grad_before, grad_after)

    key = hess_key if hess_key is not None else jax.random.PRNGKey(0)
    sharpness_top_eig = estimate_top_eigenvalue(loss_fn_before, params_before, power_iters, key)

    grads_per_class = _per_class_gradients(
        model, params_before, metric_inputs, metric_targets, num_classes, device
    )
    class_grad_conflict_mean, class_grad_conflict_min = _class_gradient_conflict(grads_per_class)

    gradient_noise_scale = _estimate_gradient_noise_scale(
        model, params_before, train_inputs, train_targets, batch_size, gns_minibatches, rng, device
    )

    displacement = _tree_sub(params_after, params_before)
    displacement_norm = float(_tree_norm(displacement))
    grad_displacement_cosine = _cosine(grad_before, displacement)
    effective_step_ratio = (
        displacement_norm / grad_norm_full if grad_norm_full > 1e-12 else float("nan")
    )

    return {
        "grad_norm_full": grad_norm_full,
        "sharpness_top_eig": sharpness_top_eig,
        "class_grad_conflict_mean": class_grad_conflict_mean,
        "class_grad_conflict_min": class_grad_conflict_min,
        "grad_cosine_persistence": grad_cosine_persistence,
        "gradient_noise_scale": gradient_noise_scale,
        "displacement_norm": displacement_norm,
        "grad_displacement_cosine": grad_displacement_cosine,
        "effective_step_ratio": effective_step_ratio,
    }


_DEEP_FIELDS = [
    "grad_norm_full",
    "sharpness_top_eig",
    "class_grad_conflict_mean",
    "class_grad_conflict_min",
    "grad_cosine_persistence",
    "gradient_noise_scale",
    "displacement_norm",
    "grad_displacement_cosine",
    "effective_step_ratio",
]


def initialize_deep_csv(csv_path: Path) -> None:
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "global_step", *_DEEP_FIELDS])


def append_deep_csv_row(
    csv_path: Path,
    epoch: int,
    global_step: int,
    raw: Dict[str, Any],
) -> None:
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, global_step, *(raw[field] for field in _DEEP_FIELDS)])


def finalize_deep_plots(
    csv_path: Path,
    output_path: Path,
    tpt_step: int = -1,
    title: str = "",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[dict[str, float]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})

    if not rows:
        return

    steps = [r["global_step"] for r in rows]

    fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True)
    ax = axes.ravel()

    ax[0].plot(steps, [r["grad_norm_full"] for r in rows], color="tab:blue")
    ax[0].set_title("Full-batch gradient norm")
    ax[0].set_yscale("log")

    ax[1].plot(steps, [r["sharpness_top_eig"] for r in rows], color="tab:red")
    ax[1].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax[1].set_title("Sharpness: top Hessian eigenvalue\n(negative = saddle, ~0 = flat)")

    ax[2].plot(steps, [r["class_grad_conflict_mean"] for r in rows], label="mean", color="tab:purple")
    ax[2].plot(steps, [r["class_grad_conflict_min"] for r in rows], label="min", color="tab:purple", alpha=0.4)
    ax[2].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax[2].set_title("Per-class gradient conflict\n(negative = classes pulling against each other)")
    ax[2].legend(fontsize=8)

    ax[3].plot(steps, [r["grad_cosine_persistence"] for r in rows], color="tab:green")
    ax[3].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax[3].set_ylim(-1.05, 1.05)
    ax[3].set_title("Gradient direction persistence\nover window (1 = smooth descent, ~0 = oscillating)")

    ax[4].plot(steps, [r["gradient_noise_scale"] for r in rows], color="tab:orange")
    ax[4].set_title("Minibatch gradient noise scale")
    ax[4].set_yscale("log")

    ax[5].plot(steps, [r["effective_step_ratio"] for r in rows], label="displacement / grad norm", color="tab:brown")
    ax5b = ax[5].twinx()
    ax5b.plot(steps, [r["grad_displacement_cosine"] for r in rows], label="cos(grad, displacement)", color="tab:cyan", alpha=0.6)
    ax5b.set_ylim(-1.05, 1.05)
    ax[5].set_title("Effective step size & alignment")
    lines1, labels1 = ax[5].get_legend_handles_labels()
    lines2, labels2 = ax5b.get_legend_handles_labels()
    ax[5].legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right")

    for a in ax:
        a.set_xlabel("global step")
        if tpt_step is not None and tpt_step >= 0:
            a.axvline(tpt_step, color="gray", linewidth=0.8, linestyle=":")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
