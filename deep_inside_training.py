"""Diagnostics aimed at explaining *why* training gets stuck, not just *that*
it is stuck.

Everything here is self-contained (only needs jax/optax/numpy/matplotlib and
the JAXModel/ParamTree types from model.py) so it can be dropped next to
training_runner.py without touching any of the other analysis modules.

--- Landscape / curvature diagnostics -------------------------------------

1. sharpness_top_eig
   Dominant eigenvalue of the loss Hessian (full-batch, on the metric
   subset), via power iteration on Hessian-vector products (no Hessian is
   ever formed explicitly).

2. eos_ratio
   learning_rate * sharpness_top_eig. For plain gradient descent, a
   quadratic direction with curvature lambda is only stable if
   lr * lambda < 2. IMPORTANT: this classical threshold is derived for raw
   GD and is often a poor fit for Adam, whose actual per-parameter step is
   rescaled by 1/(sqrt(v)+eps). A run can show eos_ratio far below 2 while
   still being unstable in the coordinate system Adam actually steps in --
   see (3)/(4).

3. preconditioned_sharpness_top_eig / preconditioned_eos_ratio
   The same power iteration, but on the Adam-preconditioned operator
   P^(1/2) H P^(1/2) where P = diag(1/(sqrt(v)+eps)) and v is Adam's live
   second-moment estimate (pulled straight out of opt_state). This is the
   curvature Adam's own update actually "feels" in a given direction, and
   is the theoretically appropriate analogue of sharpness for adaptive
   optimizers (c.f. "Adaptive Gradient Methods at the Edge of Stability").
   preconditioned_eos_ratio = learning_rate * preconditioned_sharpness_top_eig;
   again treat 2 as a rough reference, not an exact threshold, but this is
   the more honest of the two ratios for an Adam run. NaN if the optimizer
   isn't Adam (no usable second-moment state was found).

4. curvature_along_displacement / cosine_top_eigvec_displacement
   Curvature specifically along the direction the optimizer actually moved
   this window, and how aligned that displacement is with the dominant
   (raw) eigenvector. Low alignment + a persistently climbing eigenvalue
   (see 5, 6) suggests the sharp direction is not being driven window by
   window, but by a slow *cumulative* drift -- see 6.

5. eigvec_persistence_cosine
   |cosine| between this checkpoint's top eigenvector and the previous
   recorded checkpoint's. High and stable = the *same* direction persists
   across a large span of training.

6. cumulative_drift_cosine
   |cosine| between the top eigenvector and the TOTAL parameter
   displacement since initialization (not just this window's step). This
   distinguishes "the sharp direction is where the whole run has been
   slowly marching the entire time" (high, stable) from "the sharp
   direction is unrelated to where training has actually gone" (low).

7. eigvec_energy_hidden_frac / eigvec_energy_classifier_class_{i}
   Decomposes the top eigenvector's squared norm by where it structurally
   lives: fraction in the hidden layer(s) vs. fraction in each class's own
   row of the output layer. Large concentration in one class's output
   weights turns "the landscape is sharp" into "class i's decision
   boundary is the structural culprit."

--- Class-conflict diagnostics ---------------------------------------------

8. class_grad_conflict_mean / class_grad_conflict_min
   Pairwise cosine similarity between the *per-class* gradients.

9. class_grad_norm_class_{i}
   L2 norm of each class's own gradient.

--- Direct behavioral diagnostics -------------------------------------------

10. pred_churn_overall / pred_churn_class_{i}
    Fraction of metric-set points whose predicted class flips between the
    start and end of the window.

11. low_margin_persistence_jaccard / low_margin_group_accuracy /
    low_margin_relative_margin
    Track the bottom-quantile of points by logit margin (top-1 minus
    top-2 logit) each window. Jaccard overlap with the previous window's
    bottom-quantile set tells you if it is literally the SAME points stuck
    near the boundary every time (vs. different points rotating through).
    Group accuracy tells you if that persistent group is at least
    resolving over time. Relative margin (this group's mean margin
    divided by the whole set's median margin) controls for the fact that
    logit scale generally grows over training -- if this ratio never
    grows, the group is not just numerically low-margin, it is
    structurally and persistently stuck relative to everyone else.

--- Baseline gradient-noise diagnostics -------------------------------------

12. grad_cosine_persistence, gradient_noise_scale, displacement_norm,
    grad_displacement_cosine, effective_step_ratio
    See previous revision of this module for definitions; unchanged.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from model import JAXModel, ParamTree


# --------------------------------------------------------------------------
# Small pytree helpers.
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


def _tree_mul(a: ParamTree, b: ParamTree) -> ParamTree:
    return jax.tree_util.tree_map(lambda x, y: x * y, a, b)


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
# Generic power iteration on any symmetric matvec operator over the params
# pytree. Used for both the raw Hessian and the Adam-preconditioned one.
# --------------------------------------------------------------------------

def _power_iteration(matvec, template: ParamTree, num_iters: int, key: jax.Array) -> tuple[float, ParamTree]:
    v = jax.tree_util.tree_map(
        lambda x: jax.random.normal(key, x.shape, dtype=jnp.float32), template
    )
    v_norm = _tree_norm(v)
    v = _tree_scale(v, 1.0 / jnp.maximum(v_norm, 1e-12))

    eigenvalue = 0.0
    for _ in range(max(1, num_iters)):
        w = matvec(v)
        eigenvalue = float(_tree_dot(v, w))
        w_norm = _tree_norm(w)
        if float(w_norm) < 1e-12:
            break
        v = _tree_scale(w, 1.0 / w_norm)
    return eigenvalue, v


def estimate_top_eigenpair(
    loss_fn, params: ParamTree, num_iters: int, key: jax.Array
) -> tuple[float, ParamTree]:
    return _power_iteration(lambda v: _hvp(loss_fn, params, v), params, num_iters, key)


def estimate_preconditioned_top_eigenpair(
    loss_fn,
    params: ParamTree,
    precond_sqrt: ParamTree,
    num_iters: int,
    key: jax.Array,
) -> tuple[float, ParamTree]:
    """Top eigenpair of P^(1/2) H P^(1/2), where precond_sqrt = P^(1/2)."""

    def matvec(v: ParamTree) -> ParamTree:
        w = _tree_mul(precond_sqrt, v)
        hw = _hvp(loss_fn, params, w)
        return _tree_mul(precond_sqrt, hw)

    return _power_iteration(matvec, params, num_iters, key)


def _find_adam_state(opt_state: Any) -> Optional[tuple[ParamTree, Any]]:
    """Recursively search an (possibly optax.chain-wrapped) opt_state for an
    Adam-like sub-state (has `mu`, `nu`, and `count`) and return its
    (nu, count). Returns None if none is found, e.g. for SGD.
    """
    if hasattr(opt_state, "nu") and hasattr(opt_state, "mu") and hasattr(opt_state, "count"):
        return opt_state.nu, opt_state.count
    if isinstance(opt_state, (tuple, list)):
        for sub in opt_state:
            found = _find_adam_state(sub)
            if found is not None:
                return found
    return None


def _find_adam_second_moment(opt_state: Any, beta2: float) -> Optional[ParamTree]:
    """Bias-corrected Adam second-moment estimate (nu_hat), or None if the
    optimizer isn't Adam. Uses the exact same bias correction Adam itself
    applies (nu_hat = nu / (1 - beta2**count)) -- the raw, uncorrected nu
    would wildly overestimate curvature in the first few hundred steps,
    since nu starts at (and warms up slowly from) zero. Even bias-corrected,
    treat the first handful of checkpoints as noisy/unreliable -- one or two
    gradient observations is not yet a meaningful curvature estimate. The
    trend over the bulk of training is what matters.
    """
    adam_state = _find_adam_state(opt_state)
    if adam_state is None:
        return None
    nu, count = adam_state
    if int(count) <= 0:
        # True initial optimizer state, before any gradient has been seen --
        # nu is uniformly zero and the preconditioner is undefined. Skip
        # rather than reporting a meaningless huge number.
        return None
    bias_correction2 = 1.0 - jnp.power(beta2, count)
    return jax.tree_util.tree_map(lambda v: v / bias_correction2, nu)


def _adam_precond_sqrt(nu: ParamTree, eps: float) -> ParamTree:
    """P^(1/2) where P = diag(1 / (sqrt(nu) + eps))."""
    return jax.tree_util.tree_map(lambda v: 1.0 / jnp.sqrt(jnp.sqrt(jnp.maximum(v, 0.0)) + eps), nu)


# --------------------------------------------------------------------------
# Eigenvector structural attribution: how much of ||eigvec||^2 lives in the
# hidden layer(s) vs. each class's own column of the output layer.
# --------------------------------------------------------------------------

def _eigvec_energy_attribution(eigvec: ParamTree, num_classes: int) -> tuple[float, list[float]]:
    total_sq = float(_tree_dot(eigvec, eigvec))
    if total_sq <= 1e-20:
        return float("nan"), [float("nan")] * num_classes

    hidden_sq = 0.0
    for layer in eigvec.get("hidden_layers", []):
        hidden_sq += float(jnp.sum(layer["kernel"] ** 2))
        if "bias" in layer:
            hidden_sq += float(jnp.sum(layer["bias"] ** 2))

    classifier = eigvec["classifier"]
    kernel = classifier["kernel"]  # [in_dim, num_classes]
    bias = classifier.get("bias", None)

    class_energy: list[float] = []
    for c in range(num_classes):
        e = float(jnp.sum(kernel[:, c] ** 2))
        if bias is not None:
            e += float(bias[c] ** 2)
        class_energy.append(e / total_sq)

    return hidden_sq / total_sq, class_energy


# --------------------------------------------------------------------------
# Per-class gradients: conflict + norms.
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


def _class_gradient_norms(grads_per_class: list[ParamTree | None], num_classes: int) -> list[float]:
    norms: list[float] = []
    for c in range(num_classes):
        g = grads_per_class[c]
        norms.append(float(_tree_norm(g)) if g is not None else float("nan"))
    return norms


# --------------------------------------------------------------------------
# Gradient noise scale across ordinary (un-bumped) minibatches.
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

    grads_arr = jnp.stack(batch_grads, axis=0)
    mean_grad = jnp.mean(grads_arr, axis=0)
    var = jnp.mean(jnp.sum((grads_arr - mean_grad) ** 2, axis=1))
    denom = jnp.sum(mean_grad ** 2)
    noise_scale = jnp.where(denom > 1e-12, var / denom, 0.0)
    return float(noise_scale)


# --------------------------------------------------------------------------
# Prediction churn (overall + per class).
# --------------------------------------------------------------------------

def _predictions_and_margins(
    model: JAXModel, params: ParamTree, x: jax.Array
) -> tuple[np.ndarray, np.ndarray]:
    logits = model.apply(params, x)
    sorted_logits = jnp.sort(logits, axis=1)
    margins = sorted_logits[:, -1] - sorted_logits[:, -2]
    preds = jnp.argmax(logits, axis=1)
    return np.asarray(preds), np.asarray(margins)


def _prediction_churn(
    preds_before: np.ndarray,
    preds_after: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> tuple[float, list[float]]:
    flipped = preds_before != preds_after
    churn_overall = float(np.mean(flipped)) if flipped.size > 0 else float("nan")
    churn_per_class: list[float] = []
    for c in range(num_classes):
        mask = targets == c
        churn_per_class.append(float(np.mean(flipped[mask])) if np.any(mask) else float("nan"))
    return churn_overall, churn_per_class


# --------------------------------------------------------------------------
# Persistent low-margin group.
# --------------------------------------------------------------------------

def _low_margin_group(
    margins: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray,
    quantile: float,
    prev_indices: Optional[np.ndarray],
) -> tuple[float, float, float, np.ndarray]:
    n = margins.shape[0]
    k = max(1, int(round(quantile * n)))
    order = np.argsort(margins)
    low_idx = order[:k]

    if prev_indices is not None and prev_indices.size > 0:
        set_now = set(low_idx.tolist())
        set_prev = set(prev_indices.tolist())
        jaccard = len(set_now & set_prev) / max(1, len(set_now | set_prev))
    else:
        jaccard = float("nan")

    group_accuracy = float(np.mean(preds[low_idx] == targets[low_idx]))
    median_margin_all = float(np.median(margins))
    mean_margin_group = float(np.mean(margins[low_idx]))
    relative_margin = mean_margin_group / median_margin_all if abs(median_margin_all) > 1e-12 else float("nan")

    return float(jaccard), group_accuracy, relative_margin, low_idx


# --------------------------------------------------------------------------
# Public entry point: one diagnostics snapshot for a training window.
#
# `prev_state`, if provided, should be the `new_state` dict returned by the
# previous call (pass None on the first call). It threads the top
# eigenvector and low-margin index set forward so persistence metrics can
# be computed.
# --------------------------------------------------------------------------

def collect_deep_epoch(
    model: JAXModel,
    params_before: ParamTree,
    params_after: ParamTree,
    initial_params: ParamTree,
    opt_state_before: Any,
    metric_inputs: np.ndarray,
    metric_targets: np.ndarray,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    num_classes: int,
    batch_size: int,
    learning_rate: float,
    adam_eps: float,
    adam_beta2: float,
    rng: np.random.Generator,
    device: jax.Device,
    prev_state: Optional[Dict[str, Any]] = None,
    power_iters: int = 8,
    gns_minibatches: int = 5,
    low_margin_quantile: float = 0.2,
    hess_key: jax.Array | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    prev_top_eigvec = prev_state.get("top_eigvec") if prev_state else None
    prev_low_margin_indices = prev_state.get("low_margin_indices") if prev_state else None

    x_full = jax.device_put(jnp.asarray(metric_inputs, dtype=jnp.float32), device=device)
    y_full = jax.device_put(jnp.asarray(metric_targets, dtype=jnp.int32), device=device)
    loss_fn_before = _make_loss_fn(model, x_full, y_full)
    loss_fn_after = _make_loss_fn(model, x_full, y_full)

    grad_before = jax.grad(loss_fn_before)(params_before)
    grad_after = jax.grad(loss_fn_after)(params_after)

    grad_norm_full = float(_tree_norm(grad_before))
    grad_cosine_persistence = _cosine(grad_before, grad_after)

    key = hess_key if hess_key is not None else jax.random.PRNGKey(0)
    sharpness_top_eig, top_eigvec = estimate_top_eigenpair(
        loss_fn_before, params_before, power_iters, key
    )
    eos_ratio = float(learning_rate) * sharpness_top_eig

    nu_hat = _find_adam_second_moment(opt_state_before, adam_beta2)
    if nu_hat is not None:
        precond_sqrt = _adam_precond_sqrt(nu_hat, adam_eps)
        preconditioned_sharpness_top_eig, _ = estimate_preconditioned_top_eigenpair(
            loss_fn_before, params_before, precond_sqrt, power_iters, key
        )
        preconditioned_eos_ratio = float(learning_rate) * preconditioned_sharpness_top_eig
    else:
        preconditioned_sharpness_top_eig = float("nan")
        preconditioned_eos_ratio = float("nan")

    displacement = _tree_sub(params_after, params_before)
    displacement_norm = float(_tree_norm(displacement))
    grad_displacement_cosine = _cosine(grad_before, displacement)
    effective_step_ratio = (
        displacement_norm / grad_norm_full if grad_norm_full > 1e-12 else float("nan")
    )

    if displacement_norm > 1e-12:
        disp_unit = _tree_scale(displacement, 1.0 / displacement_norm)
        hv_disp = _hvp(loss_fn_before, params_before, disp_unit)
        curvature_along_displacement = float(_tree_dot(disp_unit, hv_disp))
        cosine_top_eigvec_displacement = abs(_cosine(top_eigvec, displacement))
    else:
        curvature_along_displacement = float("nan")
        cosine_top_eigvec_displacement = float("nan")

    if prev_top_eigvec is not None:
        eigvec_persistence_cosine = abs(_cosine(top_eigvec, prev_top_eigvec))
    else:
        eigvec_persistence_cosine = float("nan")

    total_drift = _tree_sub(params_before, initial_params)
    cumulative_drift_cosine = abs(_cosine(top_eigvec, total_drift))

    eigvec_energy_hidden_frac, eigvec_energy_classifier = _eigvec_energy_attribution(
        top_eigvec, num_classes
    )

    grads_per_class = _per_class_gradients(
        model, params_before, metric_inputs, metric_targets, num_classes, device
    )
    class_grad_conflict_mean, class_grad_conflict_min = _class_gradient_conflict(grads_per_class)
    class_grad_norms = _class_gradient_norms(grads_per_class, num_classes)

    gradient_noise_scale = _estimate_gradient_noise_scale(
        model, params_before, train_inputs, train_targets, batch_size, gns_minibatches, rng, device
    )

    preds_before, margins_before = _predictions_and_margins(model, params_before, x_full)
    preds_after, _ = _predictions_and_margins(model, params_after, x_full)
    pred_churn_overall, pred_churn_per_class = _prediction_churn(
        preds_before, preds_after, metric_targets, num_classes
    )

    low_margin_jaccard, low_margin_accuracy, low_margin_relative, low_margin_indices = _low_margin_group(
        margins_before, preds_before, metric_targets, low_margin_quantile, prev_low_margin_indices
    )

    raw: Dict[str, Any] = {
        "grad_norm_full": grad_norm_full,
        "sharpness_top_eig": sharpness_top_eig,
        "eos_ratio": eos_ratio,
        "preconditioned_sharpness_top_eig": preconditioned_sharpness_top_eig,
        "preconditioned_eos_ratio": preconditioned_eos_ratio,
        "curvature_along_displacement": curvature_along_displacement,
        "cosine_top_eigvec_displacement": cosine_top_eigvec_displacement,
        "eigvec_persistence_cosine": eigvec_persistence_cosine,
        "cumulative_drift_cosine": cumulative_drift_cosine,
        "eigvec_energy_hidden_frac": eigvec_energy_hidden_frac,
        "class_grad_conflict_mean": class_grad_conflict_mean,
        "class_grad_conflict_min": class_grad_conflict_min,
        "grad_cosine_persistence": grad_cosine_persistence,
        "gradient_noise_scale": gradient_noise_scale,
        "displacement_norm": displacement_norm,
        "grad_displacement_cosine": grad_displacement_cosine,
        "effective_step_ratio": effective_step_ratio,
        "pred_churn_overall": pred_churn_overall,
        "low_margin_persistence_jaccard": low_margin_jaccard,
        "low_margin_group_accuracy": low_margin_accuracy,
        "low_margin_relative_margin": low_margin_relative,
    }
    for c in range(num_classes):
        raw[f"class_grad_norm_class_{c}"] = class_grad_norms[c]
        raw[f"pred_churn_class_{c}"] = pred_churn_per_class[c]
        raw[f"eigvec_energy_classifier_class_{c}"] = eigvec_energy_classifier[c]

    new_state = {"top_eigvec": top_eigvec, "low_margin_indices": low_margin_indices}
    return raw, new_state


_DEEP_SCALAR_FIELDS = [
    "grad_norm_full",
    "sharpness_top_eig",
    "eos_ratio",
    "preconditioned_sharpness_top_eig",
    "preconditioned_eos_ratio",
    "curvature_along_displacement",
    "cosine_top_eigvec_displacement",
    "eigvec_persistence_cosine",
    "cumulative_drift_cosine",
    "eigvec_energy_hidden_frac",
    "class_grad_conflict_mean",
    "class_grad_conflict_min",
    "grad_cosine_persistence",
    "gradient_noise_scale",
    "displacement_norm",
    "grad_displacement_cosine",
    "effective_step_ratio",
    "pred_churn_overall",
    "low_margin_persistence_jaccard",
    "low_margin_group_accuracy",
    "low_margin_relative_margin",
]


def _deep_fields(num_classes: int) -> list[str]:
    fields = list(_DEEP_SCALAR_FIELDS)
    for c in range(num_classes):
        fields.append(f"class_grad_norm_class_{c}")
    for c in range(num_classes):
        fields.append(f"pred_churn_class_{c}")
    for c in range(num_classes):
        fields.append(f"eigvec_energy_classifier_class_{c}")
    return fields


def initialize_deep_csv(csv_path: Path, num_classes: int) -> None:
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "global_step", *_deep_fields(num_classes)])


def append_deep_csv_row(
    csv_path: Path,
    epoch: int,
    global_step: int,
    raw: Dict[str, Any],
    num_classes: int,
) -> None:
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, global_step, *(raw[field] for field in _deep_fields(num_classes))])


def finalize_deep_plots(
    csv_path: Path,
    output_path: Path,
    num_classes: int,
    tpt_step: int = -1,
    title: str = "",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [{k: float(v) for k, v in row.items()} for row in reader]

    if not rows:
        return

    steps = [r["global_step"] for r in rows]
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(6, 2, figsize=(13, 24), sharex=True)
    ax = axes.ravel()

    ax[0].plot(steps, [r["grad_norm_full"] for r in rows], color="tab:blue")
    ax[0].set_title("Full-batch gradient norm")
    ax[0].set_yscale("log")

    ax[1].plot(steps, [r["sharpness_top_eig"] for r in rows], color="tab:red", label="top eigenvalue (raw)")
    ax[1].plot(steps, [r["curvature_along_displacement"] for r in rows], color="tab:red", alpha=0.4, label="curvature along step taken")
    ax[1].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax[1].set_title("Raw sharpness vs. curvature actually being climbed")
    ax[1].legend(fontsize=7)

    ax[2].plot(steps, [r["eos_ratio"] for r in rows], color="tab:red", label="raw: lr * sharpness")
    ax[2].plot(steps, [r["preconditioned_eos_ratio"] for r in rows], color="tab:cyan", label="Adam-preconditioned")
    ax[2].axhline(2.0, color="black", linewidth=0.8, linestyle="--", label="GD instability ref. (~2)")
    ax[2].set_yscale("symlog", linthresh=1.0)
    ax[2].set_title("Edge-of-stability ratio: raw vs. Adam-preconditioned\n(symlog scale; preconditioned ratio is noisy in the first checkpoints while Adam's v warms up)")
    ax[2].legend(fontsize=7)

    ax[3].plot(steps, [r["eigvec_persistence_cosine"] for r in rows], color="tab:pink", label="|cos| vs previous checkpoint")
    ax[3].plot(steps, [r["cosine_top_eigvec_displacement"] for r in rows], color="tab:olive", alpha=0.6, label="|cos| vs this step's displacement")
    ax[3].plot(steps, [r["cumulative_drift_cosine"] for r in rows], color="tab:brown", label="|cos| vs TOTAL drift since init")
    ax[3].set_ylim(-0.05, 1.05)
    ax[3].set_title("Is it the SAME direction, and is it where training has been marching?")
    ax[3].legend(fontsize=7)

    ax[4].plot(steps, [r["eigvec_energy_hidden_frac"] for r in rows], color="black", linewidth=1.5, label="hidden layer(s)")
    for c in range(num_classes):
        ax[4].plot(steps, [r[f"eigvec_energy_classifier_class_{c}"] for r in rows], color=cmap(c % 10), label=f"classifier: class {c}")
    ax[4].set_ylim(-0.05, 1.05)
    ax[4].set_title("Where does the sharp direction structurally live?\n(fraction of top-eigenvector energy)")
    ax[4].legend(fontsize=7, ncol=2)

    ax[5].plot(steps, [r["class_grad_conflict_mean"] for r in rows], label="conflict mean", color="tab:purple")
    ax[5].plot(steps, [r["class_grad_conflict_min"] for r in rows], label="conflict min", color="tab:purple", alpha=0.4)
    ax[5].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax[5].set_title("Per-class gradient conflict")
    ax[5].legend(fontsize=7)

    for c in range(num_classes):
        ax[6].plot(steps, [r[f"class_grad_norm_class_{c}"] for r in rows], color=cmap(c % 10), label=f"class {c}")
    ax[6].set_title("Per-class gradient norm")
    ax[6].set_yscale("log")
    ax[6].legend(fontsize=7, ncol=2)

    ax[7].plot(steps, [r["pred_churn_overall"] for r in rows], color="black", linewidth=1.5, label="overall")
    for c in range(num_classes):
        ax[7].plot(steps, [r[f"pred_churn_class_{c}"] for r in rows], color=cmap(c % 10), alpha=0.6, label=f"class {c}")
    ax[7].set_title("Prediction churn per window")
    ax[7].legend(fontsize=7, ncol=2)

    ax[8].plot(steps, [r["low_margin_persistence_jaccard"] for r in rows], color="tab:green", label="Jaccard overlap w/ prev. window")
    ax[8].plot(steps, [r["low_margin_group_accuracy"] for r in rows], color="tab:blue", alpha=0.7, label="group accuracy")
    ax[8].set_ylim(-0.05, 1.05)
    ax8b = ax[8].twinx()
    ax8b.plot(steps, [r["low_margin_relative_margin"] for r in rows], color="tab:orange", alpha=0.7, label="relative margin (vs median)")
    ax[8].set_title(f"Persistent low-margin group (bottom quantile by margin)")
    lines1, labels1 = ax[8].get_legend_handles_labels()
    lines2, labels2 = ax8b.get_legend_handles_labels()
    ax[8].legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    ax[9].plot(steps, [r["gradient_noise_scale"] for r in rows], color="tab:orange", label="grad noise scale")
    ax9b = ax[9].twinx()
    ax9b.plot(steps, [r["effective_step_ratio"] for r in rows], color="tab:brown", alpha=0.6, label="displacement/grad norm")
    ax[9].set_yscale("log")
    ax[9].set_title("Gradient noise scale & effective step ratio")
    lines1, labels1 = ax[9].get_legend_handles_labels()
    lines2, labels2 = ax9b.get_legend_handles_labels()
    ax[9].legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    ax[10].axis("off")
    ax[11].axis("off")

    for a in ax:
        a.set_xlabel("global step")
        if tpt_step is not None and tpt_step >= 0:
            a.axvline(tpt_step, color="gray", linewidth=0.8, linestyle=":")

    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.subplots_adjust(hspace=0.45)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)