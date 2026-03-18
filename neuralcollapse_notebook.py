from pathlib import Path
import os

import gc
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from IPython import embed
from scipy.sparse.linalg import svds
from torchvision import datasets, transforms
from tqdm import tqdm

def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
debug = _env_bool("NC_DEBUG", True)  # Debug mode skips full batches

# dataset parameters
im_size = 28
padded_im_size = 32
C = 10
input_ch = 1

# Optimization Criterion
# loss_name = 'CrossEntropyLoss'
loss_name = "CrossEntropyLoss"

# Optimization hyperparameters
lr_decay = 0.1

# Best lr after hyperparameter tuning
if loss_name == "CrossEntropyLoss":
    lr = 0.0679
elif loss_name == "MSELoss":
    lr = 0.0184

# Get epochs from environment variable (must be set by run_nc_experiment.py or caller)
if "NC_EPOCHS" not in os.environ:
    raise ValueError("NC_EPOCHS environment variable not set! This must be passed from run_nc_experiment.py")
epochs = int(os.environ["NC_EPOCHS"])

# Number of optimization steps to run per epoch (must be provided by run_nc_experiment.py)
if "NC_STEPS_PER_EPOCH" not in os.environ:
    raise ValueError(
        "NC_STEPS_PER_EPOCH environment variable not set! This must be passed from run_nc_experiment.py"
    )
steps_per_epoch = int(os.environ["NC_STEPS_PER_EPOCH"])

if "NC_TOTAL_TRAINING_STEPS" not in os.environ:
    raise ValueError(
        "NC_TOTAL_TRAINING_STEPS environment variable not set! This must be passed from run_nc_experiment.py"
    )
total_training_steps = int(os.environ["NC_TOTAL_TRAINING_STEPS"])

epochs_lr_decay = [epochs // 3, epochs * 2 // 3]

batch_size = 128

# Initialize training step tracking variables
training_steps_per_epoch = None
total_training_steps_expected = None

momentum = 0.9
weight_decay = 5e-4

# bump dynamics (optional; controlled via environment variables)
bumps_before_tpt = _env_bool("NC_BUMPS_BEFORE_TPT", False)
bumps_at_tpt = _env_bool("NC_BUMPS_AT_TPT", False)
tpt_accuracy_threshold = float(os.environ.get("NC_TPT_ACCURACY_THRESHOLD", "1.0"))
bump_period_length = int(os.environ.get("NC_PERIOD_LENGTH", "2000"))
bump_w_max = float(os.environ.get("NC_W_MAX", "50.0"))

# model architecture (controlled via environment variables from run_nc_experiment.py)
model_architecture = os.environ.get("NC_MODEL_ARCHITECTURE", "resnet18").strip().lower()
if model_architecture not in {"resnet18", "mlp"}:
    raise ValueError(
        f"Unsupported NC_MODEL_ARCHITECTURE='{model_architecture}'. "
        "Expected one of: resnet18, mlp."
    )

mlp_hidden_dim = int(os.environ.get("NC_MLP_HIDDEN_DIM", "512"))
mlp_num_hidden_layers = int(os.environ.get("NC_MLP_NUM_HIDDEN_LAYERS", "2"))
mlp_use_bias = _env_bool("NC_MLP_USE_BIAS", True)


def compute_class_weights(t, focus_class, w_max, period_length, num_classes):
    slope = 2 * (w_max - 1) / period_length
    if t < period_length / 2:
        w_main_class = 1 + t * slope
    else:
        w_main_class = 2 * w_max - t * slope - 1

    weights = np.ones(num_classes, dtype=np.float32)
    weights[focus_class] = float(w_main_class)
    weights = weights / np.sum(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)

RESULTS_DIR = Path(
    os.environ.get("NC_RESULTS_DIR", "/data/samuel_lozano/dynamical-sgd/results")
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def train(
    model,
    criterion,
    device,
    num_classes,
    train_loader,
    optimizer,
    epoch,
    global_step,
    apply_bumping,
    max_steps_this_epoch,
):
    model.train()

    running_loss = 0.0
    running_correct = 0
    running_count = 0
    processed_batches = 0

    for batch_idx, (data, target) in enumerate(train_loader, start=1):
        if data.shape[0] != batch_size:
            continue

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)

        class_focus = None
        if str(criterion) == "CrossEntropyLoss()":
            if apply_bumping:
                class_focus = (global_step // bump_period_length) % num_classes
                class_weights = compute_class_weights(
                    global_step % bump_period_length,
                    class_focus,
                    bump_w_max,
                    bump_period_length,
                    num_classes,
                )
                sample_weights = class_weights[target]
                per_sample_loss = F.cross_entropy(out, target, reduction="none")
                loss = torch.mean(per_sample_loss * sample_weights)
            else:
                loss = criterion(out, target)
        elif str(criterion) == "MSELoss()":
            target_one_hot = F.one_hot(target, num_classes=num_classes).float()
            if apply_bumping:
                class_focus = (global_step // bump_period_length) % num_classes
                class_weights = compute_class_weights(
                    global_step % bump_period_length,
                    class_focus,
                    bump_w_max,
                    bump_period_length,
                    num_classes,
                )
                sample_weights = class_weights[target]
                per_sample_loss = torch.mean((out - target_one_hot) ** 2, dim=1)
                loss = torch.mean(per_sample_loss * sample_weights)
            else:
                loss = criterion(out, target_one_hot)

        loss.backward()
        optimizer.step()

        preds = torch.argmax(out, dim=1)
        running_correct += int(torch.sum(preds == target).item())
        running_count += int(target.shape[0])
        running_loss += float(loss.item())
        processed_batches += 1
        global_step += 1

        if processed_batches >= max_steps_this_epoch:
            break

        if debug and batch_idx > 20:
            break

    mean_loss = running_loss / max(1, processed_batches)
    epoch_accuracy = running_correct / max(1, running_count)
    return mean_loss, epoch_accuracy, global_step, class_focus, processed_batches


@torch.no_grad()
def evaluate(model, criterion, device, num_classes, loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        out = model(data)

        if str(criterion) == "CrossEntropyLoss()":
            loss = F.cross_entropy(out, target, reduction="sum")
        else:
            target_one_hot = F.one_hot(target, num_classes=num_classes).float()
            loss = F.mse_loss(out, target_one_hot, reduction="sum")

        total_loss += float(loss.item())
        total_correct += int(torch.sum(torch.argmax(out, dim=1) == target).item())
        total_count += int(target.shape[0])

    return total_loss / max(1, total_count), total_correct / max(1, total_count)


@torch.no_grad()
def analysis(graphs, model, criterion_summed, device, num_classes, loader):
    model.eval()

    N = torch.zeros(C, dtype=torch.long, device=device)
    mean_sum = None
    Sw = None

    loss = 0
    net_correct = 0
    NCC_match_net = 0

    for computation in ["Mean", "Cov"]:
        pbar = tqdm(total=len(loader), position=0, leave=True, disable=True)
        for batch_idx, (data, target) in enumerate(loader, start=1):
            data, target = data.to(device), target.to(device)

            output = model(data)
            h = features.value.data.view(data.shape[0], -1)  # B CHW
            if mean_sum is None:
                feat_dim = h.shape[1]
                mean_sum = torch.zeros(C, feat_dim, dtype=h.dtype, device=device)
                Sw = torch.zeros(feat_dim, feat_dim, dtype=h.dtype, device=device)

            # during calculation of class means, calculate loss
            if computation == "Mean":
                if str(criterion_summed) == "CrossEntropyLoss()":
                    loss += criterion_summed(output, target).item()
                elif str(criterion_summed) == "MSELoss()":
                    loss += criterion_summed(
                        output, F.one_hot(target, num_classes=num_classes).float()
                    ).item()
            if computation == "Mean":
                # Accumulate per-class feature sums and counts in one pass.
                mean_sum.index_add_(0, target, h)
                N += torch.bincount(target, minlength=C)
            elif computation == "Cov":
                # Use class means for each sample and accumulate covariance as z^T z.
                class_means = M.T[target]
                z = h - class_means
                Sw += z.T @ z

                # 1) network's accuracy
                net_pred = torch.argmax(output, dim=1)
                net_correct += int(torch.sum(net_pred == target).item())

                # 2) agreement between prediction and nearest class center
                NCC_scores = torch.cdist(h, M.T)
                NCC_pred = torch.argmin(NCC_scores, dim=1)
                NCC_match_net += int(torch.sum(NCC_pred == net_pred).item())

            pbar.update(1)

            if debug and batch_idx > 20:
                break
        pbar.close()

        if computation == "Mean":
            mean = torch.zeros_like(mean_sum)
            valid = N > 0
            mean[valid] = mean_sum[valid] / N[valid].unsqueeze(1)
            M = mean.T
            loss /= max(1, int(torch.sum(N).item()))
        elif computation == "Cov":
            Sw /= max(1, int(torch.sum(N).item()))

    graphs.loss.append(loss)
    total_count = max(1, int(torch.sum(N).item()))
    graphs.accuracy.append(net_correct / total_count)
    graphs.NCC_mismatch.append(1 - NCC_match_net / total_count)

    # loss with weight decay
    reg_loss = loss
    for param in model.parameters():
        reg_loss += 0.5 * weight_decay * torch.sum(param**2).item()
    graphs.reg_loss.append(reg_loss)

    # global mean
    muG = torch.mean(M, dim=1, keepdim=True)  # CHW 1

    # between-class covariance
    M_ = M - muG
    Sb = torch.matmul(M_, M_.T) / C

    # avg norm
    W = classifier.weight
    M_norms = torch.norm(M_, dim=0)
    W_norms = torch.norm(W.T, dim=0)

    # Fundamental components (to match non-MNIST combined NC figure)
    graphs.mu_c_norm_avg.append(torch.mean(M_norms).item())
    graphs.mu_G_norm.append(torch.norm(muG).item())
    graphs.M_fro_norm.append(torch.norm(M_, p="fro").item())
    graphs.W_fro_norm.append(torch.norm(W, p="fro").item())

    graphs.norm_M_CoV.append((torch.std(M_norms) / torch.mean(M_norms)).item())
    graphs.norm_W_CoV.append((torch.std(W_norms) / torch.mean(W_norms)).item())

    # Decomposition of MSE #
    if loss_name == "MSELoss":
        wd = 0.5 * weight_decay  # "\lambda" in manuscript, so this is halved
        St = Sw + Sb
        size_last_layer = Sb.shape[0]
        eye_P = torch.eye(size_last_layer).to(device)
        eye_C = torch.eye(C).to(device)

        St_inv = torch.inverse(St + (wd / (wd + 1)) * (muG @ muG.T) + wd * eye_P)

        w_LS = 1 / C * (M.T - 1 / (1 + wd) * muG.T) @ St_inv
        b_LS = (1 / C * torch.ones(C).to(device) - w_LS @ muG.T.squeeze(0)) / (1 + wd)
        w_LS_ = torch.cat([w_LS, b_LS.unsqueeze(-1)], dim=1)  # c x n
        b = classifier.bias
        if b is None:
            b = torch.zeros(C, dtype=W.dtype, device=device)
        w_ = torch.cat([W, b.unsqueeze(-1)], dim=1)  # c x n

        LNC1 = 0.5 * (
            torch.trace(w_LS @ (Sw + wd * eye_P) @ w_LS.T) + wd * torch.norm(b_LS) ** 2
        )
        LNC23 = 0.5 / C * torch.norm(w_LS @ M + b_LS.unsqueeze(1) - eye_C) ** 2

        A1 = torch.cat([St + muG @ muG.T + wd * eye_P, muG], dim=1)
        A2 = torch.cat([muG.T, torch.ones([1, 1]).to(device) + wd], dim=1)
        A = torch.cat([A1, A2], dim=0)
        Lperp = 0.5 * torch.trace((w_ - w_LS_) @ A @ (w_ - w_LS_).T)

        MSE_wd_features = (
            loss + 0.5 * weight_decay * (torch.norm(W) ** 2 + torch.norm(b) ** 2).item()
        )
        MSE_wd_features *= 0.5

        graphs.MSE_wd_features.append(MSE_wd_features)
        graphs.LNC1.append(LNC1.item())
        graphs.LNC23.append(LNC23.item())
        graphs.Lperp.append(Lperp.item())

    # tr{Sw Sb^-1}
    Sw = Sw.cpu().numpy()
    Sb = Sb.cpu().numpy()
    eigvec, eigval, _ = svds(Sb, k=C - 1)
    inv_Sb = eigvec @ np.diag(eigval ** (-1)) @ eigvec.T
    graphs.Sw_invSb.append(np.trace(Sw @ inv_Sb))

    # ||W^T - M_||
    normalized_M = M_ / torch.norm(M_, "fro")
    normalized_W = W.T / torch.norm(W.T, "fro")
    graphs.W_M_dist.append((torch.norm(normalized_W - normalized_M) ** 2).item())

    # NC2 equiangularity metrics for class means and classifiers
    # - std: Std of off-diagonal pairwise cosines
    # - mean: Avg|cos + 1/(C-1)| deviation from simplex ETF target
    def equiangular_stats(V):
        G = V.T @ V
        mask = ~torch.eye(C, dtype=torch.bool, device=device)
        off_diag = G[mask]
        target = -1.0 / (C - 1)
        std_val = torch.std(off_diag)
        mean_dev = torch.mean(torch.abs(off_diag - target))
        return std_val.item(), mean_dev.item()

    cos_M_std, cos_M_mean = equiangular_stats(M_ / M_norms)
    cos_W_std, cos_W_mean = equiangular_stats(W.T / W_norms)

    graphs.cos_M_std.append(cos_M_std)
    graphs.cos_W_std.append(cos_W_std)
    graphs.cos_M.append(cos_M_mean)
    graphs.cos_W.append(cos_W_mean)


def save_current_figures(epoch):
    figure_names = {
        1: "training_curves",
        3: "nc1_activation_collapse",
        4: "nc2_equinorm",
        5: "nc2_maximal_equiangularity",
        6: "nc3_self_duality",
        7: "nc4_convergence_to_ncc",
        8: "mse_decomposition",
    }
    for fig_num in plt.get_fignums():
        plt.figure(fig_num)
        name = figure_names.get(fig_num, f"figure_{fig_num}")
        out_path = RESULTS_DIR / f"{name}_epoch_{epoch:03d}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")


def init_results_csv():
    """Initialize the CSV file with header."""
    out_csv = RESULTS_DIR / "nc_results.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(
            "TrainingStep,RegLoss,TrainError,TrainLossEpoch,TrainAccEpoch,TestLoss,TestAcc,NC1,NC2_Means_Equinorm,NC2_Classifiers_Equinorm,"
            "NC2_Means_Equiangularity,NC2_Classifiers_Equiangularity,NC3,NC4\n"
        )


def append_results_csv_row(step, graphs_obj, index):
    """Append a single row to the CSV file."""
    out_csv = RESULTS_DIR / "nc_results.csv"
    with open(out_csv, "a", encoding="utf-8") as f:
        train_error = 1.0 - float(graphs_obj.accuracy[index])
        row = [
            str(step),
            f"{float(graphs_obj.reg_loss[index]):.10f}",
            f"{train_error:.10f}",
            f"{float(graphs_obj.train_loss_epoch[index]):.10f}",
            f"{float(graphs_obj.train_acc_epoch[index]):.10f}",
            f"{float(graphs_obj.test_loss[index]):.10f}",
            f"{float(graphs_obj.test_acc[index]):.10f}",
            f"{float(graphs_obj.Sw_invSb[index]):.10f}",
            f"{float(graphs_obj.norm_M_CoV[index]):.10f}",
            f"{float(graphs_obj.norm_W_CoV[index]):.10f}",
            f"{float(graphs_obj.cos_M[index]):.10f}",
            f"{float(graphs_obj.cos_W[index]):.10f}",
            f"{float(graphs_obj.W_M_dist[index]):.10f}",
            f"{float(graphs_obj.NCC_mismatch[index]):.10f}",
        ]
        f.write(",".join(row) + "\n")


def save_dataset_visualizations(train_dataset, test_dataset):
    # Save one batch-like class sample grid from training set.
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle("MNIST Training Dataset Samples", fontsize=16)

    labels_train = train_dataset.targets
    for class_idx in range(10):
        idxs = (labels_train == class_idx).nonzero(as_tuple=True)[0]
        row, col = divmod(class_idx, 5)
        if len(idxs) > 0:
            sample_image = train_dataset.data[idxs[0]].numpy()
            axes[row, col].imshow(sample_image, cmap="gray")
            axes[row, col].set_title(f"Class {class_idx}")
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_dataset_samples.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save train/test class distribution histogram.
    train_counts = np.bincount(train_dataset.targets.numpy(), minlength=10)
    test_counts = np.bincount(test_dataset.targets.numpy(), minlength=10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(np.arange(10), train_counts, alpha=0.8, color="skyblue")
    ax1.set_title("Train Class Distribution")
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Count")
    ax1.grid(True, alpha=0.3)

    ax2.bar(np.arange(10), test_counts, alpha=0.8, color="salmon")
    ax2.set_title("Test Class Distribution")
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Count")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "dataset_statistics.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(
        "Dataset summary | "
        f"train_samples={len(train_dataset)} | test_samples={len(test_dataset)} | "
        f"train_counts={train_counts.tolist()} | test_counts={test_counts.tolist()}"
    )


def save_bump_structure_plot(total_steps, bumps_before_tpt, bumps_at_tpt, tpt_emergence_step=None):
    if total_steps <= 0:
        return

    steps = np.arange(total_steps)
    weights_over_time = np.zeros((C, total_steps), dtype=np.float32)
    
    for s in steps:
        # Determine if bumping is active at this step
        if tpt_emergence_step is None:
            # TPT never reached - use pre-TPT bumping setting throughout
            apply_bumping = bumps_before_tpt
        else:
            # TPT was reached - switch bumping mode at that point
            if s < tpt_emergence_step:
                apply_bumping = bumps_before_tpt
            else:
                apply_bumping = bumps_at_tpt
        
        # Compute weights for this step
        if apply_bumping:
            focus_class = (s // bump_period_length) % C
            w = compute_class_weights(s % bump_period_length, focus_class, bump_w_max, bump_period_length, C)
            weights_over_time[:, s] = w.detach().cpu().numpy()
        else:
            # No bumping - uniform weights across all classes
            weights_over_time[:, s] = 1.0 / C

    plt.figure(figsize=(14, 6))
    for c in range(C):
        plt.plot(steps, weights_over_time[c], linewidth=1.0, label=f"Class {c}")
    
    # Add vertical line at TPT emergence if it occurred
    if tpt_emergence_step is not None:
        plt.axvline(
            x=tpt_emergence_step,
            color="red",
            linestyle="--",
            linewidth=2.0,
            alpha=0.7,
            label=f"TPT Emergence (step {tpt_emergence_step})"
        )
    
    plt.title("Dynamic Class Focus Weights Over Steps")
    plt.xlabel("Training Step")
    plt.ylabel("Sampling Weight")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3, fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "class_focus_dynamics.png", dpi=150, bbox_inches="tight")
    plt.close()


def first_zero_error_step(training_steps, graphs_obj):
    for step, acc in zip(training_steps, graphs_obj.accuracy):
        error = 1.0 - float(acc)
        if np.isclose(error, 0.0, atol=1e-12):
            return step
    return None


def draw_zero_error_bar_on_all_plots(zero_error_step):
    if zero_error_step is None:
        return
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        for ax in fig.get_axes():
            ax.axvline(
                x=zero_error_step,
                color="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.9,
            )


def save_nc_metrics_evolution_plot(training_steps, graphs_obj, tpt_step=None, tpt_threshold=1.0):
    """Save a 7-subplot NC metrics evolution figure matching the non-MNIST layout."""
    if not training_steps:
        return

    fig, axes = plt.subplots(3, 3, figsize=(21, 15))

    # Figure 2: NC2 - Equinorm (CV)
    ax = axes[0, 0]
    ax.plot(training_steps, graphs_obj.norm_M_CoV, "o-", linewidth=2, markersize=6, color="#2E86AB", label="Class-means")
    ax.plot(training_steps, graphs_obj.norm_W_CoV, "s-", linewidth=2, markersize=6, color="#F18F01", label="Classifiers")
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Coefficient of Variation", fontsize=12, fontweight="bold")
    ax.set_title(
        "Figure 2: NC2 - Equinorm\nStd(||mu_c - mu_G||) / Avg(||mu_c - mu_G||)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, linewidth=1.5, label="Target: 0")
    ax.legend(fontsize=10)

    # Figure 3: NC2 - Equiangularity (Std)
    ax = axes[0, 1]
    ax.plot(training_steps, graphs_obj.cos_M_std, "o-", linewidth=2, markersize=6, color="#2E86AB", label="Class-means")
    ax.plot(training_steps, graphs_obj.cos_W_std, "s-", linewidth=2, markersize=6, color="#F18F01", label="Classifiers")
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Std of Cosines", fontsize=12, fontweight="bold")
    ax.set_title("Figure 3: NC2 - Equiangularity (Std)\nStd(cos(c,c'))", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, linewidth=1.5, label="Target: 0")
    ax.legend(fontsize=10)

    # Figure 4: NC2 - Equiangularity (Mean)
    ax = axes[0, 2]
    ax.plot(training_steps, graphs_obj.cos_M, "o-", linewidth=2, markersize=6, color="#2E86AB", label="Class-means")
    ax.plot(training_steps, graphs_obj.cos_W, "s-", linewidth=2, markersize=6, color="#F18F01", label="Classifiers")
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Avg |cos + 1/(C-1)|", fontsize=12, fontweight="bold")
    ax.set_title("Figure 4: NC2 - Equiangularity (Mean)\nAvg|cos(c,c') + 1/(C-1)|", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, linewidth=1.5, label="Target: 0")
    ax.legend(fontsize=10)

    # Figure 5: NC3 - Self-Duality
    ax = axes[1, 0]
    ax.plot(training_steps, graphs_obj.W_M_dist, "o-", linewidth=2, markersize=6, color="#D62246")
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("NC3 Metric", fontsize=12, fontweight="bold")
    ax.set_title("Figure 5: NC3 - Self-Duality\n||W^T - M||_F", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, linewidth=1.5, label="Target: 0")
    ax.legend()

    # Figure 6: NC1 - Variability Collapse
    ax = axes[1, 1]
    ax.plot(training_steps, graphs_obj.Sw_invSb, "o-", linewidth=2, markersize=6, color="#2E86AB", label="Within-Class Variation")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Tr{Sw Sb^-1} (log scale)", fontsize=12, fontweight="bold")
    ax.set_title("Figure 6: NC1 - Variability Collapse\nTr{Sw Sb^-1}", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    # Figure 7: NC4 - Nearest Class-Center Mismatch
    ax = axes[1, 2]
    ax.plot(training_steps, graphs_obj.NCC_mismatch, "o-", linewidth=2, markersize=6, color="#A23B72")
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Proportion of Disagreements", fontsize=12, fontweight="bold")
    ax.set_title("Figure 7: Classifier -> NCC\nProportion where Classifier != arg min||h-mu_c||", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, linewidth=1.5, label="Target: 0")
    ax.legend()

    # Fundamental components evolution panel
    ax = axes[2, 0]
    ax.plot(training_steps, graphs_obj.mu_c_norm_avg, "o-", linewidth=2, markersize=6, color="#2E86AB", label="<||mu_c - mu_G||>")
    ax.plot(training_steps, graphs_obj.mu_G_norm, "s-", linewidth=2, markersize=6, color="#F18F01", label="||mu_G||")
    ax.plot(training_steps, graphs_obj.M_fro_norm, "^-", linewidth=2, markersize=6, color="#D62246", label="||M||_F")
    ax.plot(training_steps, graphs_obj.W_fro_norm, "v-", linewidth=2, markersize=6, color="#A23B72", label="||W||_F")
    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Component Norms", fontsize=12, fontweight="bold")
    ax.set_title("Fundamental Components Evolution\nNorms of mu_c, mu_G, M, W", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")

    # Hide unused subplots to keep total active subplots = 7
    axes[2, 1].axis("off")
    axes[2, 2].axis("off")

    if tpt_step is not None:
        tpt_label = f"{tpt_threshold * 100:.0f}% Train Acc" if tpt_threshold < 1.0 else "100% Train Acc"
        for axis in axes.flat:
            if axis.axison:
                axis.axvline(x=tpt_step, color="black", linestyle="-", alpha=0.8, linewidth=2)
        axes[0, 0].text(
            tpt_step,
            axes[0, 0].get_ylim()[1] * 0.9,
            tpt_label,
            rotation=90,
            verticalalignment="top",
            fontsize=10,
            color="black",
            fontweight="bold",
        )

    plt.suptitle("Neural Collapse Metrics Evolution + Fundamental Components", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "nc_metrics_evolution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_hidden_layers, num_classes, use_bias=True):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim, bias=use_bias))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers) if layers else nn.Identity()
        self.fc = nn.Linear(in_dim, num_classes, bias=use_bias)

    def forward(self, x):
        x = torch.flatten(x, 1)
        h = self.feature_extractor(x)
        return self.fc(h)


if model_architecture == "resnet18":
    model = models.resnet18(pretrained=False, num_classes=C)
    model.conv1 = nn.Conv2d(
        input_ch,
        model.conv1.weight.shape[0],
        3,
        1,
        1,
        bias=False,
    )  # Small dataset filter size used by He et al. (2015)
    model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
else:
    mlp_input_dim = input_ch * im_size * im_size
    model = MLPClassifier(
        input_dim=mlp_input_dim,
        hidden_dim=mlp_hidden_dim,
        num_hidden_layers=mlp_num_hidden_layers,
        num_classes=C,
        use_bias=mlp_use_bias,
    )
model = model.to(device)


class features:
    pass


def hook(self, input, output):
    features.value = input[0].detach()


# register hook that saves last-layer input into features
classifier = model.fc
classifier.register_forward_hook(hook)

if model_architecture == "resnet18":
    transform = transforms.Compose(
        [
            transforms.Pad((padded_im_size - im_size) // 2),
            transforms.ToTensor(),
            transforms.Normalize(0.1307, 0.3081),
        ]
    )
else:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(0.1307, 0.3081),
        ]
    )

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("../data", train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True,
)

analysis_loader = torch.utils.data.DataLoader(
    datasets.MNIST("../data", train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("../data", train=False, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=False,
)

train_dataset_raw = datasets.MNIST("../data", train=True, download=True)
test_dataset_raw = datasets.MNIST("../data", train=False, download=True)
save_dataset_visualizations(train_dataset_raw, test_dataset_raw)

if loss_name == "CrossEntropyLoss":
    criterion = nn.CrossEntropyLoss()
    criterion_summed = nn.CrossEntropyLoss(reduction="sum")
elif loss_name == "MSELoss":
    criterion = nn.MSELoss()
    criterion_summed = nn.MSELoss(reduction="sum")

optimizer = optim.SGD(
    model.parameters(),
    lr=lr,
    momentum=momentum,
    weight_decay=weight_decay,
)

lr_scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=epochs_lr_decay,
    gamma=lr_decay,
)


class graphs:
    def __init__(self):
        self.accuracy = []
        self.loss = []
        self.reg_loss = []
        self.train_loss_epoch = []
        self.train_acc_epoch = []
        self.test_loss = []
        self.test_acc = []

        # NC1
        self.Sw_invSb = []

        # NC2
        self.norm_M_CoV = []
        self.norm_W_CoV = []
        self.cos_M_std = []
        self.cos_W_std = []
        self.cos_M = []
        self.cos_W = []

        # NC3
        self.W_M_dist = []

        # NC4
        self.NCC_mismatch = []

        # Fundamental components
        self.mu_c_norm_avg = []
        self.mu_G_norm = []
        self.M_fro_norm = []
        self.W_fro_norm = []

        # Decomposition
        self.MSE_wd_features = []
        self.LNC1 = []
        self.LNC23 = []
        self.Lperp = []


graphs = graphs()

global_step = 0
terminal_full_training_reached = False
total_steps_seen = 0

training_steps = []

# Initialize CSV file before training starts
init_results_csv()

for epoch in range(1, epochs + 1):
    if global_step >= total_training_steps:
        break

    if terminal_full_training_reached:
        apply_bumping = bumps_at_tpt
    else:
        apply_bumping = bumps_before_tpt

    steps_remaining = total_training_steps - global_step
    steps_this_epoch = min(steps_per_epoch, steps_remaining)

    train_loss, train_acc, global_step, class_focus, processed_batches = train(
        model,
        criterion,
        device,
        C,
        train_loader,
        optimizer,
        epoch,
        global_step,
        apply_bumping,
        steps_this_epoch,
    )
    total_steps_seen += processed_batches

    if processed_batches < steps_this_epoch:
        print(
            f"⚠️  Warning: Epoch {epoch} processed {processed_batches} steps, "
            f"below requested steps={steps_this_epoch}."
        )

    # Compute training steps per epoch on first epoch
    if epoch == 1:
        training_steps_per_epoch = processed_batches
        total_training_steps_expected = total_training_steps
        print(f"\n[Training Configuration]")
        print(f"  Epochs: {epochs}")
        print(f"  Steps per epoch (configured): {steps_per_epoch}")
        print(f"  Steps per epoch (epoch 1 actual): {training_steps_per_epoch}")
        print(f"  Total expected training steps: {total_training_steps_expected}")
        print(f"  Batch size: {batch_size}")
        print(f"  Model architecture: {model_architecture}")
        if model_architecture == "mlp":
            print(f"  MLP hidden layers: {mlp_num_hidden_layers}")
            print(f"  MLP hidden width: {mlp_hidden_dim}")
            print(f"  MLP bias enabled: {mlp_use_bias}")
        print(f"  Bumping enabled before TPT: {bumps_before_tpt}")
        print(f"  Bumping enabled at TPT: {bumps_at_tpt}")
        print(f"  TPT accuracy threshold: {tpt_accuracy_threshold}")
        print()

    if (not terminal_full_training_reached) and train_acc >= tpt_accuracy_threshold:
        terminal_full_training_reached = True
        print(f"\n[Terminal Phase Training (TPT) reached at step {global_step}]")
        print(f"  Epoch: {epoch}")
        print(f"  Training accuracy: {train_acc:.4f}")
        print()

    lr_scheduler.step()

    training_steps.append(global_step)
    analysis(graphs, model, criterion_summed, device, C, analysis_loader)
    test_loss, test_acc = evaluate(model, criterion, device, C, test_loader)
    graphs.train_loss_epoch.append(train_loss)
    graphs.train_acc_epoch.append(train_acc)
    graphs.test_loss.append(test_loss)
    graphs.test_acc.append(test_acc)

    # Write metrics to CSV incrementally after each epoch
    current_index = len(training_steps) - 1
    append_results_csv_row(global_step, graphs, current_index)

    plt.figure(1, figsize=(12, 5))
    plt.clf()
    ax_loss = plt.subplot(1, 2, 1)
    ax_loss.plot(training_steps, graphs.train_loss_epoch, label="Train Loss", linewidth=2.0)
    ax_loss.plot(training_steps, graphs.test_loss, label="Test Loss", linewidth=2.0)
    ax_loss.set_xlabel("Training Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    ax_acc = plt.subplot(1, 2, 2)
    ax_acc.plot(training_steps, graphs.train_acc_epoch, label="Train Accuracy", linewidth=2.0)
    ax_acc.plot(training_steps, graphs.test_acc, label="Test Accuracy", linewidth=2.0)
    ax_acc.set_xlabel("Training Step")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend()

    plt.tight_layout()

    plt.figure(3)
    plt.clf()
    plt.semilogy(training_steps, graphs.Sw_invSb)
    plt.xlabel("Training Step")
    plt.ylabel("Tr{Sw Sb^-1}")
    plt.title("NC1: Activation Collapse")

    plt.figure(4)
    plt.clf()
    plt.plot(training_steps, graphs.norm_M_CoV)
    plt.plot(training_steps, graphs.norm_W_CoV)
    plt.legend(["Class Means", "Classifiers"])
    plt.xlabel("Training Step")
    plt.ylabel("Std/Avg of Norms")
    plt.title("NC2: Equinorm")

    plt.figure(5)
    plt.clf()
    plt.plot(training_steps, graphs.cos_M)
    plt.plot(training_steps, graphs.cos_W)
    plt.legend(["Class Means", "Classifiers"])
    plt.xlabel("Training Step")
    plt.ylabel("Avg|Cos + 1/(C-1)|")
    plt.title("NC2: Maximal Equiangularity")

    plt.figure(6)
    plt.clf()
    plt.plot(training_steps, graphs.W_M_dist)
    plt.xlabel("Training Step")
    plt.ylabel("||W^T - H||^2")
    plt.title("NC3: Self Duality")

    plt.figure(7)
    plt.clf()
    plt.plot(training_steps, graphs.NCC_mismatch)
    plt.xlabel("Training Step")
    plt.ylabel("Proportion Mismatch from NCC")
    plt.title("NC4: Convergence to NCC")

    # Plot decomposition of MSE loss
    if loss_name == "MSELoss":
        plt.figure(8)
        plt.clf()
        plt.semilogy(training_steps, graphs.MSE_wd_features)
        plt.semilogy(training_steps, graphs.LNC1)
        plt.semilogy(training_steps, graphs.LNC23)
        plt.semilogy(training_steps, graphs.Lperp)
        plt.legend(["MSE+wd", "LNC1", "LNC2/3", "Lperp"])
        plt.xlabel("Training Step")
        plt.ylabel("Value")
        plt.title("Decomposition of MSE")

    zero_error_step = first_zero_error_step(training_steps, graphs)
    draw_zero_error_bar_on_all_plots(zero_error_step)
    if epoch == epochs:
        save_nc_metrics_evolution_plot(training_steps, graphs, zero_error_step, tpt_accuracy_threshold)
        save_bump_structure_plot(total_steps_seen, bumps_before_tpt, bumps_at_tpt, zero_error_step)
        save_current_figures(epoch)
        plt.show()
    else:
        plt.close("all")

print(f"\n[Training Summary]")
print(f"  Total training steps completed: {training_steps[-1] if training_steps else 0}")
print(f"  Total epochs completed: {epochs}")
print(f"  Steps per epoch (configured): {steps_per_epoch}")
print(f"  Steps per epoch (epoch 1 actual): {training_steps_per_epoch if training_steps_per_epoch else 'N/A'}")
print(f"  TPT reached: {'Yes' if terminal_full_training_reached else 'No'}")
if terminal_full_training_reached and training_steps:
    tpt_step_index = next((i for i, acc in enumerate(graphs.accuracy) if (1.0 - acc) < 1e-12), None)
    if tpt_step_index is not None:
        print(f"  TPT emergence step: {training_steps[tpt_step_index]}")
print(f"  Results saved to: {RESULTS_DIR}")
print()
