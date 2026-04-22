import numpy as np
import matplotlib.pyplot as plt


class Graphs:
    def __init__(self):
        self.accuracy = []
        self.loss = []
        self.reg_loss = []
        self.train_loss_epoch = []
        self.train_acc_epoch = []
        self.test_loss = []
        self.test_acc = []

        self.Sw_invSb = []

        self.norm_M_CoV = []
        self.norm_W_CoV = []
        self.cos_M_std = []
        self.cos_W_std = []
        self.cos_M = []
        self.cos_W = []

        self.W_M_dist = []
        self.NCC_mismatch = []

        self.mu_c_norm_avg = []
        self.mu_G_norm = []
        self.M_fro_norm = []
        self.W_fro_norm = []

        self.MSE_wd_features = []
        self.LNC1 = []
        self.LNC23 = []
        self.Lperp = []


def init_results_csv(results_dir):
    out_csv = results_dir / "nc_results.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(
            "TrainingStep,RegLoss,TrainError,TrainLossEpoch,TrainAccEpoch,TestLoss,TestAcc,NC1,NC2_Means_Equinorm,NC2_Classifiers_Equinorm,"
            "NC2_Means_Equiangularity,NC2_Classifiers_Equiangularity,NC3,NC4\n"
        )


def append_results_csv_row(results_dir, step, graphs, index):
    out_csv = results_dir / "nc_results.csv"
    with open(out_csv, "a", encoding="utf-8") as f:
        train_error = 1.0 - float(graphs.accuracy[index])
        row = [
            str(step),
            f"{float(graphs.reg_loss[index]):.10f}",
            f"{train_error:.10f}",
            f"{float(graphs.train_loss_epoch[index]):.10f}",
            f"{float(graphs.train_acc_epoch[index]):.10f}",
            f"{float(graphs.test_loss[index]):.10f}",
            f"{float(graphs.test_acc[index]):.10f}",
            f"{float(graphs.Sw_invSb[index]):.10f}",
            f"{float(graphs.norm_M_CoV[index]):.10f}",
            f"{float(graphs.norm_W_CoV[index]):.10f}",
            f"{float(graphs.cos_M[index]):.10f}",
            f"{float(graphs.cos_W[index]):.10f}",
            f"{float(graphs.W_M_dist[index]):.10f}",
            f"{float(graphs.NCC_mismatch[index]):.10f}",
        ]
        f.write(",".join(row) + "\n")


def save_current_figures(results_dir, epoch):
    names = {
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
        out = results_dir / f"{names.get(fig_num, f'figure_{fig_num}')}_epoch_{epoch:03d}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")


def first_zero_error_step(training_steps, graphs):
    for step, acc in zip(training_steps, graphs.accuracy):
        if np.isclose(1.0 - float(acc), 0.0, atol=1e-12):
            return step
    return None


def draw_zero_error_bar_on_all_plots(zero_error_step):
    if zero_error_step is None:
        return
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        for ax in fig.get_axes():
            ax.axvline(x=zero_error_step, color="red", linestyle="--", linewidth=1.5, alpha=0.9)


def save_bump_structure_plot(results_dir, total_steps, num_classes, bump_period_length, bump_w_max, bumps_before_tpt, bumps_at_tpt, compute_class_weights, tpt_emergence_step=None):
    if total_steps <= 0:
        return
    steps = np.arange(total_steps)
    weights_over_time = np.zeros((num_classes, total_steps), dtype=np.float32)

    for s in steps:
        if tpt_emergence_step is None:
            apply = bumps_before_tpt
        else:
            apply = bumps_before_tpt if s < tpt_emergence_step else bumps_at_tpt

        if apply:
            focus = (s // bump_period_length) % num_classes
            w = compute_class_weights(s % bump_period_length, focus, bump_w_max, bump_period_length, num_classes)
            weights_over_time[:, s] = w.detach().cpu().numpy()
        else:
            weights_over_time[:, s] = 1.0 / num_classes

    plt.figure(figsize=(14, 6))
    for c in range(num_classes):
        plt.plot(steps, weights_over_time[c], linewidth=1.0, label=f"Class {c}")
    if tpt_emergence_step is not None:
        plt.axvline(x=tpt_emergence_step, color="red", linestyle="--", linewidth=2.0, alpha=0.7)
    plt.title("Dynamic Class Focus Weights Over Steps")
    plt.xlabel("Training Step")
    plt.ylabel("Sampling Weight")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3, fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(results_dir / "class_focus_dynamics.png", dpi=150, bbox_inches="tight")
    plt.close()
