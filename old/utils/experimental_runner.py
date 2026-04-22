from collections import OrderedDict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from .config_wrapper import RunnerSettings
    from .data_generation import build_dataloaders
    from .model_factory import create_model
    from .training_core import train_epoch, evaluate, analysis, compute_class_weights
    from .metrics_reporting import (
        Graphs,
        init_results_csv,
        append_results_csv_row,
        first_zero_error_step,
        draw_zero_error_bar_on_all_plots,
        save_bump_structure_plot,
        save_current_figures,
    )
except ImportError:
    # Fallback for direct script execution: python utils/experimental_runner.py
    from old.utils.config_wrapper import RunnerSettings
    from old.utils.data_generation import build_dataloaders
    from old.utils.model_factory import create_model
    from old.utils.training_core import train_epoch, evaluate, analysis, compute_class_weights
    from old.utils.metrics_reporting import (
        Graphs,
        init_results_csv,
        append_results_csv_row,
        first_zero_error_step,
        draw_zero_error_bar_on_all_plots,
        save_bump_structure_plot,
        save_current_figures,
    )


def run_experiment() -> None:
    settings = RunnerSettings.from_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[Runner Configuration]")
    print(f"  Debug mode: {settings.debug}")
    print(f"  Train batch size: {settings.batch_size}")
    print(f"  Eval batch size: {settings.eval_batch_size}")
    print(f"  Steps per epoch: {settings.steps_per_epoch}")
    print()

    model, classifier, feature_store = create_model(settings, device)
    train_loader, analysis_loader, test_loader = build_dataloaders(settings)

    if settings.loss_name == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
        criterion_summed = nn.CrossEntropyLoss(reduction="sum")
    else:
        criterion = nn.MSELoss()
        criterion_summed = nn.MSELoss(reduction="sum")

    if settings.optimizer_type.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=settings.learning_rate,
            betas=(settings.beta1, settings.beta2),
            eps=settings.eps,
            weight_decay=settings.weight_decay,
        )
    elif settings.optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=settings.learning_rate,
            momentum=settings.momentum,
            weight_decay=settings.weight_decay,
        )
    elif settings.optimizer_type.lower() == "rmsprop":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=settings.learning_rate,
            momentum=settings.momentum,
            weight_decay=settings.weight_decay,
            eps=settings.eps,
        )
    else:
        raise ValueError(f"Unsupported optimizer_type: {settings.optimizer_type}")

    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=settings.epochs_lr_decay,
        gamma=settings.lr_decay,
    )

    graphs = Graphs()
    training_steps = []

    global_step = 0
    total_steps_seen = 0
    terminal_full_training_reached = False

    init_results_csv(settings.results_dir)

    for epoch in range(1, settings.epochs + 1):
        if global_step >= settings.total_training_steps:
            break

        apply_bumping = settings.bumps_at_tpt if terminal_full_training_reached else settings.bumps_before_tpt

        steps_remaining = settings.total_training_steps - global_step
        steps_this_epoch = min(settings.steps_per_epoch, steps_remaining)

        train_loss, train_acc, global_step, _class_focus, processed_batches = train_epoch(
            model,
            criterion,
            device,
            settings.num_classes,
            train_loader,
            optimizer,
            settings.batch_size,
            global_step,
            apply_bumping,
            settings.bump_period_length,
            settings.bump_w_max,
            steps_this_epoch,
        )
        total_steps_seen += processed_batches

        if not terminal_full_training_reached and train_acc >= settings.tpt_accuracy_threshold:
            terminal_full_training_reached = True
            print(f"[TPT reached at step {global_step}]")

        lr_scheduler.step()

        training_steps.append(global_step)
        analysis(
            graphs,
            model,
            criterion_summed,
            device,
            settings.num_classes,
            analysis_loader,
            feature_store,
            classifier,
            settings.weight_decay,
            settings.loss_name,
        )
        test_loss, test_acc = evaluate(model, criterion, device, settings.num_classes, test_loader)
        graphs.train_loss_epoch.append(train_loss)
        graphs.train_acc_epoch.append(train_acc)
        graphs.test_loss.append(test_loss)
        graphs.test_acc.append(test_acc)

        idx = len(training_steps) - 1
        append_results_csv_row(settings.results_dir, global_step, graphs, idx)

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

        zero_error_step = first_zero_error_step(training_steps, graphs)
        draw_zero_error_bar_on_all_plots(zero_error_step)

        if epoch == settings.epochs:
            save_bump_structure_plot(
                settings.results_dir,
                total_steps_seen,
                settings.num_classes,
                settings.bump_period_length,
                settings.bump_w_max,
                settings.bumps_before_tpt,
                settings.bumps_at_tpt,
                lambda t, f, w, p, n: compute_class_weights(t, f, w, p, n, device),
                zero_error_step,
            )
            save_current_figures(settings.results_dir, epoch)
            plt.show()
        else:
            plt.close("all")

    print("\n[Training Summary]")
    print(f"  Total training steps completed: {training_steps[-1] if training_steps else 0}")
    print(f"  Total epochs completed: {settings.epochs}")
    print(f"  Steps per epoch (configured): {settings.steps_per_epoch}")
    print(f"  TPT reached: {'Yes' if terminal_full_training_reached else 'No'}")
    print(f"  Results saved to: {settings.results_dir}")


if __name__ == "__main__":
    run_experiment()
