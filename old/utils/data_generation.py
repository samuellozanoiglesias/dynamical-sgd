import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from .config_wrapper import RunnerSettings


def generate_spiral_data(
    points_per_class: int,
    num_classes: int,
    revolutions: float,
    noise_std: float,
    random_seed: int,
    angular_offsets=None,
    randomize_offsets: bool = False,
    min_radius: float = 0.05,
):
    rng = np.random.default_rng(random_seed)
    n = points_per_class
    c = num_classes

    if angular_offsets is not None:
        if len(angular_offsets) != c:
            raise ValueError(f"angular_offsets must have {c} values")
        offsets = np.deg2rad(np.array(angular_offsets, dtype=np.float32))
    elif randomize_offsets:
        offsets = rng.uniform(0.0, 2.0 * np.pi, size=c).astype(np.float32)
    else:
        offsets = np.array([2.0 * np.pi * j / c for j in range(c)], dtype=np.float32)

    x_all = np.zeros((n * c, 2), dtype=np.float32)
    y_all = np.zeros((n * c,), dtype=np.int64)

    for j in range(c):
        ix = slice(n * j, n * (j + 1))
        r = np.linspace(min_radius, 1.0, n, dtype=np.float32)
        theta = np.linspace(offsets[j], offsets[j] + revolutions * np.pi, n, dtype=np.float32)
        theta += rng.normal(0.0, noise_std, size=n).astype(np.float32)
        x_all[ix, 0] = r * np.cos(theta)
        x_all[ix, 1] = r * np.sin(theta)
        y_all[ix] = j

    return x_all, y_all


def create_train_test_split(x, y, test_ratio=0.25, random_seed=0):
    rng = np.random.default_rng(random_seed)
    classes = np.unique(y)

    train_idx = []
    test_idx = []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_test = int(len(idx) * test_ratio)
        test_idx.append(idx[:n_test])
        train_idx.append(idx[n_test:])

    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]


def save_mnist_visualizations(train_dataset, test_dataset, out_dir):
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
    plt.savefig(out_dir / "training_dataset_samples.png", dpi=150, bbox_inches="tight")
    plt.close()

    train_counts = np.bincount(train_dataset.targets.numpy(), minlength=10)
    test_counts = np.bincount(test_dataset.targets.numpy(), minlength=10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(np.arange(10), train_counts, alpha=0.8, color="skyblue")
    ax1.set_title("Train Class Distribution")
    ax2.bar(np.arange(10), test_counts, alpha=0.8, color="salmon")
    ax2.set_title("Test Class Distribution")
    for ax in (ax1, ax2):
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "dataset_statistics.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_spiral_visualizations(x_train, y_train, x_test, y_test, num_classes, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for c in range(num_classes):
        m = y_train == c
        if np.any(m):
            axes[0].scatter(x_train[m, 0], x_train[m, 1], s=6, alpha=0.6, label=f"Class {c}")
    axes[0].set_title("Spiral Training Samples")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=2, fontsize=8)

    for c in range(num_classes):
        m = y_test == c
        if np.any(m):
            axes[1].scatter(x_test[m, 0], x_test[m, 1], s=6, alpha=0.6, label=f"Class {c}")
    axes[1].set_title("Spiral Test Samples")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=2, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "spiral_dataset_samples.png", dpi=150, bbox_inches="tight")
    plt.close()


def build_dataloaders(settings: RunnerSettings):
    if settings.dataset_name == "mnist":
        if settings.model_architecture == "resnet18":
            transform = transforms.Compose(
                [
                    transforms.Pad((settings.padded_im_size - settings.im_size) // 2),
                    transforms.ToTensor(),
                    transforms.Normalize(0.1307, 0.3081),
                ]
            )
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)])

        train_dataset = datasets.MNIST(str(settings.data_dir), train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(str(settings.data_dir), train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True)
        analysis_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=settings.eval_batch_size,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=settings.eval_batch_size,
            shuffle=False,
        )

        raw_train = datasets.MNIST(str(settings.data_dir), train=True, download=True)
        raw_test = datasets.MNIST(str(settings.data_dir), train=False, download=True)
        save_mnist_visualizations(raw_train, raw_test, settings.results_dir)
        return train_loader, analysis_loader, test_loader

    x_full, y_full = generate_spiral_data(
        points_per_class=settings.spiral_points_per_class,
        num_classes=settings.num_classes,
        revolutions=settings.spiral_revolutions,
        noise_std=settings.spiral_noise_std,
        random_seed=settings.spiral_random_seed,
        angular_offsets=settings.spiral_angular_offsets,
        randomize_offsets=settings.spiral_randomize_offsets,
        min_radius=settings.spiral_min_radius,
    )

    x_train, y_train, x_test, y_test = create_train_test_split(
        x_full,
        y_full,
        test_ratio=settings.spiral_test_ratio,
        random_seed=settings.spiral_random_seed,
    )

    if settings.model_architecture == "resnet18":
        x_train_t = torch.from_numpy(x_train).view(-1, 2, 1, 1)
        x_test_t = torch.from_numpy(x_test).view(-1, 2, 1, 1)
    else:
        x_train_t = torch.from_numpy(x_train)
        x_test_t = torch.from_numpy(x_test)

    train_dataset = torch.utils.data.TensorDataset(x_train_t, torch.from_numpy(y_train))
    test_dataset = torch.utils.data.TensorDataset(x_test_t, torch.from_numpy(y_test))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True)
    analysis_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=settings.eval_batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=settings.eval_batch_size,
        shuffle=False,
    )

    save_spiral_visualizations(x_train, y_train, x_test, y_test, settings.num_classes, settings.results_dir)
    return train_loader, analysis_loader, test_loader
