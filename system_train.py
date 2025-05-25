# -*- coding: utf-8 -*-

!nvidia-smi

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
from matplotlib import gridspec
import pickle
import sys
from collections import namedtuple
import os
import jax
from jax import grad, jit
from jax.tree_util import tree_map
import jax.numpy as jnp
from jax import random
import numpy.random as npr
import seaborn as sns
from jax.example_libraries import stax
sns.set(style="white")
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec

"""# Pipeline"""

from tqdm import tqdm
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import random, jit, grad, value_and_grad
import optax
from jax import jvp, vmap
from functools import partial
jax.config.update('jax_platform_name', 'gpu')
import imageio
from pathlib import Path
from collections import defaultdict
from scipy.stats import entropy
from scipy.stats import gaussian_kde, entropy
from jax.scipy.special import kl_div
from scipy.interpolate import interp1d


class SpiralClassifier:

    """
    A class for training a neural network to classify points from the spiral dataset
    with dynamic batch training focusing on different classes periodically,
    real-time visualization of the decision boundary, and tracking the evolution of weight differences.
    """
    def __init__(self, points_per_class=100, num_classes=3, nn_width=100, learning_rate=0.01, real_time_visualization=False, vis_step_interval=100, track_weight_diff=False, weight_diff_step_interval=100, track_periodic_weight_diff=False, T=5000, l2_reg = 1e-4, include_biases = False, label = "osc70", show_markers = True):
        self.points_per_class = points_per_class
        self.num_classes = num_classes
        self.nn_width = nn_width
        self.learning_rate = learning_rate
        self.real_time_visualization = real_time_visualization
        self.vis_step_interval = vis_step_interval
        self.track_weight_diff = track_weight_diff
        self.weight_diff_step_interval = weight_diff_step_interval
        self.track_periodic_weight_diff = track_periodic_weight_diff
        self.period_length = T
        self.initial_weights = None
        self.weights_after_period = None
        self.current_weights = None
        self.model = self.create_model()
        self.optimizer = optax.adam(learning_rate)
        self.key = random.PRNGKey(0)
        self.last_params = None
        self.include_biases = include_biases
        self.label = label
        self.highlight_indices = {}
        self.weights_history = []
        self.param_shapes = None
        self.show_markers = show_markers
        self.all_gradients = []
        self.previous_distributions = []
        self.previous_distributions_part1 = []
        self.previous_distributions_part2 = []
        self.distributions_disjoint = []
        self.kl_divergences_part1 = {i: {'12': [], '13': [], '23': []} for i in range(4)}
        self.kl_divergences_part2 = {i: {'12': [], '13': [], '23': []} for i in range(4)}
        self.kl_divergences_disjoint = {i: {'12': [], '13': [], '23': []} for i in range(4)}
        self.gradients_buffer = []
        self.entropies_part1 = {layer: {'1': [], '2': [], '3': []} for layer in range(4)}
        self.entropies_part2 = {layer: {'1': [], '2': [], '3': []} for layer in range(4)}

    def create_model(self):
        """
        Defines the neural network model.

        Returns:
        - A tuple containing the initialization and application functions of the model.
        """
        return stax.serial(
            stax.Dense(self.nn_width), stax.Relu,
            stax.Dense(self.num_classes)
        )

    def initialize_params(self):
        """
        Initializes the model parameters.

        Returns:
        - The initialized parameters of the model.
        """
        _, params = self.model[0](random.PRNGKey(0), (-1, 2))
        return params

    @partial(jit, static_argnums=(0,))
    def make_dataset(self, revolutions=4,seed = None):
        """
        Generates the spiral dataset.

        Parameters:
        - revolutions: Number of spiral revolutions.

        Returns:
        - X: The input features of the dataset.
        - Y: The one-hot encoded labels of the dataset.
        """
        key = random.PRNGKey(seed) if seed is not None else random.PRNGKey(0)
        N, C, pi = self.points_per_class, self.num_classes, jnp.pi
        X = jnp.zeros((N * C, 2))
        Y = jnp.zeros((N * C, C))

        for j in range(C):
            ix = slice(N * j, N * (j + 1))
            r = jnp.linspace(0., 1, N)  # radius
            omega = 2 * pi / C
            theta_max = revolutions * pi

            key, subkey = random.split(key)

            t = jnp.linspace(omega * j, omega * j + theta_max, N) + random.normal(subkey, (N,)) * 0.2
            X = X.at[ix].set(jnp.c_[r*jnp.cos(t), r*jnp.sin(t)])
            Y = Y.at[ix, j].set(1)
        return jax.device_put(X), jax.device_put(Y)

    def c_fn(self, t, i, w_max, T):
        """
        Compute the weights for the loss function at time step t for class i, following a periodic pattern.

        Parameters:
        - t: Current time step.
        - i: Index of the class in focus.
        - w_max: Maximum weight for the class in focus.
        - T: Period length.

        Returns:
        - Normalized weights for all classes.
        """
        slope = 2 * (w_max - 1) / T
        w_main_class = jnp.where(t < T / 2., 1 + t * slope, 2 * w_max - t * slope - 1)
        res = jnp.ones(self.num_classes) + (w_main_class - 1) * jnp.eye(self.num_classes)[i]
        res = res / jnp.sum(res)
        return res

    @partial(jit, static_argnums=(0,))
    def loss_fn(self, params, X, Y):
        """
        Computes the loss for a given batch of data.

        Parameters:
        - params: Model parameters.
        - X: Batch of input features.
        - Y: Batch of one-hot encoded labels.

        Returns:
        - The computed cross-entropy loss for the batch.
        """
        preds = self.model[1](params, X)
        return -jnp.mean(jnp.sum(jax.nn.log_softmax(preds) * Y, axis=-1))

    @partial(jit, static_argnums=(0,))
    def accuracy(self, params, X, Y):
        """
        Compute accuracy
        """
        preds = self.model[1](params, X)
        return jnp.mean(jnp.argmax(preds, axis=1) == jnp.argmax(Y, axis=1))

    def update_step(self, params, opt_state, X_batch, Y_batch):
        """
        Performs a single update step on the model parameters.

        Parameters:
        - params: Current model parameters.
        - opt_state: Current state of the optimizer.
        - X_batch: Batch of input features.
        - Y_batch: Batch of one-hot encoded labels.

        Returns:
        - new_params: Updated model parameters.
        - new_opt_state: Updated state of the optimizer.
        """
        grads = grad(self.loss_fn)(params, X_batch, Y_batch)
        updates, new_opt_state = self.optimizer.update(grads, opt_state,)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, grads


    def calculate_l2_distance(self, weights_a, weights_b): # imp
        """
        Calculates the L2 distance between two sets of weights.

        Parameters:
        - weights_a: The first set of weights.
        - weights_b: The second set of weights.

        Returns:
        - The L2 distance between the two sets of weights.
        """

        squared_diff = tree_map(lambda a, b: jnp.sum((a - b) ** 2), weights_a, weights_b)
        total_squared_diff = sum(jax.tree_util.tree_leaves(squared_diff))
        return total_squared_diff


    def sample_by_class(self, X_train, Y_train, class_counts, num_classes, rng_key, class_indices): 
        indices = []
        for i in range(num_classes):
            rng_key, sub_key = random.split(rng_key)
            sampled_indices = random.choice(sub_key, class_indices[i], shape=(class_counts[i],), replace=True)
            indices.append(sampled_indices)
        final_indices = jnp.concatenate(indices)
        return X_train[final_indices], Y_train[final_indices]


    def plot_decision_boundary(self, params, X_train, Y_train, X_test, Y_test):
        """
        Visualizes the decision boundary of the model by plotting it along with the training data.

        Parameters:
        - params: The parameters of the neural network model used to define the decision boundary.
        """
        x_min, x_max = -1.5, 1.5
        y_min, y_max = -1.5, 1.5
        h = 0.005

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = np.c_[xx.ravel(), yy.ravel()]
        Z = self.model[1](params, jax.device_put(Z))
        Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)

        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 8))
        colors = ['red', 'blue', 'yellow']
        plt.contourf(xx, yy, Z, alpha=0.8, levels=[-1, 0, 1, 2], colors=colors)
        labels = np.array(Y_train.argmax(axis=1))
        class_to_color = {0: 'red', 1: 'blue', 2: 'yellow'}
        point_colors = [class_to_color[label] for label in labels]
        scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=point_colors, s=40, edgecolor='k', marker='o')
        scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=point_colors, s=40, edgecolor='k', marker='*')
        plt.title('Decision Boundary', fontsize=16)

        plt.xticks([])
        plt.yticks([])

        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.show()


    def train(self, X_train, Y_train, X_test, Y_test, T, w_max, total_steps, n_steps_per_epoch, batch_size):
        rng = random.PRNGKey(0)
        rng, key = random.split(rng)
        self.X_train = X_train
        self.Y_train = Y_train
        losses = []
        accuracies = []
        test_losses = []
        test_accuracies = []

        dist_origin = []
        relative_dist = []
        eigv_list = []
        eigv_list_total = []
        n_epochs = int(total_steps / n_steps_per_epoch)
        params = self.initialize_params()
        opt_state = self.optimizer.init(params)
        class_indices = [jnp.where(Y_train[:, i] == 1)[0] for i in range(self.num_classes)]

        distances_initial = {f"layer_{i+1}_initial": [] for i in range(3)}
        distances_last_period = {f"layer_{i+1}_last_period": [] for i in range(3)}
        total_distances_initial = []
        total_distances_last_period = []

        dist_init_combined = []
        dist_last_period_combined= []
        dist_init_class_spec= []
        dist_last_period_class_spec= []
        dist_init= []
        dist_last_period = []

        total_entropies = []
        times = []

        output_dir = Path("cosine_similarity_frames")
        output_dir.mkdir(exist_ok=True)
        filenames = []

        output_dir_hist = Path("hist_evolution")
        output_dir_hist.mkdir(exist_ok=True)
        filenames_hist = []

        output_dir_fisher = Path("fisher_matrix")
        output_dir_fisher.mkdir(exist_ok=True)
        filenames_fisher = []

        output_dir_visuals = Path("visuals")
        output_dir_visuals.mkdir(exist_ok=True)
        filenames_visuals = []

        output_dir_weights_corr = Path("weights_corr")
        output_dir_weights_corr.mkdir(exist_ok=True)
        filenames_weights_corr = []

        output_dir_distributions = Path("disjoint_distributions")
        output_dir_distributions.mkdir(exist_ok=True)
        filenames_distributions = []


        output_dir_distributions_all = Path("disjoint_distributions_all")
        output_dir_distributions_all.mkdir(exist_ok=True)
        filenames_distributions_all = []

        output_dir_distributions_all_split = Path("disjoint_distributions_all_split")
        output_dir_distributions_all_split.mkdir(exist_ok=True)
        filenames_distributions_all_split = []

        initial1, final1 = 2300, 3000
        initial2, final2 = 3500, 5000
        period = 5000

        red_part1_distributions = []
        red_part2_distributions = []
        blue_part1_distributions = []
        blue_part2_distributions = []
        yellow_part1_distributions = []
        yellow_part2_distributions = []

        x_min, x_max = -1.2, 1.2
        y_min, y_max = -1.2, 1.2
        h = 0.005

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        plt.figure(figsize=(8, 8))
        train_colors = ['#8B0000', '#00008B', '#FFD700']
        test_colors = ['#FF6347', '#1E90FF', '#FFD700']
        train_labels = np.array(Y_train.argmax(axis=1))
        test_labels = np.array(Y_test.argmax(axis=1))

        class_to_train_color = {0: train_colors[0], 1: train_colors[1], 2: train_colors[2]}
        class_to_test_color = {0: test_colors[0], 1: test_colors[1], 2: test_colors[2]}
        train_point_colors = [class_to_train_color[label] for label in train_labels]
        test_point_colors = [class_to_test_color[label] for label in test_labels]

        train_scatter = plt.scatter(X_train[:, 0], X_train[:, 1],
                                    c=train_point_colors, s=60, edgecolor='k', marker='o', label='Train Set')

        test_scatter = plt.scatter(X_test[:, 0], X_test[:, 1],
                                  c=test_point_colors, s=80, edgecolor='none', marker='s', label='Test Set', alpha=0.7)

        plt.title('Training and Test Datasets', fontsize=16)

        plt.xticks([])
        plt.yticks([])

        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor='black', markersize=10, label='Training Data'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#D3D3D3', markersize=10, label='Test Data')
        ]

        plt.legend(handles=legend_elements, fontsize=12, loc='upper right', frameon=True,)
        plt.grid(True, alpha=0.7)
        plt.show()

        with tqdm(total=total_steps) as pbar:
            for epoch in range(n_epochs):
                for step in range(n_steps_per_epoch):
                    t = epoch * n_steps_per_epoch + step

                    class_focus = int((t // T) % self.num_classes)
                    current_weights = self.c_fn(t % T, class_focus, w_max, T)
                    class_counts = (current_weights * batch_size).astype(int)

                    rng, key = random.split(rng)
                    X_batch, Y_batch = self.sample_by_class(X_train, Y_train, class_counts, self.num_classes, key, class_indices)
                    params, opt_state, grads = self.update_step(params, opt_state, X_batch, Y_batch)

                    if step == n_steps_per_epoch - 1:
                        train_loss = self.loss_fn(params, X_train, Y_train)
                        losses.append(train_loss)
                        test_loss = self.loss_fn(params, X_test, Y_test)
                        test_losses.append(test_loss)
                        train_accuracy = self.accuracy(params, X_train, Y_train)
                        accuracies.append(train_accuracy)
                        test_accuracy = self.accuracy(params, X_test, Y_test)
                        test_accuracies.append(test_accuracy)

                    pbar.update(1)

        self.last_params = params

        description = f"width_{self.nn_width}-batchsize_{batch_size}-wmax_{w_max}-period_{T}-Nperiods_{int(total_steps/T)}"

        plt.figure(figsize=(30, 5))
        plt.plot(losses, label='Training Loss', color='black', linestyle='-', linewidth=2)
        plt.plot(test_losses, label='Test Loss', color='red', linestyle='--', linewidth=2, alpha = 0.7)
        plt.xlabel('Step', fontsize=14)
        plt.title('Learning Curve', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.7)
        plt.yscale('log')
        plt.show()

        with open(f'losses_{description}.pkl', 'wb') as f:
            pickle.dump(losses, f)

        plt.figure(figsize=(30, 5))
        plt.plot(accuracies, label='Training Accuracy', color='black', linestyle='-', linewidth=2)
        plt.plot(test_accuracies, label='Test Accuracy', color='red', linestyle='--', linewidth=2, alpha = 0.7)
        plt.xlabel('Step', fontsize=14)
        plt.title('Accuracy Curve', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.7)
        plt.show()

        with open(f'accuracies_{description}.pkl', 'wb') as f:
            pickle.dump(accuracies, f)

        return losses, accuracies

"""## bs=50 (small)"""

w_max_values = [1, 50, 100, 150]
T_values = [100, 500, 1000, 5000]

for w in w_max_values:
    if w == 1:
        Ts = [5000]
    else:
        Ts = T_values
    for T in Ts:
        print(f"\nRunning with w_max = {w} and T = {T} ----------------------------\n")
        classifier = SpiralClassifier(
            points_per_class=100,
            num_classes=3,
            nn_width=50,
            learning_rate=1,
            real_time_visualization=True,
            vis_step_interval=100,
            track_periodic_weight_diff=True,
            T=T,
            track_weight_diff=True,
            weight_diff_step_interval=1,
            l2_reg=0.0,
            label="testing",
            show_markers=True
        )

        X_train, Y_train = classifier.make_dataset(seed=0)
        test_data_generator = SpiralClassifier(points_per_class=100, num_classes=3)
        X_test, Y_test = test_data_generator.make_dataset(seed=1)
        total_steps =  15 * 5000 + 1
        n_steps_per_epoch = 1
        batch_size = 50
        classifier.train(
            X_train,
            Y_train,
            X_test,
            Y_test,
            T=T,
            w_max=w,
            total_steps=total_steps,
            n_steps_per_epoch=n_steps_per_epoch,
            batch_size=batch_size
        )
        test_accuracy = classifier.accuracy(classifier.last_params, X_test, Y_test)
        print(f"Test Accuracy: {test_accuracy * 100:.4f}%\n")

"""## bs=200 (large)"""

w_max_values = [1, 50, 100, 150]
T_values = [100, 500, 1000, 5000]

for w in w_max_values:
    if w == 1:
        Ts = [5000]
    else:
        Ts = T_values
    for T in Ts:
        print(f"\nRunning with w_max = {w} and T = {T} ----------------------------\n")
        classifier = SpiralClassifier(
            points_per_class=100,
            num_classes=3,
            nn_width=50,
            learning_rate=1,
            real_time_visualization=True,
            vis_step_interval=100,
            track_periodic_weight_diff=True,
            T=T,
            track_weight_diff=True,
            weight_diff_step_interval=1,
            l2_reg=0.0,
            label="testing",
            show_markers=True
        )
        X_train, Y_train = classifier.make_dataset(seed=0)
        test_data_generator = SpiralClassifier(points_per_class=100, num_classes=3)
        X_test, Y_test = test_data_generator.make_dataset(seed=1)
        total_steps =  15 * 5000 + 1
        n_steps_per_epoch = 1
        batch_size = 200
        classifier.train(
            X_train,
            Y_train,
            X_test,
            Y_test,
            T=T,
            w_max=w,
            total_steps=total_steps,
            n_steps_per_epoch=n_steps_per_epoch,
            batch_size=batch_size
        )
        # test accuracy
        test_accuracy = classifier.accuracy(classifier.last_params, X_test, Y_test)
        print(f"Test Accuracy: {test_accuracy * 100:.4f}%\n")

"""## bs=50 (small)"""

w_max_values = [1, 50, 100, 150]
T_values = [100, 500, 1000, 5000]

for w in w_max_values:
    if w == 1:
        Ts = [5000]
    else:
        Ts = T_values
    for T in Ts:
        print(f"\nRunning with w_max = {w} and T = {T} ----------------------------\n")
        classifier = SpiralClassifier(
            points_per_class=100,
            num_classes=3,
            nn_width=50,
            learning_rate=0.002,
            real_time_visualization=True,
            vis_step_interval=100,
            track_periodic_weight_diff=True,
            T=T,
            track_weight_diff=True,
            weight_diff_step_interval=1,
            l2_reg=0.0,
            label="testing",
            show_markers=True
        )
        X_train, Y_train = classifier.make_dataset(seed=0)
        test_data_generator = SpiralClassifier(points_per_class=100, num_classes=3)
        X_test, Y_test = test_data_generator.make_dataset(seed=1)
        total_steps =  15 * 5000 + 1
        n_steps_per_epoch = 1
        batch_size = 50
        classifier.train(
            X_train,
            Y_train,
            X_test,
            Y_test,
            T=T,
            w_max=w,
            total_steps=total_steps,
            n_steps_per_epoch=n_steps_per_epoch,
            batch_size=batch_size
        )
        test_accuracy = classifier.accuracy(classifier.last_params, X_test, Y_test)
        print(f"Test Accuracy: {test_accuracy * 100:.4f}%\n")

"""## bs=200 (large)"""

w_max_values = [1, 50, 100, 150]
T_values = [100, 500, 1000, 5000]

for w in w_max_values:
    if w == 1:
        Ts = [5000]
    else:
        Ts = T_values
    for T in Ts:
        print(f"\nRunning with w_max = {w} and T = {T} ----------------------------\n")
        classifier = SpiralClassifier(
            points_per_class=100,
            num_classes=3,
            nn_width=50,
            learning_rate=0.002,
            real_time_visualization=True,
            vis_step_interval=100,
            track_periodic_weight_diff=True,
            T=T,
            track_weight_diff=True,
            weight_diff_step_interval=1,
            l2_reg=0.0,
            label="testing",
            show_markers=True
        )
        X_train, Y_train = classifier.make_dataset(seed=0)
        test_data_generator = SpiralClassifier(points_per_class=100, num_classes=3)
        X_test, Y_test = test_data_generator.make_dataset(seed=1)
        total_steps =  15 * 5000 + 1
        n_steps_per_epoch = 1
        batch_size = 200
        classifier.train(
            X_train,
            Y_train,
            X_test,
            Y_test,
            T=T,
            w_max=w,
            total_steps=total_steps,
            n_steps_per_epoch=n_steps_per_epoch,
            batch_size=batch_size
        )
        test_accuracy = classifier.accuracy(classifier.last_params, X_test, Y_test)
        print(f"Test Accuracy: {test_accuracy * 100:.4f}%\n")