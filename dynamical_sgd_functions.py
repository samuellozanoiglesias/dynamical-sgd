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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec
sns.set(style="darkgrid")

"""# Pipeline"""

from tqdm import tqdm
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import random, jit, grad, value_and_grad
import optax
from jax import jvp, vmap
from functools import partial
jax.config.update('jax_platform_name', 'cpu')

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
        self.all_weights = []
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
            #stax.Dropout(self.dropout_rate),
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
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        flat_weights = self.flatten_params(new_params)
        self.weights_history.append(flat_weights)
        return new_params, new_opt_state, grads


    def calculate_l2_distance(self, weights_a, weights_b):
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

    def compare_weights_layer_all(self, current_step, params):
        """
        Handles specific layer operations, concatenates class-specific weights with thresholds,
        and computes distances for concatenated vectors with the first layer.
        """
        if current_step == 0:
            self.initial_weights = params
            self.weights_after_period = params
            return {
            "dist_init_combined": [0, 0, 0],
            "dist_last_period_combined": [0, 0, 0],

            "dist_init_class_spec": [0, 0, 0],
            "dist_last_period_class_spec": [0, 0, 0],

            "dist_init": [0, 0],
            "dist_last_period": [0, 0],
            }

        dist_init_combined = []
        dist_last_period_combined = []
        dist_init_class_spec = []
        dist_last_period_class_spec = []
        dist_init = []
        dist_last_period = []
        dist_init_whole_layer = []
        dist_las_period_whole_layer = []

        if current_step % self.weight_diff_step_interval == 0 and current_step != 0:
            for i, (current_layer, initial_layer) in enumerate(zip(params, self.initial_weights)):
              if i != 1:
                layer_distance_from_initial = self.calculate_l2_distance(current_layer, initial_layer)
                dist_init.append(layer_distance_from_initial.item())

              if i == len(params) - 1:  # last layer
                  for j in range(current_layer[0].shape[1]):  # number of classes
                      class_spec_weights = current_layer[0][:, j]
                      class_spec_threshold = current_layer[1][j]

                      class_spec_params = jnp.concatenate([class_spec_weights, jnp.array([class_spec_threshold])])
                      first_layer = jnp.concatenate([params[0][0].flatten(), params[0][1]])
                      combined_class_spec = jnp.concatenate([first_layer, class_spec_params])

                      initial_class_spec_params = jnp.concatenate([self.initial_weights[-1][0][:, j], jnp.array([self.initial_weights[-1][1][j]])])
                      initial_combined_class_spec = jnp.concatenate([self.initial_weights[0][0].flatten(), self.initial_weights[0][1], initial_class_spec_params])

                      dist_from_initial_combined_class_spec = self.calculate_l2_distance(combined_class_spec, initial_combined_class_spec)
                      dist_from_initial_class_spec = self.calculate_l2_distance(class_spec_params, initial_class_spec_params)

                      dist_init_combined.append(dist_from_initial_combined_class_spec)
                      dist_init_class_spec.append(dist_from_initial_class_spec)

            if self.weights_after_period is not None:
              for i, (current_layer, last_period_layer) in enumerate(zip(params, self.weights_after_period)):
                if i != 1:
                  layer_distance_from_last_period = self.calculate_l2_distance(current_layer, last_period_layer)
                  dist_last_period.append(layer_distance_from_last_period)

                if i == len(params) - 1:  # last layer
                  for j in range(current_layer[0].shape[1]):  # number of classes
                      class_spec_weights = current_layer[0][:, j]
                      class_spec_threshold = current_layer[1][j]

                      class_spec_params = jnp.concatenate([class_spec_weights, jnp.array([class_spec_threshold])])
                      first_layer = jnp.concatenate([params[0][0].flatten(), params[0][1]])
                      combined_class_spec = jnp.concatenate([first_layer, class_spec_params])

                      last_period_class_spec_params = jnp.concatenate([self.weights_after_period[-1][0][:, j], jnp.array([self.weights_after_period[-1][1][j]])])
                      last_period_combined_class_spec = jnp.concatenate([self.weights_after_period[0][0].flatten(), self.weights_after_period[0][1], last_period_class_spec_params])

                      dist_from_last_period_combined_class_spec = self.calculate_l2_distance(combined_class_spec, last_period_combined_class_spec)
                      dist_from_last_period_class_spec = self.calculate_l2_distance(class_spec_params, last_period_class_spec_params)

                      dist_last_period_combined.append(dist_from_last_period_combined_class_spec)
                      dist_last_period_class_spec.append(dist_from_last_period_class_spec)

            else:
              distances_from_last_period = [0] * len(params)
              dist_last_period.append(layer_distance_from_last_period)

        self.weights_after_period = params

        return {
            "dist_init_combined": dist_init_combined,
            "dist_last_period_combined": dist_last_period_combined,

            "dist_init_class_spec": dist_init_class_spec,
            "dist_last_period_class_spec": dist_last_period_class_spec,

            "dist_init": dist_init,
            "dist_last_period": dist_last_period,
            }


    def plot_decision_boundary(self, params):
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
        labels = np.array(self.Y_train.argmax(axis=1))
        class_to_color = {0: 'red', 1: 'blue', 2: 'yellow'}
        point_colors = [class_to_color[label] for label in labels]
        scatter = plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=point_colors, s=40, edgecolor='k', marker='o')

        plt.title('Decision Boundary', fontsize=16)
        plt.xticks([])
        plt.yticks([])

        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.show()


    def sample_by_class(self, X_train, Y_train, class_counts, num_classes, rng_key, class_indices): # imp
        indices = []
        for i in range(num_classes):
            rng_key, sub_key = random.split(rng_key)
            sampled_indices = random.choice(sub_key, class_indices[i], shape=(class_counts[i],), replace=True)
            indices.append(sampled_indices)
        final_indices = jnp.concatenate(indices)
        return X_train[final_indices], Y_train[final_indices]

    '''

    Internal reconfiguration visuals

    '''

    def visualize_layer(self, layer_params, layer_grads, layer_index):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))


        weights = layer_params.flatten()
        axes[0].plot(weights, ".")
        axes[0].set_title(f'Layer {layer_index} Weights')

        gradients = layer_grads.flatten()
        axes[1].plot(gradients, ".")
        axes[1].set_title(f'Layer {layer_index} Gradients')

        plt.tight_layout()
        plt.show()

    def visualize_training(self, params, grads):
        for i, (layer_params, layer_grads) in enumerate(zip(params, grads)):
            if isinstance(layer_params, (tuple, list)):
                for j, (p, g) in enumerate(zip(layer_params, layer_grads)):
                    self.visualize_layer(p, g, f"{i}-{j}")
            else:
                self.visualize_layer(layer_params, layer_grads, i)


    def visualize_layer_colors(self, layer_params, layer_grads, layer_index, fig, axes, index, highlight_indices=None):
        layer_params_2d = layer_params.reshape(layer_params.shape) if len(layer_params.shape) > 1 else layer_params.reshape(1, -1)
        layer_grads_2d = layer_grads.reshape(layer_grads.shape) if len(layer_grads.shape) > 1 else layer_grads.reshape(1, -1)

        norm_params = (layer_params_2d - jnp.mean(layer_params_2d)) / jnp.max(jnp.abs(layer_params_2d))
        norm_grads = (layer_grads_2d - jnp.mean(layer_grads_2d)) / jnp.max(jnp.abs(layer_grads_2d))

        row, col = divmod(index, 6)

        im1 = axes[row, col].imshow(norm_params, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[row, col].set_title(f'Layer {layer_index} Weights')
        axes[row, col].grid(False)

        row, col = divmod(index + 1, 6)
        im2 = axes[row, col].imshow(norm_grads, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[row, col].set_title(f'Layer {layer_index} Gradients')
        axes[row, col].grid(False)


    def visualize_biggest_gradients(self, layer_params, layer_grads, fig, axes, index, step, N, m_biggest, subset_index,):
        layer_params_2d = layer_params.reshape(layer_params.shape) if len(layer_params.shape) > 1 else layer_params.reshape(1, -1)
        norm_params = (layer_params_2d - jnp.mean(layer_params_2d)) / jnp.max(jnp.abs(layer_params_2d))

        color_sequence = ['red', 'blue', 'yellow']
        color_index = ((step-N) // 5000) % 3
        color = color_sequence[color_index]

        if (step-N) % 5000 == 0 and step != 0:
            sorted_indices = jnp.argsort(jnp.abs(layer_grads.flatten()))[::-1]
            top_indices = sorted_indices[:m_biggest]
            new_highlight_indices = {int(idx): color for idx in top_indices}
            if subset_index not in self.highlight_indices:
                self.highlight_indices[subset_index] = {}
            self.highlight_indices[subset_index].update(new_highlight_indices)

        row, col = divmod(index + 2, 6)
        im = axes[row, col].imshow(norm_params, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[row, col].grid(False)
        fig.colorbar(im, ax=axes[row, col])

        if self.show_markers:
          for idx, color in self.highlight_indices.get(subset_index, {}).items():
              row_idx, col_idx = divmod(idx, norm_params.shape[1])
              axes[row, col].scatter([col_idx], [row_idx], color=color, s=100, marker='x', alpha=0.9)

    def visualize_training_colors(self, params, grads, step, output_dir, output_dir_weights_corr, filenames, filenames_weights_corr, N=3000, m_biggest=5,):
        sns.set(style="darkgrid")
        fig, axes = plt.subplots(2, 6, figsize=(30, 10))
        fig.suptitle(f'Step {step}', fontsize=16)

        layer_index = 0
        for i, (layer_params, layer_grads) in enumerate(zip(params, grads)):
            if isinstance(layer_params, (tuple, list)):
                for j, (p, g) in enumerate(zip(layer_params, layer_grads)):
                    subset_index = f"{i}-{j}"
                    if (step-N) % 5000 == 0 and step != 0:
                        self.visualize_biggest_gradients(p, g, fig, axes, layer_index * 3, step, N, m_biggest, subset_index)
                    self.visualize_layer_colors(p, g, f"{i}-{j}", fig, axes, layer_index * 3, self.highlight_indices.get(subset_index))
                    layer_index += 1
            else:
                subset_index = f"{i}"
                if (step-N) % 5000 == 0 and step != 0:
                    self.visualize_biggest_gradients(layer_params, layer_grads, fig, axes, layer_index * 3, step, N, m_biggest, subset_index)
                self.visualize_layer_colors(layer_params, layer_grads, i, fig, axes, layer_index * 3, self.highlight_indices.get(subset_index))
                layer_index += 1

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = output_dir / f"step_{step}.png"
        plt.savefig(filename)
        filenames.append(filename)
        plt.show()

        self.visualize_correlation_matrix(step, params, output_dir_weights_corr, filenames_weights_corr)

    '''

    visualize correlation matrix for the weights evolution

    '''

    def flatten_params(self, params, return_shapes=False):
        flat_params = []
        param_shapes = []
        for layer_params in params:
            if isinstance(layer_params, (tuple, list)):
                for p in layer_params:
                    if p.size > 0:  # Skip empty parameters
                        flat_params.append(p.flatten())
                        param_shapes.append(p.shape)
            elif layer_params.size > 0:  # Skip empty parameters
                flat_params.append(layer_params.flatten())
                param_shapes.append(layer_params.shape)
        if return_shapes:
            return jnp.concatenate(flat_params), param_shapes
        else:
            return jnp.concatenate(flat_params)


    def compute_correlation_matrix(self, only_T_prev=None):

        if only_T_prev is not None and len(self.weights_history) > only_T_prev:
            weights_matrix = jnp.stack(self.weights_history[-only_T_prev:])
        else:
            weights_matrix = jnp.stack(self.weights_history)

        correlation_matrix = jnp.corrcoef(weights_matrix.T)
        return correlation_matrix


    def visualize_correlation_matrix(self, step, params, output_dir, filenames):
        flat_params = self.flatten_params(params)
        correlation_matrix = self.compute_correlation_matrix()

        plt.figure(figsize=(20, 20))
        sns.heatmap(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, annot=False, square=True, cbar_kws={'aspect': 50, 'shrink': 0.8, 'pad': 0.01})
        plt.title(f'correlation matrix for the weights evolution at step {step}', fontsize=20, pad=20)
        plt.grid(False)

        num_params = correlation_matrix.shape[0]
        tick_labels = []
        nested_labels = []
        position = 0

        if self.param_shapes is not None:
            for idx, shape in enumerate(self.param_shapes):
                num_elements = int(jnp.prod(jnp.array(shape)))
                center_pos = position + num_elements // 2
                if idx == 0:
                    nested_labels.append(('weights Layer 1', center_pos, num_elements))
                    tick_labels.extend([f'{i}' for i in range(position, position + num_elements)])
                elif idx == 1:
                    nested_labels.append(('biases Layer 1', center_pos, num_elements))
                    tick_labels.extend([f'{i}' for i in range(position, position + num_elements)])
                elif idx == 2:
                    nested_labels.append(('weights Layer 2', center_pos, num_elements))
                    tick_labels.extend([f'{i}' for i in range(position, position + num_elements)])
                else:
                    nested_labels.append(('biases Layer 2', center_pos, num_elements))
                    tick_labels.extend([f'{i}' for i in range(position, position + num_elements)])
                position += num_elements

            ticks = jnp.arange(0, num_params, 10)
            plt.xticks(ticks=ticks, labels=[tick_labels[i] for i in ticks], rotation=90, fontsize=10)
            plt.yticks(ticks=ticks, labels=[tick_labels[i] for i in ticks], rotation=0, fontsize=10)

        for label, pos, num_elements in nested_labels:

            plt.annotate('', xy=(pos - num_elements // 2, num_params + 12), xytext=(pos + num_elements // 2, num_params + 12),
                         arrowprops=dict(arrowstyle='|-|', lw=1.5, color='black'), annotation_clip=False)
            plt.annotate('', xy=(-12, pos - num_elements // 2), xytext=(-12, pos + num_elements // 2),
                         arrowprops=dict(arrowstyle='|-|', lw=1.5, color='black'), annotation_clip=False)

            plt.text(pos, num_params + 20, label, rotation=0, horizontalalignment='center', fontsize=12, color='black')
            plt.text(-20, pos, label, rotation=90, verticalalignment='center', fontsize=12, color='black')

        plt.tight_layout(pad=3.0)
        filename = output_dir / f"correlation_matrix_step_{step}.png"
        plt.savefig(filename)
        filenames.append(filename)
        plt.close()

    '''

    KL div

    '''

    def normalize(self, layer_grads):
      layer_grads = jnp.array(layer_grads)
      layer_grads_2d = layer_grads.reshape(1, -1)
      norm_grads = layer_grads_2d
      return norm_grads


    def save_gradients_blocks(self, grads):
      grads_block = []
      for i, layer_grads in enumerate(grads):
        if isinstance(layer_grads, (tuple, list)):
          for j, g in enumerate(layer_grads):
            grads_block.append(self.normalize(g))
      self.all_gradients.append(grads_block)


    def save_weights_blocks(self, params):
      weights_block = []
      for i, layer_grads in enumerate(params):
        if isinstance(layer_grads, (tuple, list)):
          for j, g in enumerate(layer_grads):
            weights_block.append(self.normalize(g))
      self.all_weights.append(weights_block)


    def softmax(self, x):
        exp_x = jnp.exp(x - jnp.max(x))
        return exp_x / jnp.sum(exp_x)

    def compute_mean_gradients(self, initial, final, current_period, period, KL = False):
      initial = current_period * period + initial
      final = current_period * period + final
      selected = self.all_gradients[initial:final]

      mean_gradients = []

      for layer_idx in range(4):

          layer_gradients = [step[layer_idx] for step in selected]

          stacked_gradients = jnp.stack(layer_gradients, axis=0)

          mean_gradient = jnp.mean(stacked_gradients, axis=0)

          if KL:

            normalized_grads = self.softmax(mean_gradient)

            mean_gradients.append(normalized_grads)

          else:

            mean_gradients.append(mean_gradient)

      return mean_gradients


    def compute_distributions(self, initial1, final1, initial2, final2, current_period, period):
        dist1 = self.compute_mean_gradients(initial1, final1, current_period, period)
        dist2 = self.compute_mean_gradients(initial2, final2, current_period, period)

        if dist1 is not None:
            self.previous_distributions_part1.append(dist1)

        if dist2 is not None:
            self.previous_distributions_part2.append(dist2)

        return dist1, dist2

    def compute_kl_divergence(self, current_distribution, previous_distribution):
        p = current_distribution / jnp.sum(current_distribution)
        q = previous_distribution / jnp.sum(previous_distribution)
        epsilon = 1e-10
        p = jnp.where(p == 0, epsilon, p)
        q = jnp.where(q == 0, epsilon, q)
        kl_div = jnp.sum(p * jnp.log(p / q))
        return kl_div

    def compute_kl_divergence_modified(self, current_distribution, previous_distribution):
        print("\n ---------- KL modified ----------")
        p = (current_distribution + 1) / jnp.sum(current_distribution + 1)
        q = (previous_distribution + 1) / jnp.sum(previous_distribution + 1)
        p_abs = jnp.abs(current_distribution) / jnp.sum(jnp.abs(current_distribution))
        epsilon = 1e-10
        p = jnp.where(p == 0, epsilon, p)
        q = jnp.where(q == 0, epsilon, q)
        kl_div = jnp.sum(p_abs * jnp.log(p / q))
        return kl_div


    def plot_distributions(self, distributions, title, current_period):
        sns.set_context("talk")

        sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        fig, axs = plt.subplots(4, 1, figsize=(25, 20))
        axs = axs.flatten()

        for layer_idx in range(4):
            indices = np.arange(len(distributions[0][layer_idx].flatten()))

            colors = ['red', 'blue', 'yellow']

            markers = ['s', '^', 'o']

            for period_idx, distribution in enumerate(distributions):

                dist_flat = distribution[layer_idx].flatten()

                if len(dist_flat) >= 4:
                    f_interp = interp1d(indices, dist_flat, kind='cubic')
                else:
                    f_interp = interp1d(indices, dist_flat, kind='linear')

                indices_new = np.linspace(indices.min(), indices.max(), num=500)
                dist_interp = f_interp(indices_new)

                axs[layer_idx].plot(indices_new, dist_interp, label=f'Period {period_idx + 1}', color=colors[period_idx], linewidth=2)
                axs[layer_idx].fill_between(indices_new, dist_interp, alpha=0.3, color=colors[period_idx])

                axs[layer_idx].plot(indices, dist_flat, color=colors[period_idx], marker=markers[period_idx], linestyle='None', markersize=8)

            axs[layer_idx].set_title(f'{title} Layer {layer_idx + 1}', fontsize=18)
            axs[layer_idx].set_xlabel("Weight Index", fontsize=14)
            axs[layer_idx].set_ylabel("Mean of Normalized Gradients", fontsize=14)
            axs[layer_idx].grid(True, alpha=0.7)
            axs[layer_idx].legend(fontsize=12)

        plt.tight_layout()
        plt.show()


    def visualize_gradients_distributions(self, initial1, final1, initial2, final2, period, current_period, metric='cosine'):
        blue_list = []
        red_list = []
        yellow_list = []
        dist1, dist2 = self.compute_distributions(initial1, final1, initial2, final2, current_period, period)

        if (current_period + 1) % 3 == 0 and current_period != 0:
            if len(self.previous_distributions_part1) >= 3 and len(self.previous_distributions_part2) >= 3:
                last_three_part1 = self.previous_distributions_part1[-3:]
                last_three_part2 = self.previous_distributions_part2[-3:]
                #self.plot_distributions(last_three_part1, "Part 1 Distributions", current_period)
                #self.plot_distributions(last_three_part2, "Part 2 Distributions", current_period)

                for layer_idx in range(4):
                    if metric == 'KL':
                        compute_divergence = self.compute_kl_divergence
                    elif metric == 'cosine':
                        compute_divergence = self.compute_cosine_similarity
                    elif metric == 'L2':
                        compute_divergence = self.compute_l2_distance
                    else:
                        raise ValueError("Invalid metric specified. Choose 'KL', 'cosine', or 'L2'.")

                    div_12_part1 = compute_divergence(last_three_part1[0][layer_idx], last_three_part1[1][layer_idx])
                    div_13_part1 = compute_divergence(last_three_part1[0][layer_idx], last_three_part1[2][layer_idx])
                    div_23_part1 = compute_divergence(last_three_part1[1][layer_idx], last_three_part1[2][layer_idx])

                    self.kl_divergences_part1[layer_idx]['12'].append(div_12_part1)
                    self.kl_divergences_part1[layer_idx]['13'].append(div_13_part1)
                    self.kl_divergences_part1[layer_idx]['23'].append(div_23_part1)

                    div_12_part2 = compute_divergence(last_three_part2[0][layer_idx], last_three_part2[1][layer_idx])
                    div_13_part2 = compute_divergence(last_three_part2[0][layer_idx], last_three_part2[2][layer_idx])
                    div_23_part2 = compute_divergence(last_three_part2[1][layer_idx], last_three_part2[2][layer_idx])

                    self.kl_divergences_part2[layer_idx]['12'].append(div_12_part2)
                    self.kl_divergences_part2[layer_idx]['13'].append(div_13_part2)
                    self.kl_divergences_part2[layer_idx]['23'].append(div_23_part2)

                    # disjoint divergence: color injection  - previous correction
                    #print("\n blue term corr:", last_three_part2[0][layer_idx]/jnp.linalg.norm(jnp.squeeze(last_three_part2[0][layer_idx], 0)))
                    #print("\n blue term inc:", last_three_part1[0][layer_idx]/jnp.linalg.norm(jnp.squeeze(last_three_part1[0][layer_idx], 0)))
                    # # blue = last_three_part2[0][layer_idx]/jnp.linalg.norm(jnp.squeeze(last_three_part2[0][layer_idx], 0)) - last_three_part1[1][layer_idx]/jnp.linalg.norm(jnp.squeeze(last_three_part1[1][layer_idx], 0)) # first normalize
                    # # yellow = last_three_part2[1][layer_idx]/jnp.linalg.norm(jnp.squeeze(last_three_part2[1][layer_idx], 0)) - last_three_part1[2][layer_idx]/jnp.linalg.norm(jnp.squeeze(last_three_part1[2][layer_idx], 0))
                    # # red = last_three_part2[2][layer_idx]/jnp.linalg.norm(jnp.squeeze(last_three_part2[2][layer_idx], 0)) - last_three_part1[0][layer_idx]/jnp.linalg.norm(jnp.squeeze(last_three_part1[0][layer_idx], 0))
                    # disjoint: color - its correction
                    # red = last_three_part2[0][layer_idx]/jnp.linalg.norm(jnp.squeeze(last_three_part2[0][layer_idx], 0)) + last_three_part1[0][layer_idx]/jnp.linalg.norm(jnp.squeeze(last_three_part1[0][layer_idx], 0)) # first normalize
                    # blue = last_three_part2[1][layer_idx]/jnp.linalg.norm(jnp.squeeze(last_three_part2[1][layer_idx], 0)) + last_three_part1[1][layer_idx]/jnp.linalg.norm(jnp.squeeze(last_three_part1[1][layer_idx], 0))
                    # yellow = last_three_part2[2][layer_idx]/jnp.linalg.norm(jnp.squeeze(last_three_part2[2][layer_idx], 0)) + last_three_part1[2][layer_idx]/jnp.linalg.norm(jnp.squeeze(last_three_part1[2][layer_idx], 0))


    def compute_shannon_entropy(self, distribution):
        p = jnp.array(distribution).flatten()
        p = p / p.sum()  # normalize to make it a probability distribution
        entropy = -jnp.sum(p * jnp.log(p + 1e-10))
        return entropy

    def plot_shannon_entropies(self):
        sns.set_context("talk")
        sns.set_style("white")
        palette = ['#FF4500', '#1E90FF', '#FFD700']

        fig, axs = plt.subplots(2, 2, figsize=(20, 10))
        axs = axs.flatten()

        for layer_idx in range(4):
            for period, values in self.entropies_part1[layer_idx].items():
                axs[layer_idx].plot(values, label=f'Part 1 - Period {period}', marker='o', markersize=10, alpha=1.0, color=palette[int(period)-1], linewidth=2.5)

            axs[layer_idx].set_title(f'Shannon Entropy for Layer {layer_idx + 1} (Part 1)', fontsize=18)
            axs[layer_idx].set_xlabel('Period', fontsize=14)
            axs[layer_idx].set_ylabel('Shannon Entropy', fontsize=14)
            axs[layer_idx].legend(fontsize=10)

        plt.tight_layout()
        plt.show()

        fig, axs = plt.subplots(2, 2, figsize=(20, 10))
        axs = axs.flatten()

        for layer_idx in range(4):
            for period, values in self.entropies_part2[layer_idx].items():
                axs[layer_idx].plot(values, label=f'Part 2 - Period {period}', marker='o', markersize=10, alpha=1.0, color=palette[int(period)-1], linewidth=2.5)

            axs[layer_idx].set_title(f'Shannon Entropy for Layer {layer_idx + 1} (Part 2)', fontsize=18)
            axs[layer_idx].set_xlabel('Period', fontsize=14)
            axs[layer_idx].set_ylabel('Shannon Entropy', fontsize=14)
            axs[layer_idx].legend(fontsize=10)

        plt.tight_layout()
        plt.show()


    def compute_cosine_similarity(self, dist1, dist2):
        dist1 = jnp.squeeze(dist1, 0)
        dist2 = jnp.squeeze(dist2, 0)
        dot_product = jnp.dot(dist1, dist2)
        norm_a = jnp.linalg.norm(dist1)
        norm_b = jnp.linalg.norm(dist2)
        return dot_product / (norm_a * norm_b)

    def compute_l2_distance(self, dist1, dist2):
        dist1 = jnp.array(dist1)
        dist2 = jnp.array(dist2)
        return jnp.linalg.norm(dist1 - dist2)


    def plot_kl_divergences(self):
        sns.set_context("talk")
        sns.set_style("white")
        palette = ['#000000', '#555555', '#AAAAAA']

        fig, axs = plt.subplots(2, 2, figsize=(20, 10))
        axs = axs.flatten()

        color_mapping = {'12': 0, '13': 1, '23': 2}

        for layer_idx in range(4):
            for comparison, values in self.kl_divergences_part1[layer_idx].items():
                color_index = color_mapping[comparison]
                axs[layer_idx].plot(values, label=f'Part 1 - Comparison classes {comparison}', marker='o', markersize=8, alpha=1.0, color=palette[color_index], linewidth = "2.5")

            axs[layer_idx].set_title(f'Layer {layer_idx + 1} (Part 1)', fontsize=18)
            axs[layer_idx].set_xlabel('Period', fontsize=14)
            axs[layer_idx].grid(False)
            axs[layer_idx].legend(fontsize=10)

        plt.tight_layout()
        plt.show()

        fig, axs = plt.subplots(2, 2, figsize=(20, 10))
        axs = axs.flatten()

        for layer_idx in range(4):
            for comparison, values in self.kl_divergences_part2[layer_idx].items():
                color_index = color_mapping[comparison]
                axs[layer_idx].plot(values, label=f'Part 2 - Comparison classes {comparison}', marker='o', markersize=8, alpha=1.0, color=palette[color_index], linewidth = "2.5")

            axs[layer_idx].set_title(f'Layer {layer_idx + 1} (Part 2)', fontsize=18)
            axs[layer_idx].set_xlabel('Period', fontsize=14)
            axs[layer_idx].grid(False)
            axs[layer_idx].legend(fontsize=10)

        plt.tight_layout()
        plt.show()

        fig, axs = plt.subplots(2, 2, figsize=(20, 10))
        axs = axs.flatten()

    '''
    def disjoint_distributions(self, part1, part2, current_period):
        if not hasattr(self, 'disjoint_plot_initialized'):
            self.disjoint_plot_initialized = True
            self.fig, self.axs = plt.subplots(4, 1, figsize=(25, 20))
            self.axs = self.axs.flatten()

        cmap = plt.get_cmap('Reds')
        num_colors = 10  # Define how many colors you want to use
        color_range = np.linspace(0.3, 1.0, num_colors)

        def plot_distribution(distribution, layer_idx, linestyle, color):
            indices = np.arange(len(distribution.flatten()))
            if len(distribution.flatten()) >= 4:
                f_interp = interp1d(indices, distribution.flatten(), kind='cubic')
            else:
                f_interp = interp1d(indices, distribution.flatten(), kind='linear')
            indices_new = np.linspace(indices.min(), indices.max(), num=500)
            dist_interp = f_interp(indices_new)

            self.axs[layer_idx].plot(indices_new, dist_interp, linestyle=linestyle, color=color, linewidth=2)
            self.axs[layer_idx].fill_between(indices_new, dist_interp, alpha=0.3, color=color)

        for layer_idx in range(4):
            part1_color = cmap(color_range[current_period % num_colors])
            part2_color = cmap(color_range[current_period % num_colors])

            plot_distribution(part1[layer_idx], layer_idx, '-', part1_color)  # Continuous line for part1
            plot_distribution(part2[layer_idx], layer_idx, '--', part2_color)  # Dotted line for part2

            self.axs[layer_idx].set_title(f'Part 1 and Part 2 Distributions Layer {layer_idx + 1}', fontsize=18)
            self.axs[layer_idx].set_xlabel("Weight Index", fontsize=14)
            self.axs[layer_idx].set_ylabel("Mean of Normalized Gradients", fontsize=14)
            self.axs[layer_idx].grid(True, alpha=0.7)

        plt.tight_layout()
        plt.show()'''


    def disjoint_distributions(self, part1_distributions, part2_distributions, current_period, class_color, output_dir, filenames):
        sns.set_style("white")
        fig, axs = plt.subplots(4, 1, figsize=(25, 20))
        axs = axs.flatten()

        cmap = plt.get_cmap(class_color)
        num_phases = len(part1_distributions)
        color_range = np.linspace(0.3, 1.0, num_phases)

        marker = {'Reds': 's', 'Blues': '^', 'YlOrBr': 'o'}
        current_marker = marker[class_color]

        def plot_distribution(distribution, layer_idx, linestyle, color, facecolor='auto', alpha=1.0):
            indices = np.arange(len(distribution.flatten()))
            dist_flat = distribution.flatten()
            if len(dist_flat) >= 4:
                f_interp = interp1d(indices, dist_flat, kind='cubic')
            else:
                f_interp = interp1d(indices, dist_flat, kind='linear')
            indices_new = np.linspace(indices.min(), indices.max(), num=500)
            dist_interp = f_interp(indices_new)

            axs[layer_idx].plot(indices_new, dist_interp, linestyle=linestyle, color=color, linewidth=2, alpha=alpha)

            if facecolor == 'none':
                axs[layer_idx].scatter(indices, dist_flat, edgecolor=color, marker=current_marker, facecolor='none', s=50, zorder=5, alpha=alpha)
            else:
                axs[layer_idx].scatter(indices, dist_flat, color=color, marker=current_marker, s=50, zorder=5, alpha=alpha)
                axs[layer_idx].fill_between(indices_new, dist_interp, alpha=alpha * 0.3, color=color)

        for i in range(num_phases):
            alpha_value = (i +1)/ (num_phases+1)
            for layer_idx in range(4):
                color = cmap(color_range[i])
                plot_distribution(part1_distributions[i][layer_idx], layer_idx, '-', color, facecolor=color, alpha=alpha_value)  # Continuous line for part1
                plot_distribution(part2_distributions[i][layer_idx], layer_idx, '--', color, facecolor='none', alpha=alpha_value)  # Dotted line for part2

        for layer_idx in range(4):
            axs[layer_idx].set_title(f'Part 1 and Part 2 Gradient Distributions Layer {layer_idx + 1}', fontsize=18)
            axs[layer_idx].set_xlabel("Weight Index", fontsize=14)
            axs[layer_idx].set_ylabel("Gradient Distribution", fontsize=14)
            axs[layer_idx].grid(True, alpha=0.7)
            axs[layer_idx].set_yscale('symlog', linthresh=1e-6, linscale=1)  # Set y-axis to log scale

        custom_lines = [
            Line2D([0], [0], color='k', lw=2, linestyle='-', label='Large central step'),
            Line2D([0], [0], color='k', lw=2, linestyle='--', label='Space between oscillations'),
            Patch(facecolor=cmap(0.3), edgecolor='k', label='Initial steps', alpha=0.3),
            Patch(facecolor=cmap(1.0), edgecolor='k', label='Final part', alpha=0.9)
        ]
        axs[0].legend(handles=custom_lines, loc='upper right', fontsize=12)

        for ax in axs:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=num_phases))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.02, pad=0.01)

        plt.tight_layout()

        filename = output_dir / f"{class_color}_distributions_period_{current_period}.png"
        plt.savefig(filename)
        filenames.append(filename)
        plt.show()


    def disjoint_distributions_all(self, data, red_part1_distributions, red_part2_distributions, blue_part1_distributions, blue_part2_distributions, yellow_part1_distributions, yellow_part2_distributions, current_period, output_dir, filenames, grads_calcs):
        sns.set_style("white")
        fig, axs = plt.subplots(4, 1, figsize=(25, 20))
        axs = axs.flatten()

        cmap_red = plt.get_cmap('Reds')
        cmap_blue = plt.get_cmap('Blues')
        cmap_yellow = plt.get_cmap('YlOrBr')

        num_phases = len(red_part1_distributions)
        color_range = np.linspace(0.3, 1.0, num_phases)

        def plot_distribution(distribution, layer_idx, linestyle, color, current_marker, facecolor='auto', alpha=1.0):
            indices = np.arange(len(distribution.flatten()))
            dist_flat = distribution.flatten()
            if len(dist_flat) >= 4:
                f_interp = interp1d(indices, dist_flat, kind='cubic')
            else:
                f_interp = interp1d(indices, dist_flat, kind='linear')
            indices_new = np.linspace(indices.min(), indices.max(), num=500)
            dist_interp = f_interp(indices_new)

            axs[layer_idx].plot(indices_new, dist_interp, linestyle=linestyle, color=color, linewidth=2, alpha=alpha)

            if facecolor == 'none':
                axs[layer_idx].scatter(indices, dist_flat, edgecolor=color, marker=current_marker, facecolor='none', s=50, zorder=5, alpha=alpha)
            else:
                axs[layer_idx].scatter(indices, dist_flat, color=color, marker=current_marker, s=50, zorder=5, alpha=alpha)

        for i in range(num_phases):
            alpha_value = (i + 1) / (num_phases + 1)
            for layer_idx in range(4):
                color_red = cmap_red(color_range[i])
                color_blue = cmap_blue(color_range[i])
                color_yellow = cmap_yellow(color_range[i])

                # Plot part 1 distributions (solid lines)
                plot_distribution(red_part1_distributions[i][layer_idx], layer_idx, '-', color_red, current_marker='s', facecolor=color_red, alpha=alpha_value)
                plot_distribution(blue_part1_distributions[i][layer_idx], layer_idx, '-', color_blue, current_marker='^', facecolor=color_blue, alpha=alpha_value)
                plot_distribution(yellow_part1_distributions[i][layer_idx], layer_idx, '-', color_yellow, current_marker='o', facecolor=color_yellow, alpha=alpha_value)

                # Plot part 2 distributions (dashed lines)
                plot_distribution(red_part2_distributions[i][layer_idx], layer_idx, '--', color_red, current_marker='s', facecolor='none', alpha=alpha_value)
                plot_distribution(blue_part2_distributions[i][layer_idx], layer_idx, '--', color_blue, current_marker='^', facecolor='none', alpha=alpha_value)
                plot_distribution(yellow_part2_distributions[i][layer_idx], layer_idx, '--', color_yellow, current_marker='o', facecolor='none', alpha=alpha_value)

        for layer_idx in range(4):
            axs[layer_idx].set_title(f'Red, Blue, Yellow {data} Distributions Layer {layer_idx + 1}', fontsize=18)
            axs[layer_idx].set_xlabel("Weight Index", fontsize=14)
            axs[layer_idx].set_ylabel(f"{data} Distribution", fontsize=14)
            axs[layer_idx].grid(True, alpha=0.7)
            if grads_calcs:
              axs[layer_idx].set_yscale('symlog', linthresh=1e-6, linscale=1)  # Set y-axis to log scale

        custom_lines = [
            Line2D([0], [0], color='red', lw=2, linestyle='-', label='Red part1 distribution'),
            Line2D([0], [0], color='red', lw=2, linestyle='--', label='Red part2 distribution'),
            Line2D([0], [0], color='blue', lw=2, linestyle='-', label='Blue part1 distribution'),
            Line2D([0], [0], color='blue', lw=2, linestyle='--', label='Blue part2 distribution'),
            Line2D([0], [0], color='orange', lw=2, linestyle='-', label='Yellow part1 distribution'),
            Line2D([0], [0], color='orange', lw=2, linestyle='--', label='Yellow part2 distribution')
        ]
        axs[0].legend(handles=custom_lines, loc='upper right', fontsize=12)

        plt.tight_layout()
        filename = output_dir / f"distributions_period_{current_period}_{self.label}.png"
        plt.savefig(filename)
        filenames.append(filename)


    def disjoint_distributions_all_split(self, data, red_part1_distributions, red_part2_distributions, blue_part1_distributions, blue_part2_distributions, yellow_part1_distributions, yellow_part2_distributions, current_period, output_dir, filenames, grads_calcs):
        sns.set_style("white")

        cmap_black = plt.get_cmap('Greys')
        cmap_red = plt.get_cmap('Reds')
        cmap_blue = plt.get_cmap('Blues')
        cmap_yellow = plt.get_cmap('YlOrBr')

        num_phases = len(red_part1_distributions)
        color_range = np.linspace(0.3, 1.0, num_phases)

        def plot_distribution(distribution, ax, linestyle, color, current_marker, facecolor='auto', alpha=1.0):
            indices = np.arange(len(distribution.flatten()))
            dist_flat = distribution.flatten()
            if len(dist_flat) >= 4:
                f_interp = interp1d(indices, dist_flat, kind='cubic')
            else:
                f_interp = interp1d(indices, dist_flat, kind='linear')
            indices_new = np.linspace(indices.min(), indices.max(), num=500)
            dist_interp = f_interp(indices_new)

            ax.plot(indices_new, dist_interp, linestyle=linestyle, color=color, linewidth=2, alpha=alpha)

            if facecolor == 'none':
                ax.scatter(indices, dist_flat, edgecolor=color, marker=current_marker, facecolor='none', s=50, zorder=5, alpha=alpha)
            else:
                ax.scatter(indices, dist_flat, color=color, marker=current_marker, s=50, zorder=5, alpha=alpha)
                ax.fill_between(indices_new, dist_interp, alpha=alpha * 0.3, color=color)

        for layer_idx in range(4):
            fig = plt.figure(figsize=(30, 15))
            gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.07)

            axs = [fig.add_subplot(gs[i]) for i in range(3)]

            for i in range(num_phases):
                alpha_value = (i + 1) / (num_phases + 1)
                color_red = cmap_red(color_range[i])
                color_blue = cmap_blue(color_range[i])
                color_yellow = cmap_yellow(color_range[i])

                # row 0
                plot_distribution(red_part1_distributions[i][layer_idx], axs[0], '-', color_red, current_marker='s', facecolor=color_red, alpha=alpha_value)
                plot_distribution(red_part2_distributions[i][layer_idx], axs[0], '--', color_red, current_marker='s', facecolor='none', alpha=alpha_value)

                # row 1
                plot_distribution(blue_part1_distributions[i][layer_idx], axs[1], '-', color_blue, current_marker='^', facecolor=color_blue, alpha=alpha_value)
                plot_distribution(blue_part2_distributions[i][layer_idx], axs[1], '--', color_blue, current_marker='^', facecolor='none', alpha=alpha_value)

                # row 2
                plot_distribution(yellow_part1_distributions[i][layer_idx], axs[2], '-', color_yellow, current_marker='o', facecolor=color_yellow, alpha=alpha_value)
                plot_distribution(yellow_part2_distributions[i][layer_idx], axs[2], '--', color_yellow, current_marker='o', facecolor='none', alpha=alpha_value)

            axs[0].set_ylabel(f"{data} Distribution", fontsize=14)
            axs[1].set_ylabel(f"{data} Distribution", fontsize=14)
            axs[2].set_ylabel(f"{data} Distribution", fontsize=14)

            axs[2].set_xlabel("Weight Index", fontsize=14)

            axs[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            axs[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

            for ax in axs:
                ax.grid(True, alpha=0.7)
                if grads_calcs:
                  ax.set_yscale('symlog', linthresh=1e-6, linscale=1)
                ax.xaxis.set_major_locator(MultipleLocator(5))

            fig.suptitle(f'{data} Distributions for Layer {layer_idx + 1}', fontsize=18)

            sm_red = plt.cm.ScalarMappable(cmap=cmap_red, norm=plt.Normalize(vmin=0, vmax=num_phases))
            sm_red.set_array([])
            cbar_red = plt.colorbar(sm_red, ax=axs[0], orientation='vertical', fraction=0.02, pad=0.01)

            sm_blue = plt.cm.ScalarMappable(cmap=cmap_blue, norm=plt.Normalize(vmin=0, vmax=num_phases))
            sm_blue.set_array([])
            cbar_blue = plt.colorbar(sm_blue, ax=axs[1], orientation='vertical', fraction=0.02, pad=0.01)

            sm_yellow = plt.cm.ScalarMappable(cmap=cmap_yellow, norm=plt.Normalize(vmin=0, vmax=num_phases))
            sm_yellow.set_array([])
            cbar_yellow = plt.colorbar(sm_yellow, ax=axs[2], orientation='vertical', fraction=0.02, pad=0.01)

            custom_lines = [
                Line2D([0], [0], color='k', lw=2, linestyle='-', label='Space between oscillations Initial'),
                Line2D([0], [0], color='k', lw=2, linestyle='--', label='Space between oscillations Final'),
                Patch(facecolor=cmap_black(0.3), edgecolor='k', label='Initial steps', alpha=0.3),
                Patch(facecolor=cmap_black(0.8), edgecolor='k', label='Final part', alpha=0.9)
            ]
            axs[0].legend(handles=custom_lines, loc='upper right', fontsize=12)

            plt.tight_layout()
            filename = output_dir / f"distributions_period_{current_period}_layer{layer_idx}_{self.label}.png"
            plt.savefig(filename)
            filenames.append(filename)


    def train(self, X_train, Y_train, T, w_max, total_steps, n_steps_per_epoch, batch_size, initial1, final1, initial2, final2, period, grads_calcs):
        rng = random.PRNGKey(0)
        rng, key = random.split(rng)
        self.X_train = X_train
        self.Y_train = Y_train
        losses = []
        accuracies = []
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

        # disjoint:
        output_dir_distributions = Path("disjoint_distributions")
        output_dir_distributions.mkdir(exist_ok=True)
        filenames_distributions = []

        output_dir_distributions_all = Path("disjoint_distributions_all")
        output_dir_distributions_all.mkdir(exist_ok=True)
        filenames_distributions_all = []

        output_dir_distributions_all_split = Path("disjoint_distributions_all_split")
        output_dir_distributions_all_split.mkdir(exist_ok=True)
        filenames_distributions_all_split = []

        # #initial1, final1 = 0, 1500 #2100, 3000 #w = 10
        # initial1, final1 = 2300, 3000 #2100, 3000 #w = 10
        # initial2, final2 = 3500, 5000 #3500, 5000
        # period = 5000
        # initial1, final1 = 175, 325 #2100, 3000 #w = 10
        # initial2, final2 = 400, 500 #3500, 5000
        # period = 500
        #initial1, final1 = 23, 30
        #initial2, final2 = 35, 50
        #period = 50

        red_part1_distributions = []
        red_part2_distributions = []
        blue_part1_distributions = []
        blue_part2_distributions = []
        yellow_part1_distributions = []
        yellow_part2_distributions = []
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
                    if self.param_shapes is None:
                      _, self.param_shapes = self.flatten_params(params, return_shapes=True)
                    self.save_gradients_blocks(grads) if grads_calcs else self.save_gradients_blocks(params)

                    if t % period == 0 and t!=0:
                      focus = ((t // period) - 1) % self.num_classes
                      print("\n class_focus", focus)
                      current_period = t // period - 1
                      print("current_period", current_period)

                      self.visualize_gradients_distributions(initial1, final1, initial2, final2, period, current_period)
                      part1 = self.previous_distributions_part1[-1]  # latest distribution for part1
                      part2 = self.previous_distributions_part2[-1]  # latest distribution for part2

                      if focus == 0: # red class
                           red_part1_distributions.append(part1)
                           red_part2_distributions.append(part2)

                      elif focus == 1:  # blue class
                          blue_part1_distributions.append(part1)
                          blue_part2_distributions.append(part2)

                      elif focus == 2:  # yellow class
                          yellow_part1_distributions.append(part1)
                          yellow_part2_distributions.append(part2)

                      if (current_period+1) % 3 == 0:
                       print("\nKL Divergences")
                       self.plot_kl_divergences()

                       name = "Gradient" if grads_calcs else "Weight"

                       if grads_calcs:
                        self.disjoint_distributions_all_split(name, red_part1_distributions, red_part2_distributions, blue_part1_distributions, blue_part2_distributions, yellow_part1_distributions, yellow_part2_distributions, current_period, output_dir_distributions_all_split, filenames_distributions_all_split, grads_calcs)
                       else:
                        self.disjoint_distributions_all(name, red_part1_distributions, red_part2_distributions, blue_part1_distributions, blue_part2_distributions, yellow_part1_distributions, yellow_part2_distributions, current_period, output_dir_distributions_all, filenames_distributions_all, grads_calcs)

                    if self.track_periodic_weight_diff and (t % self.weight_diff_step_interval == 0):

                        dists = self.compare_weights_layer_all(t, params)

                        dist_init_combined.append(dists['dist_init_combined'])
                        dist_last_period_combined.append(dists["dist_last_period_combined"])
                        dist_init_class_spec.append(dists["dist_init_class_spec"])
                        dist_last_period_class_spec.append(dists["dist_last_period_class_spec"])

                        dist_init.append(dists["dist_init"])
                        dist_last_period.append(dists["dist_last_period"])

                    if step == n_steps_per_epoch - 1:
                        batch_loss = self.loss_fn(params, X_train, Y_train) # total set
                        losses.append(batch_loss)
                        epoch_accuracy = self.accuracy(params, X_train, Y_train)
                        accuracies.append(epoch_accuracy)

                    pbar.update(1)

        self.last_params = params


        with imageio.get_writer(f'visuals_{self.label}.mp4', fps=2) as writer:
            for filename_visuals in filenames_visuals:
                image = imageio.imread(filename_visuals)
                writer.append_data(image)

        for filename_visuals in filenames_visuals:
            filename_visuals.unlink()

        with imageio.get_writer(f'cosine_similarity_evolution_{self.label}.mp4', fps=2) as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        for filename in filenames:
            filename.unlink()

        with imageio.get_writer(f'hist_frames_{self.label}.mp4', fps=2) as writer:
            for filename_hist in filenames_hist:
                image = imageio.imread(filename_hist)
                writer.append_data(image)

        for filename_hist in filenames_hist:
            filename_hist.unlink()

        with imageio.get_writer(f'fisher_matrix{self.label}.mp4', fps=2) as writer:
            for filename_fisher in filenames_fisher:
                image = imageio.imread(filename_fisher)
                writer.append_data(image)

        for filename_fisher in filenames_fisher:
            filename_fisher.unlink()

        with imageio.get_writer(f'weights_corr{self.label}.mp4', fps=2) as writer:
            for filename_weights_corr in filenames_weights_corr:
                image = imageio.imread(filename_weights_corr)
                writer.append_data(image)

        for filename_weights_corr in filenames_weights_corr:
            filename_weights_corr.unlink()

        description = f"width_{self.nn_width}-batchsize_{batch_size}-wmax_{w_max}-period_{T}-Nperiods_{int(total_steps/T)}"

        plt.figure(figsize=(25, 5))
        plt.plot(losses, label='Training Loss',)
        plt.xlabel('step')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(False)
        plt.show()

        with open(f'losses_{description}.pkl', 'wb') as f:
            pickle.dump(losses, f)

        plt.figure(figsize=(25, 5))
        plt.plot(accuracies, label='Training Accuracy',)
        plt.xlabel('step')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.grid(False)
        plt.show()

        with open(f'accuracies_{description}.pkl', 'wb') as f:
            pickle.dump(accuracies, f)

        plt.figure(figsize=(25, 5))
        plt.plot(dist_init_combined, label='dist_init_combined',)
        plt.xlabel('step')
        plt.legend()
        plt.grid(False)
        plt.show()

        with open(f'dist_init_combined_{description}.pkl', 'wb') as f:
            pickle.dump(dist_init_combined, f)

        plt.figure(figsize=(25, 5))
        plt.plot(dist_last_period_combined, label='dist_last_period_combined',)
        plt.xlabel('step')
        plt.legend()
        plt.grid(False)
        plt.show()

        with open(f'dist_last_period_combined_{description}.pkl', 'wb') as f:
            pickle.dump(dist_last_period_combined, f)

        plt.figure(figsize=(25, 5))
        plt.plot(dist_init_class_spec, label='dist_init_class_spec',)
        plt.xlabel('step')
        plt.legend()
        plt.grid(False)
        plt.show()

        with open(f'dist_init_class_spec_{description}.pkl', 'wb') as f:
            pickle.dump(dist_init_class_spec, f)

        plt.figure(figsize=(25, 5))
        plt.plot(dist_last_period_class_spec, label='dist_last_period_class_spec',)
        plt.xlabel('step')
        plt.legend()
        plt.grid(False)
        plt.show()

        with open(f'dist_last_period_class_spec_{description}.pkl', 'wb') as f:
            pickle.dump(dist_last_period_class_spec, f)

        plt.figure(figsize=(25, 5))
        plt.plot(dist_init, label='dist_init',)
        plt.xlabel('step')
        plt.legend()
        plt.grid(False)
        plt.show()

        with open(f'dist_init_{description}.pkl', 'wb') as f:
            pickle.dump(dist_init, f)

        plt.figure(figsize=(25, 5))
        plt.plot(dist_last_period, label='dist_last_period',)
        plt.xlabel('step')
        plt.legend()
        plt.grid(False)
        plt.show()

        with open(f'dist_last_period_{description}.pkl', 'wb') as f:
            pickle.dump(dist_last_period, f)

        return losses, accuracies

"""# Gradients

## inicio y escalon
"""

initial1, final1 = 0, 1500
initial2, final2 = 2300, 3000
period = 5000
grads_calcs = True

classifier = SpiralClassifier(points_per_class=100, num_classes=3, nn_width=50, learning_rate=0.002, real_time_visualization=True, vis_step_interval=100, track_periodic_weight_diff=True, T=5000, track_weight_diff=True, weight_diff_step_interval = 1, l2_reg = 0., label = "grads_inicial_escalon", show_markers = True)#0.5*1e-3)
X_train, Y_train = classifier.make_dataset(seed = 0)

T = 5000
total_steps =  25 * 5000 + 1
n_steps_per_epoch = 1
batch_size = 50
classifier.train(X_train, Y_train, T =T, w_max = 70, total_steps = total_steps, n_steps_per_epoch = n_steps_per_epoch, batch_size= batch_size, initial1=initial1, final1=final1, initial2=initial2, final2=final2, period=period, grads_calcs=grads_calcs)

test_data_generator = SpiralClassifier(points_per_class=100, num_classes=3)
X_test, Y_test = test_data_generator.make_dataset(seed = 1)
test_accuracy = classifier.accuracy(classifier.last_params, X_test, Y_test)
print(f"Test Accuracy: {test_accuracy * 100:.4f}%")

"""## escalon y final"""

initial1, final1 = 2300, 3000
initial2, final2 = 3500, 5000
period = 5000

grads_calcs = True

classifier = SpiralClassifier(points_per_class=100, num_classes=3, nn_width=50, learning_rate=0.002, real_time_visualization=True, vis_step_interval=100, track_periodic_weight_diff=True, T=5000, track_weight_diff=True, weight_diff_step_interval = 1, l2_reg = 0., label = "grads_escalon_final", show_markers = True)#0.5*1e-3)
X_train, Y_train = classifier.make_dataset(seed = 0)
T = 5000
total_steps =  25 * 5000 + 1
n_steps_per_epoch = 1
batch_size = 50
classifier.train(X_train, Y_train, T =T, w_max = 70, total_steps = total_steps, n_steps_per_epoch = n_steps_per_epoch, batch_size= batch_size, initial1=initial1, final1=final1, initial2=initial2, final2=final2, period=period, grads_calcs=grads_calcs)

# test accuracy
test_data_generator = SpiralClassifier(points_per_class=100, num_classes=3)
X_test, Y_test = test_data_generator.make_dataset(seed = 1)
test_accuracy = classifier.accuracy(classifier.last_params, X_test, Y_test)
print(f"Test Accuracy: {test_accuracy * 100:.4f}%")

"""## inicio y final"""

initial1, final1 = 0, 1500
initial2, final2 = 3500, 5000
period = 5000

grads_calcs = True

classifier = SpiralClassifier(points_per_class=100, num_classes=3, nn_width=50, learning_rate=0.002, real_time_visualization=True, vis_step_interval=100, track_periodic_weight_diff=True, T=5000, track_weight_diff=True, weight_diff_step_interval = 1, l2_reg = 0., label = "grads_inicio_final", show_markers = True)#0.5*1e-3)
X_train, Y_train = classifier.make_dataset(seed = 0)
T = 5000
total_steps =  25 * 5000 + 1
n_steps_per_epoch = 1
batch_size = 50
classifier.train(X_train, Y_train, T =T, w_max = 70, total_steps = total_steps, n_steps_per_epoch = n_steps_per_epoch, batch_size= batch_size, initial1=initial1, final1=final1, initial2=initial2, final2=final2, period=period, grads_calcs=grads_calcs)

# test accuracy
test_data_generator = SpiralClassifier(points_per_class=100, num_classes=3)
X_test, Y_test = test_data_generator.make_dataset(seed = 1)
test_accuracy = classifier.accuracy(classifier.last_params, X_test, Y_test)
print(f"Test Accuracy: {test_accuracy * 100:.4f}%")

"""# Weights

## inicio y escalon
"""

initial1, final1 = 0, 1500
initial2, final2 = 2300, 3000
period = 5000
grads_calcs = False

classifier = SpiralClassifier(points_per_class=100, num_classes=3, nn_width=50, learning_rate=0.002, real_time_visualization=True, vis_step_interval=100, track_periodic_weight_diff=True, T=5000, track_weight_diff=True, weight_diff_step_interval = 1, l2_reg = 0., label = "weights_inicial_escalon", show_markers = True)#0.5*1e-3)
X_train, Y_train = classifier.make_dataset(seed = 0)

T = 5000
total_steps =  25 * 5000 + 1
n_steps_per_epoch = 1
batch_size = 50
classifier.train(X_train, Y_train, T =T, w_max = 70, total_steps = total_steps, n_steps_per_epoch = n_steps_per_epoch, batch_size= batch_size, initial1=initial1, final1=final1, initial2=initial2, final2=final2, period=period, grads_calcs=grads_calcs)

# test accuracy
test_data_generator = SpiralClassifier(points_per_class=100, num_classes=3)
X_test, Y_test = test_data_generator.make_dataset(seed = 1)
test_accuracy = classifier.accuracy(classifier.last_params, X_test, Y_test)
print(f"Test Accuracy: {test_accuracy * 100:.4f}%")

"""## escalon y final"""

initial1, final1 = 2300, 3000
initial2, final2 = 3500, 5000
period = 5000

grads_calcs = False

classifier = SpiralClassifier(points_per_class=100, num_classes=3, nn_width=50, learning_rate=0.002, real_time_visualization=True, vis_step_interval=100, track_periodic_weight_diff=True, T=5000, track_weight_diff=True, weight_diff_step_interval = 1, l2_reg = 0., label = "weights_escalon_final", show_markers = True)#0.5*1e-3)
X_train, Y_train = classifier.make_dataset(seed = 0)
T = 5000
total_steps =  25 * 5000 + 1
n_steps_per_epoch = 1
batch_size = 50
classifier.train(X_train, Y_train, T =T, w_max = 70, total_steps = total_steps, n_steps_per_epoch = n_steps_per_epoch, batch_size= batch_size, initial1=initial1, final1=final1, initial2=initial2, final2=final2, period=period, grads_calcs=grads_calcs)

# test accuracy
test_data_generator = SpiralClassifier(points_per_class=100, num_classes=3)
X_test, Y_test = test_data_generator.make_dataset(seed = 1)
test_accuracy = classifier.accuracy(classifier.last_params, X_test, Y_test)
print(f"Test Accuracy: {test_accuracy * 100:.4f}%")

"""## inicio y final"""

initial1, final1 = 0, 1500
initial2, final2 = 3500, 5000
period = 5000

grads_calcs = False

classifier = SpiralClassifier(points_per_class=100, num_classes=3, nn_width=50, learning_rate=0.002, real_time_visualization=True, vis_step_interval=100, track_periodic_weight_diff=True, T=5000, track_weight_diff=True, weight_diff_step_interval = 1, l2_reg = 0., label = "weights_inicio_final", show_markers = True)#0.5*1e-3)
X_train, Y_train = classifier.make_dataset(seed = 0)
T = 5000
total_steps =  25 * 5000 + 1
n_steps_per_epoch = 1
batch_size = 50
classifier.train(X_train, Y_train, T =T, w_max = 70, total_steps = total_steps, n_steps_per_epoch = n_steps_per_epoch, batch_size= batch_size, initial1=initial1, final1=final1, initial2=initial2, final2=final2, period=period, grads_calcs=grads_calcs)

# test accuracy
test_data_generator = SpiralClassifier(points_per_class=100, num_classes=3)
X_test, Y_test = test_data_generator.make_dataset(seed = 1)
test_accuracy = classifier.accuracy(classifier.last_params, X_test, Y_test)
print(f"Test Accuracy: {test_accuracy * 100:.4f}%")

"""# No osc

## Gradients
"""

initial1, final1 = 2300, 3000
initial2, final2 = 3500, 5000
period = 5000

grads_calcs = True

classifier = SpiralClassifier(points_per_class=100, num_classes=3, nn_width=50, learning_rate=0.005, real_time_visualization=True, vis_step_interval=100, track_periodic_weight_diff=True, T=5000, track_weight_diff=True, weight_diff_step_interval = 1, l2_reg = 0., label = "grads_inicio_final", show_markers = True)#0.5*1e-3)
X_train, Y_train = classifier.make_dataset(seed = 0)
T = 5000
total_steps =  25 * 5000 + 1
n_steps_per_epoch = 1
batch_size = 50
classifier.train(X_train, Y_train, T =T, w_max = 1, total_steps = total_steps, n_steps_per_epoch = n_steps_per_epoch, batch_size= batch_size, initial1=initial1, final1=final1, initial2=initial2, final2=final2, period=period, grads_calcs=grads_calcs)

# test accuracy
test_data_generator = SpiralClassifier(points_per_class=100, num_classes=3)
X_test, Y_test = test_data_generator.make_dataset(seed = 1)
test_accuracy = classifier.accuracy(classifier.last_params, X_test, Y_test)
print(f"Test Accuracy: {test_accuracy * 100:.4f}%")

"""## Weights"""

initial1, final1 = 2300, 3000
initial2, final2 = 3500, 5000
period = 5000

grads_calcs = False

classifier = SpiralClassifier(points_per_class=100, num_classes=3, nn_width=50, learning_rate=0.005, real_time_visualization=True, vis_step_interval=100, track_periodic_weight_diff=True, T=5000, track_weight_diff=True, weight_diff_step_interval = 1, l2_reg = 0., label = "weights_inicio_final", show_markers = True)#0.5*1e-3)
X_train, Y_train = classifier.make_dataset(seed = 0)
T = 5000
total_steps =  25 * 5000 + 1
n_steps_per_epoch = 1
batch_size = 50
classifier.train(X_train, Y_train, T =T, w_max = 1, total_steps = total_steps, n_steps_per_epoch = n_steps_per_epoch, batch_size= batch_size, initial1=initial1, final1=final1, initial2=initial2, final2=final2, period=period, grads_calcs=grads_calcs)

# test accuracy
test_data_generator = SpiralClassifier(points_per_class=100, num_classes=3)
X_test, Y_test = test_data_generator.make_dataset(seed = 1)
test_accuracy = classifier.accuracy(classifier.last_params, X_test, Y_test)
print(f"Test Accuracy: {test_accuracy * 100:.4f}%")

"""# angulo 50"""

initial1, final1 = 2300, 3000
initial2, final2 = 3500, 5000
period = 5000

grads_calcs = True

classifier = SpiralClassifier(points_per_class=100, num_classes=3, nn_width=50, learning_rate=0.002, real_time_visualization=True, vis_step_interval=100, track_periodic_weight_diff=True, T=5000, track_weight_diff=True, weight_diff_step_interval = 1, l2_reg = 0., label = "grads_escalon_final", show_markers = True)#0.5*1e-3)
X_train, Y_train = classifier.make_dataset(seed = 0)
T = 5000
total_steps =  25 * 5000 + 1
n_steps_per_epoch = 1
batch_size = 50
classifier.train(X_train, Y_train, T =T, w_max = 70, total_steps = total_steps, n_steps_per_epoch = n_steps_per_epoch, batch_size= batch_size, initial1=initial1, final1=final1, initial2=initial2, final2=final2, period=period, grads_calcs=grads_calcs)

# test accuracy
test_data_generator = SpiralClassifier(points_per_class=100, num_classes=3)
X_test, Y_test = test_data_generator.make_dataset(seed = 1)
test_accuracy = classifier.accuracy(classifier.last_params, X_test, Y_test)
print(f"Test Accuracy: {test_accuracy * 100:.4f}%")

"""# angulo 500"""

initial1, final1 = 2300, 3000
initial2, final2 = 3500, 5000
period = 5000

grads_calcs = True

classifier = SpiralClassifier(points_per_class=100, num_classes=3, nn_width=500, learning_rate=0.002, real_time_visualization=True, vis_step_interval=100, track_periodic_weight_diff=True, T=5000, track_weight_diff=True, weight_diff_step_interval = 1, l2_reg = 0., label = "grads_escalon_final", show_markers = True)#0.5*1e-3)
X_train, Y_train = classifier.make_dataset(seed = 0)
T = 5000
total_steps =  25 * 5000 + 1
n_steps_per_epoch = 1
batch_size = 50
classifier.train(X_train, Y_train, T =T, w_max = 70, total_steps = total_steps, n_steps_per_epoch = n_steps_per_epoch, batch_size= batch_size, initial1=initial1, final1=final1, initial2=initial2, final2=final2, period=period, grads_calcs=grads_calcs)

# test accuracy
test_data_generator = SpiralClassifier(points_per_class=100, num_classes=3)
X_test, Y_test = test_data_generator.make_dataset(seed = 1)
test_accuracy = classifier.accuracy(classifier.last_params, X_test, Y_test)
print(f"Test Accuracy: {test_accuracy * 100:.4f}%")