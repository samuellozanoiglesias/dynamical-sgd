# Dynamical SGD: Neural Network Learning as Non-Equilibrium Physics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-latest-green.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project implements a novel approach to studying neural network learning dynamics by treating machine learning as a non-equilibrium physical system. Instead of using a dynamically evolving cost function, we employ **dynamic batch composition** - where training batches have periodically varying class proportions over time.

### Key Innovation

Rather than selecting completely random subsets of data in each iteration (traditional SGD), we:
- Sample random examples from different classes
- Vary the class proportions cyclically over time
- Study how neural networks internally reconfigure under this periodic training regime

This approach allows us to analyze how different parts of the network coordinate to solve different tasks without destructive interference, similar to the sequential resolution of frustrations observed in physical networks.

## Features

- 🌀 **Spiral Dataset Generation**: Multi-class spiral datasets with configurable parameters
- 🔄 **Dynamic Class Focus**: Periodic emphasis on different classes during training
- 📊 **Comprehensive Analysis**: Weight evolution, gradient distributions, and layer-wise dynamics
- 📈 **Rich Visualizations**: Decision boundaries, training curves, and dynamical analysis plots
- ⚙️ **Flexible Configuration**: YAML-based configuration system for easy experimentation
- 🚀 **GPU Acceleration**: JAX-based implementation with automatic GPU detection
- 📝 **Detailed Logging**: Comprehensive experiment tracking and reproducibility

## Installation

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU (optional, but recommended for larger experiments)
- CUDA 11.0+ (if using GPU)

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/NicolasRW/dynamical-sgd.git
   cd dynamical-sgd
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run a basic experiment:**
   ```bash
   python run_experiment.py --config config/default_config.yaml
   ```

### Development Installation

For development and customization:

```bash
pip install -e .
```

## Usage

### Basic Usage

Run a standard experiment with default parameters:
```bash
python run_experiment.py
```

### Custom Configuration

Create a custom configuration file or override parameters:

```bash
# Using a custom config file
python run_experiment.py --config my_config.yaml

# Override specific parameters
python run_experiment.py --override model.nn_width=200 --override training.batch_size=100

# Multiple overrides
python run_experiment.py \
    --override model.nn_width=500 \
    --override dynamics.w_max=150 \
    --override training.total_steps=100000
```

### Systematic Studies

Run systematic parameter sweeps:
```bash
python systematic_train.py --config config/systematic_study_config.yaml
python systematic_train.py --quick                                    # Fast test
python systematic_train.py --extensive                                # Comprehensive sweep
```

## Configuration

The system uses YAML configuration files for maximum flexibility. Key configuration sections:

### Data Configuration
```yaml
data:
  points_per_class: 100      # Points per spiral class
  num_classes: 3             # Number of spiral classes
  revolutions: 4.0           # Spiral revolutions
  noise_std: 0.2            # Noise level
  test_ratio: 0.2            # Test set fraction
```

### Model Configuration
```yaml
model:
  nn_width: 100              # Hidden layer width
  activation: "relu"         # Activation function
  use_bias: true            # Include bias terms
```

### Dynamics Configuration
```yaml
dynamics:
  period_length: 5000        # Period T for class focus
  w_max: 70.0               # Maximum class weight
  enable_dynamics: true      # Enable dynamic training
```

### Training Configuration
```yaml
training:
  total_steps: 75000         # Total training steps
  batch_size: 50            # Batch size
  learning_rate: 0.01       # Learning rate
```

See `config/default_config.yaml` for complete configuration options.

## Advanced Analysis Modules

For researchers interested in deep analysis of neural network dynamics, additional specialized tools are available in the `analysis/` directory:

### Gradient Distribution Analysis
```bash
# Advanced gradient and KL divergence analysis
python analysis/advanced_gradient_analysis.py --experiment_dir outputs/experiment_001
```

### Weight Correlation Analysis  
```bash
# Parameter correlation matrices and trajectory analysis
python analysis/correlation_analysis.py --weights_file outputs/experiment_001/weights_history.pkl
```

### Specialized Research Experiments
```bash
# Recreate specific analysis workflows from research papers
python analysis/dynamics_experiments.py --experiment_type gradient_phase_analysis
python analysis/dynamics_experiments.py --experiment_type systematic_study
```

These modules provide:
- **KL divergence computation** between gradient distributions across training phases
- **Disjoint distribution analysis** for different parts of the training period
- **Shannon entropy evolution** tracking for gradient and weight distributions
- **Parameter correlation matrices** with layer-wise decomposition and visualization
- **Weight trajectory analysis** including displacement statistics and evolution patterns
- **Specialized experimental workflows** recreating analysis from original research notebooks

## Project Structure

```
dynamical-sgd/
├── src/                    # Core source code
│   └── models/
│       └── spiral_classifier.py  # Main classifier implementation
├── utils/                  # Utility modules
│   ├── data_utils.py      # Data generation and processing
│   ├── visualization.py   # Plotting and visualization
│   ├── analysis.py        # Dynamical analysis tools
│   └── metrics.py         # Metrics and evaluation
├── config/                 # Configuration files
│   ├── experiment_config.py # Configuration classes
│   └── default_config.yaml # Default parameters
├── analysis/              # Advanced research analysis tools
│   ├── advanced_gradient_analysis.py  # KL divergence, distribution analysis
│   ├── correlation_analysis.py        # Weight correlation and trajectories
│   └── dynamics_experiments.py        # Specialized research experiments
├── outputs/               # Experiment outputs (created automatically)
├── docs/                  # Documentation
├── run_experiment.py      # Main experiment script
├── systematic_train.py    # Systematic parameter studies
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Key Concepts

### Dynamic Class Focus

The training alternates between focusing on different classes with period T:
- **Phase 1** (0 ≤ t < T/2): Weight for focus class increases linearly from 1 to w_max
- **Phase 2** (T/2 ≤ t < T): Weight decreases linearly from w_max to 1
- **Class Rotation**: Every T steps, focus moves to the next class

### Analysis Methods

1. **Macroscopic Analysis**:
   - L₂ distance from origin
   - Decision boundary evolution
   - Training and test performance

2. **Microscopic Analysis**:
   - Layer-wise parameter changes
   - Gradient distribution evolution
   - Weight correlation matrices
   - KL divergences between training phases

3. **Dynamical Metrics**:
   - Shannon entropy of weight distributions
   - Cosine similarity between gradient patterns
   - Effective rank of weight matrices

## Results and Visualizations

The system generates comprehensive visualizations:

- **Decision Boundaries**: How classification regions evolve over training
- **Training Curves**: Loss and accuracy progression with period markers
- **Class Focus Dynamics**: Visualization of periodic class emphasis
- **Weight Evolution**: Layer-wise distance from initialization
- **Gradient Distributions**: Analysis of gradient patterns across training phases

Example outputs are saved to `outputs/experiment_name_timestamp/`:
- `config.yaml`: Experiment configuration
- `results.pkl`: Training metrics and final accuracies
- `training_curves.png`: Loss and accuracy plots
- `decision_boundary.png`: Final classification regions
- `class_focus_dynamics.png`: Dynamic class weighting visualization

## Scientific Background

This work is based on the hypothesis that neural network learning can be understood as a non-equilibrium physical process. Key insights:

1. **Frustration Resolution**: Like physical systems with competing interactions, neural networks must resolve conflicts between different learning objectives.

2. **Sequential Learning**: Dynamic class focus allows the network to learn different tasks sequentially while maintaining overall performance.

3. **Internal Coordination**: Analysis reveals how different network components coordinate their changes to avoid destructive interference.

4. **Emergent Organization**: Periodic training induces spontaneous organization patterns in the parameter space.

## Research Applications

This framework enables research into:

- **Catastrophic Forgetting**: How dynamic training helps maintain performance on all classes
- **Continual Learning**: Understanding how networks adapt to new tasks
- **Transfer Learning**: Analyzing how learned representations transfer between related tasks
- **Network Architecture**: Studying how architecture affects learning dynamics
- **Optimization Landscapes**: Mapping the loss landscape under dynamic training

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines. Key areas for contribution:

- Additional analysis methods
- New visualization techniques
- Extended model architectures
- Theoretical analysis tools
- Performance optimizations

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{dynamical_sgd_2024,
  title={Dynamical SGD: Neural Network Learning as Non-Equilibrium Physics},
  author={Nicolas Ratier Werbin},
  year={2024},
  url={https://github.com/NicolasRW/dynamical-sgd}
}
```

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Author

**Nicolas Ratier Werbin**
- Email: nicolasratierwerbin@gmail.com
- GitHub: [@NicolasRW](https://github.com/NicolasRW)

## Acknowledgments

This work explores the intersection of machine learning and statistical physics, building on concepts from:
- Non-equilibrium statistical mechanics
- Dynamical systems theory
- Neural network optimization
- Continual learning research

---

## Quick Examples

### Run a Fast Test
```bash
# Small experiment for testing
python run_experiment.py \
    --override data.points_per_class=50 \
    --override model.nn_width=20 \
    --override training.total_steps=5000
```

### Large-Scale Experiment
```bash
# Comprehensive analysis
python run_experiment.py \
    --override model.nn_width=500 \
    --override training.total_steps=150000 \
    --override analysis.track_activations=true \
    --override visualization.create_animations=true
```

### Disable Dynamics (Standard SGD)
```bash
# Compare with standard training
python run_experiment.py \
    --override dynamics.enable_dynamics=false
```

For more examples and tutorials, see the `docs/` directory.

