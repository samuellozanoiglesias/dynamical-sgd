# Installation

Clone the repository and create the Conda environment:

```bash
conda env create -f environment.yml
conda activate <environment_name>
```

Alternatively, if you prefer using `pip`:

```bash
pip install -r requirements.txt
```

## Running Experiments

The main entry point is:

```bash
python launch.py --config <config_file>
```

For example:

```bash
python launch.py --config config/spiral-no_bumps.yaml
```

Configuration files are stored in the `config/` directory and define all the experimental parameters.

## Repository Structure

```
.
├── analysis/                        # Scripts to generate plots and analyze different training runs
├── config/                          # YAML configuration files for experiments
├── data/                            # Datasets (currently MNIST)
├── old/                             # Deprecated code (can be ignored)
│
├── launch.py                        # Main script to launch experiments
├── training_runner.py               # Training pipeline
├── generate_dataset.py              # Dataset generation utilities
├── model.py                         # Neural network models
│
├── hyperplanes.py                   # Hyperplane analysis
├── classifier_metrics.py            # Classification metrics
├── separability_measures.py         # Feature separability measures
├── neural_collapse.py               # Neural Collapse metrics
├── PCA_analysis.py                  # PCA-based analyses
├── PCA_geometric_overlapping.py     # Geometric overlap analysis using PCA
├── projection_PCA_analysis.py       # Projection-based PCA analyses
│
├── environment.yml                  # Conda environment
├── requirements.txt                 # Python dependencies
└── README.md
```

## Main Components

### `launch.py`

Main entry point of the project. It reads a YAML configuration file and launches the corresponding experiment.

### `config/`

Contains the YAML configuration files used to reproduce different experiments.

### `analysis/`

Contains scripts used to generate plots and perform analyses of different training runs.

### `data/`

Contains the datasets used by the project (currently only MNIST).

### Core training modules

- `training_runner.py`: orchestrates the training process.
- `generate_dataset.py`: dataset creation and preprocessing.
- `model.py`: neural network architectures.

### Analysis and metrics

Several standalone scripts compute different geometric and classification metrics:

- `hyperplanes.py`
- `classifier_metrics.py`
- `separability_measures.py`
- `neural_collapse.py`
- `PCA_analysis.py`
- `PCA_geometric_overlapping.py`
- `projection_PCA_analysis.py`

These scripts are used to evaluate trained models and study the geometry of their learned representations.

## Dependencies

The repository includes both:

- `environment.yml`: recommended for reproducing the complete Conda environment.
- `requirements.txt`: pip equivalent containing the required Python packages.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{dynamical_sgd_2026,
  title={Dynamical SGD: Neural Network Learning as Non-Equilibrium Physics},
  author={Samuel Lozano Iglesias and Nicolas Ratier Werbin},
  year={2026},
  url={https://github.com/NicolasRW/dynamical-sgd}
}
```

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Authors

**Samuel Lozano Iglesias**
- Email: samuel.lozano@ucm.es
- GitHub: [@NicolasRW](https://github.com/samuellozanoiglesias)

**Nicolas Ratier Werbin**
- Email: nicolasratierwerbin@gmail.com
- GitHub: [@NicolasRW](https://github.com/NicolasRW)
