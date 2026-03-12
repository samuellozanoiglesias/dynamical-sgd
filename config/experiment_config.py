"""
Configuration management for dynamical SGD experiments.

This module provides configuration classes and utilities for managing
experiment parameters, model settings, and training configurations.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union, Tuple
import json
import yaml
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for dataset generation and processing."""
    
    # Dataset selection
    dataset_name: str = "spiral"  # "spiral" or "mnist"
    
    # Common parameters
    num_classes: int = 3
    test_ratio: float = 0.2
    random_seed: Optional[int] = 0
    
    # Spiral-specific parameters
    points_per_class: int = 100
    revolutions: float = 4.0
    noise_std: float = 0.2
    normalization_method: str = "none"  # "standardize", "minmax", "none"
    augmentation: bool = False
    augmentation_noise_std: float = 0.05
    augmentation_rotation_range: float = 0.1
    randomize_offsets: bool = False  # Generate random angular offsets for spirals
    angular_offsets: Optional[List[float]] = None  # Custom angular offsets in degrees
    min_radius: float = 0.05  # Minimum radius to avoid points at origin (0,0)
    
    # MNIST-specific parameters
    data_dir: str = "./data"
    flatten: bool = True          # Flatten 28x28 images to 784-dim vectors for MLP
    normalize: bool = True        # Apply standard MNIST normalization
    download: bool = True         # Download MNIST if not present


@dataclass
class ModelConfig:
    """Configuration for neural network model."""
    
    input_dim: int = 2            # Input dimension (2 for spiral, 784 for MNIST)
    nn_width: int = 100
    num_hidden_layers: int = 1  # Number of Dense → BatchNorm → ReLU blocks
    num_classes: int = 3
    activation: str = "relu"
    use_bias: bool = True
    use_batchnorm: bool = True  # Whether to include BatchNorm layers
    weight_init_scale: float = 1.0
    random_seed: Optional[int] = 0
    
    # Architecture selection (NEW for DenseNet-40 support)
    architecture: str = "mlp"  # "mlp" or "densenet40"
    
    # DenseNet-40 specific parameters (NEW)
    growth_rate: int = 12          # Number of feature maps per layer
    num_init_features: int = 16    # Initial convolution filters
    block_config: Tuple[int, int, int] = (12, 12, 12)  # Layers in each dense block
    compression: float = 0.5       # Compression factor in transition layers
    dropout_rate: float = 0.0      # Dropout rate (0.0 for NC papers)


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""
    
    optimizer_type: str = "adam"  # "adam", "sgd", "rmsprop"
    learning_rate: float = 0.01
    momentum: float = 0.9  # for SGD
    beta1: float = 0.9  # for Adam
    beta2: float = 0.999  # for Adam
    eps: float = 1e-8  # for Adam
    l2_reg: float = 0.0
    gradient_clipping: Optional[float] = None


@dataclass
class DynamicsConfig:
    """Configuration for dynamical training parameters."""
    
    period_length: int = 5000
    w_max: float = 70.0
    class_focus_pattern: str = "sequential"  # "sequential", "random"
    bumps_before_TPT: bool = True   # Apply bumps before reaching TPT accuracy threshold
    bumps_at_TPT: bool = False      # Apply bumps during/after reaching TPT accuracy threshold
    tpt_accuracy_threshold: float = 1.0  # Accuracy threshold for Terminal Phase Training (e.g., 0.99)
    
    def dynamics_enabled(self) -> bool:
        """Check if any dynamics are enabled."""
        return self.bumps_before_TPT or self.bumps_at_TPT


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    
    total_steps: int = 75000
    batch_size: int = 50
    validation_interval: int = 100
    checkpoint_interval: int = 1000
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    random_seed: Optional[int] = None  # Random seed for reproducibility


@dataclass
class AnalysisConfig:
    """Configuration for analysis and tracking."""
    
    track_weight_diff: bool = True
    weight_diff_step_interval: int = 100
    track_gradients: bool = True
    gradient_tracking_interval: int = 10
    track_distributions: bool = True
    distribution_analysis_ranges: List[Tuple[int, int]] = field(
        default_factory=lambda: [(0, 1500), (2300, 3000), (3500, 5000)]
    )
    compute_kl_divergences: bool = True
    compute_correlations: bool = True
    track_activations: bool = False
    track_neural_collapse: bool = False
    nc_snapshot_interval: int = 2500


@dataclass
class VisualizationConfig:
    """Configuration for visualization and output."""
    
    real_time_visualization: bool = False
    vis_step_interval: int = 100
    save_decision_boundaries: bool = True
    decision_boundary_interval: int = 1000
    save_training_curves: bool = True
    save_weight_evolution: bool = True
    save_gradient_distributions: bool = True
    create_animations: bool = False
    animation_fps: int = 2
    plot_style: str = "seaborn"
    figure_format: str = "png"
    figure_dpi: int = 150
    save_nc_visualizations: bool = True


@dataclass
class OutputConfig:
    """Configuration for output paths and file management."""
    
    output_dir: str = "outputs"
    experiment_name: str = "dynamical_sgd_experiment"
    config_name: Optional[str] = None  # Config file name for nested folder structure
    experiment_timestamp: Optional[str] = None  # For paired experiments
    save_checkpoints: bool = True
    checkpoint_interval: int = 10000
    save_final_model: bool = True
    save_metrics: bool = True
    save_analysis_data: bool = True
    cleanup_intermediate_files: bool = False
    compress_outputs: bool = False


@dataclass
class ExperimentConfig:
    """Main experiment configuration combining all sub-configurations."""
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Global settings
    description: str = "Dynamical SGD experiment with spiral dataset"
    random_seed: Optional[int] = 42
    device: str = "auto"  # "cpu", "gpu", "auto"
    precision: str = "float32"  # "float32", "float64"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate data config
        # points_per_class is only applicable for spiral datasets (can be None for MNIST)
        if self.data.points_per_class is not None and self.data.points_per_class <= 0:
            raise ValueError("points_per_class must be positive when specified")
        if self.data.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        
        # test_ratio validation - can be None for datasets with predefined splits (like MNIST)
        if self.data.test_ratio is not None and not 0 <= self.data.test_ratio <= 1:
            raise ValueError("test_ratio must be between 0 and 1 when specified")
        
        # Validate dataset-specific parameters
        dataset_name = getattr(self.data, 'dataset_name', 'spiral')
        if dataset_name == 'spiral':
            if self.data.points_per_class is None:
                raise ValueError("points_per_class must be specified for spiral dataset")
            if self.data.test_ratio is None:
                raise ValueError("test_ratio must be specified for spiral dataset")
        elif dataset_name == 'mnist':
            # For MNIST, both points_per_class and test_ratio should be None
            # (but we can be lenient if they are specified)
            pass
        
        # Validate model config  
        if self.model.nn_width <= 0:
            raise ValueError("nn_width must be positive")
        if self.model.num_classes != self.data.num_classes:
            self.model.num_classes = self.data.num_classes  # Auto-correct
        
        # Validate optimizer config
        if self.optimizer.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.optimizer.optimizer_type not in ["adam", "sgd", "rmsprop"]:
            raise ValueError("optimizer_type must be one of: adam, sgd, rmsprop")
        
        # Validate dynamics config
        if self.dynamics.period_length <= 0:
            raise ValueError("period_length must be positive")
        if self.dynamics.w_max < 1:
            raise ValueError("w_max must be >= 1")
        
        # Validate training config
        if self.training.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.training.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, '__dict__'):
                # Nested dataclass
                result[field_name] = field_value.__dict__
            else:
                result[field_name] = field_value
        
        return result
    
    def save(self, filepath: Union[str, Path], format: str = "yaml") -> None:
        """
        Save configuration to file.
        
        Args:
            filepath: Path to save configuration
            format: File format ("yaml" or "json")
        """
        filepath = Path(filepath)
        config_dict = self.to_dict()
        
        if format.lower() == "yaml":
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError("format must be 'yaml' or 'json'")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ExperimentConfig':
        """
        Load configuration from file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            Loaded configuration
        """
        filepath = Path(filepath)
        
        if filepath.suffix.lower() in ['.yml', '.yaml']:
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif filepath.suffix.lower() == '.json':
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError("Configuration file must be .yaml, .yml, or .json")
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Configuration object
        """
        # Extract sub-configurations
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        optimizer_config = OptimizerConfig(**config_dict.get('optimizer', {}))
        dynamics_config = DynamicsConfig(**config_dict.get('dynamics', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        analysis_config = AnalysisConfig(**config_dict.get('analysis', {}))
        visualization_config = VisualizationConfig(**config_dict.get('visualization', {}))
        output_config = OutputConfig(**config_dict.get('output', {}))
        
        # Extract global settings
        global_settings = {k: v for k, v in config_dict.items() 
                          if k not in ['data', 'model', 'optimizer', 'dynamics', 
                                     'training', 'analysis', 'visualization', 'output']}
        
        return cls(
            data=data_config,
            model=model_config,
            optimizer=optimizer_config,
            dynamics=dynamics_config,
            training=training_config,
            analysis=analysis_config,
            visualization=visualization_config,
            output=output_config,
            **global_settings
        )
    
    def get_experiment_id(self) -> str:
        """Generate a unique experiment ID based on key parameters."""
        return (f"{self.output.experiment_name}_"
                f"w{self.model.nn_width}_"
                f"bs{self.training.batch_size}_"
                f"lr{self.optimizer.learning_rate}_"
                f"wmax{self.dynamics.w_max}_"
                f"T{self.dynamics.period_length}")


def create_default_config() -> ExperimentConfig:
    """Create a default experiment configuration."""
    return ExperimentConfig()


def create_small_test_config() -> ExperimentConfig:
    """Create a configuration for small/fast testing."""
    config = ExperimentConfig()
    
    # Smaller dataset
    config.data.points_per_class = 50
    
    # Smaller model
    config.model.nn_width = 20
    
    # Shorter training
    config.training.total_steps = 5000
    config.training.batch_size = 20
    
    # Shorter periods
    config.dynamics.period_length = 1000
    
    # Less frequent tracking
    config.analysis.weight_diff_step_interval = 50
    config.visualization.vis_step_interval = 200
    
    return config


def create_large_experiment_config() -> ExperimentConfig:
    """Create a configuration for large-scale experiments."""
    config = ExperimentConfig()
    
    # Larger dataset
    config.data.points_per_class = 200
    
    # Larger model
    config.model.nn_width = 500
    
    # Longer training
    config.training.total_steps = 150000
    config.training.batch_size = 100
    
    # Enable comprehensive analysis
    config.analysis.track_activations = True
    config.analysis.compute_correlations = True
    
    # Enable animations
    config.visualization.create_animations = True
    
    return config


def load_config_with_overrides(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None
) -> ExperimentConfig:
    """
    Load configuration from file with optional parameter overrides.
    
    Args:
        config_path: Path to configuration file
        overrides: Dictionary of parameter overrides
        
    Returns:
        Configuration with overrides applied
    """
    config = ExperimentConfig.load(config_path)
    
    if overrides:
        config_dict = config.to_dict()
        
        # Apply overrides using dot notation (e.g., "model.nn_width": 200)
        for key, value in overrides.items():
            if '.' in key:
                section, param = key.split('.', 1)
                if section in config_dict:
                    config_dict[section][param] = value
            else:
                config_dict[key] = value
        
        config = ExperimentConfig.from_dict(config_dict)
    
    return config