from dataclasses import asdict, dataclass, field
import math


@dataclass
class LearningRateConfig:
    early: float = 1e-3
    middle: float = 1e-4
    late: float = 2e-5
    minimum: float = 2e-6
    maximum: float = 1e-3
    plateau_patience: int = 50
    plateau_factor: float = 0.5


@dataclass
class FineTuningConfig:
    enabled: bool = False
    epochs: int = 50
    autonomy: float = 0.99
    learning_rate: float = 1e-5
    action_noise_std: float = 0.01


@dataclass
class ESSConfig:
    data_dir: str = 'data/cstr'
    output_dir: str = 'outputs/training'
    training_file: str = 'training_reference.mat'
    validation_file: str = 'validation_reference.mat'
    training_excitation: str = 'training_disturbances.mat'
    validation_excitation: str = 'validation_disturbances.mat'
    pid_file: str = 'pid_parameters.mat'
    sample_time_min: float = 0.01
    dataset_fraction: float = 1.0
    architecture: str = 'LeakyMLP4'
    lstm_window: int = 500
    error_window: int = 300
    error_retention: float = 1.0
    epochs: int = 308
    accumulation_steps: int = 128
    seed: int = 0
    deterministic: bool = True
    device: str = 'auto'
    sampling_schedule: str = 'inverse_sigmoid'
    weighted_actions: bool = True
    action_noise_std: float = 0.1
    gradient_clip_norm: float | None = 3.0
    weight_decay: float = 0.001
    sequence_score_enabled: bool = True
    sequence_score_weight: float = 0.1
    sequence_length: int = 15
    middle_autonomy: float = 0.3
    late_autonomy: float = 0.95
    selection_min_autonomy: float = 0.95
    validation_target: str = 'historical_pid'
    plot_every: int = 20
    checkpoint_every: int = 20
    show_plots: bool = True
    export_validation: bool = True
    lr: LearningRateConfig = field(default_factory=LearningRateConfig)
    fine_tuning: FineTuningConfig = field(default_factory=FineTuningConfig)

    def to_dict(self):
        """Return JSON-serializable run settings, including nested stage settings."""
        return asdict(self)

    def validate(self):
        """Reject contradictory schedules, unsupported architectures and invalid ranges."""
        from .scheduling import ann_probability

        if self.architecture not in {'LeakyMLP4', 'MLP8', 'LSTM'}:
            raise ValueError('architecture must be LeakyMLP4, MLP8 or LSTM')
        if self.device not in {'auto', 'cpu', 'cuda'}:
            raise ValueError('device must be auto, cpu or cuda')
        if self.validation_target not in {'historical_pid', 'live_pid'}:
            raise ValueError('validation_target must be historical_pid or live_pid')
        if self.sampling_schedule not in {'inverse_sigmoid', 'linear'}:
            raise ValueError('sampling_schedule must be inverse_sigmoid or linear')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('seed must be an integer in [0, 2**32)')
        for name in ['epochs', 'accumulation_steps', 'lstm_window', 'error_window', 'sequence_length']:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ['plot_every', 'checkpoint_every']:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if not math.isfinite(self.sample_time_min) or self.sample_time_min <= 0:
            raise ValueError('sample_time_min must be finite and positive')
        if not 0 < self.dataset_fraction <= 1:
            raise ValueError('dataset_fraction must be in (0, 1]')
        if not 0 <= self.error_retention <= 1:
            raise ValueError('error_retention must be in [0, 1]')
        if not 0 < self.middle_autonomy < self.late_autonomy <= self.selection_min_autonomy <= 1:
            raise ValueError('Require 0 < middle_autonomy < late_autonomy <= selection_min_autonomy <= 1')
        last_probability = ann_probability(self.epochs - 1, self.epochs, self.sampling_schedule)
        if last_probability < self.selection_min_autonomy:
            raise ValueError(f'This schedule reaches only {last_probability:.6f} autonomy; '
                             'increase epochs or lower the late/selection thresholds for a short check')
        for name in ['action_noise_std', 'weight_decay', 'sequence_score_weight']:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.gradient_clip_norm is not None and (not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0):
            raise ValueError('gradient_clip_norm must be positive or None')
        rates = [self.lr.minimum, self.lr.late, self.lr.middle, self.lr.early, self.lr.maximum]
        if not all(math.isfinite(x) and x > 0 for x in rates) or rates != sorted(rates):
            raise ValueError('Require 0 < minimum <= late <= middle <= early <= maximum learning rates')
        if type(self.lr.plateau_patience) is not int or self.lr.plateau_patience < 0:
            raise ValueError('plateau_patience must be a nonnegative integer')
        if not 0 < self.lr.plateau_factor < 1:
            raise ValueError('plateau_factor must be in (0, 1)')
        ft = self.fine_tuning
        if type(ft.epochs) is not int or ft.epochs < 1:
            raise ValueError('fine_tuning.epochs must be a positive integer')
        if not 0.99 <= ft.autonomy <= 1:
            raise ValueError('fine_tuning.autonomy must be in [0.99, 1]')
        if not math.isfinite(ft.action_noise_std) or ft.action_noise_std < 0:
            raise ValueError('fine_tuning.action_noise_std must be finite and nonnegative')
        if ft.enabled:
            if ft.autonomy < self.selection_min_autonomy:
                raise ValueError('Fine-tuning autonomy must meet checkpoint eligibility')
            if not self.lr.minimum <= ft.learning_rate <= self.lr.late:
                raise ValueError('Fine-tuning LR must lie between the global minimum and late-stage LR')
