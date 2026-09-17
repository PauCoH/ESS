import math
import torch


def ann_probability(epoch, epochs, schedule):
    """Return the existing zero-based ANN probability; also used for WANNA weighting."""
    if schedule == 'linear':
        return epoch / epochs
    if schedule == 'inverse_sigmoid':
        return 1 - 110 / (110 + math.exp((epoch / epochs) * 1000 / 110))
    raise ValueError(f'Unknown sampling schedule: {schedule}')


def phase_for(autonomy, config, fine_tuning=False):
    """Assign phase from ANN autonomy, with a separate optional fine-tuning stage."""
    if fine_tuning:
        return 'fine_tuning'
    if autonomy < config.middle_autonomy:
        return 'early'
    if autonomy < config.late_autonomy:
        return 'middle'
    return 'late'


class StageLRScheduler:
    def __init__(self, optimizer, config):
        """Coordinate phase caps and plateau reductions through a single LR owner."""
        self.optimizer = optimizer
        self.config = config
        self.phase = None
        self.plateau = None
        self.cap = config.lr.maximum

    @property
    def metric_name(self):
        """Early/middle phases use training loss; late/fine-tuning use closed-loop loss."""
        return 'training_loss' if self.phase in {'early', 'middle'} else 'validation_loss'

    def _bound(self):
        """Enforce global bounds and the active phase cap after every adjustment."""
        for group in self.optimizer.param_groups:
            group['lr'] = max(self.config.lr.minimum, min(group['lr'], self.cap, self.config.lr.maximum))

    def begin_phase(self, phase):
        """Apply a phase cap once; preserve previous reductions and reset metric history."""
        if phase != self.phase:
            targets = {'early': self.config.lr.early, 'middle': self.config.lr.middle,
                       'late': self.config.lr.late, 'fine_tuning': self.config.fine_tuning.learning_rate}
            self.cap = targets[phase]
            self._bound()
            self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=self.config.lr.plateau_factor,
                patience=self.config.lr.plateau_patience, min_lr=self.config.lr.minimum)
            self.phase = phase
        self._bound()
        return self.optimizer.param_groups[0]['lr']

    def step(self, training_loss, validation_loss):
        """Reduce on the selected metric without restoring an earlier learning rate."""
        if self.plateau is None:
            raise RuntimeError('Call begin_phase before scheduler.step')
        metric = training_loss if self.metric_name == 'training_loss' else validation_loss
        self.plateau.step(metric)
        self._bound()
        return self.optimizer.param_groups[0]['lr'], metric
