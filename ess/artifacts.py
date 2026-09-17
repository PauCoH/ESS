import csv
import json
import math
import numpy as np
from pathlib import Path
import h5py
import torch


def write_json(path, values):
    """Write readable metadata with a trailing newline."""
    Path(path).write_text(json.dumps(values, indent=2) + '\n')


def export_weights(model, path):
    """Export named tensors; architecture and preprocessing are recorded in config.json."""
    with h5py.File(path, 'w') as handle:
        for name, tensor in model.state_dict().items():
            handle.create_dataset(name, data=tensor.detach().cpu().numpy())


def write_history(path, rows):
    """Persist epoch records after every completed epoch."""
    if rows:
        with Path(path).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


class CheckpointSelector:
    def __init__(self, directory, stage, minimum_autonomy):
        """Keep unrestricted monitoring and eligible deployment candidates separate."""
        self.directory = Path(directory)
        self.stage = stage
        self.minimum_autonomy = minimum_autonomy
        self.monitoring = None
        self.best = None
        self.path = self.directory / f'best_{stage}.pkl'

    def consider(self, model, score, epoch, autonomy, phase):
        """Select finite validation improvements only after the required late-stage threshold."""
        if not math.isfinite(score):
            return
        record = dict(validation_loss=float(score), epoch=epoch, autonomy=autonomy, phase=phase, stage=self.stage)
        if self.monitoring is None or score < self.monitoring['validation_loss']:
            self.monitoring = record.copy()
            torch.save(model.state_dict(), self.directory / f'best_monitoring_{self.stage}.pkl')
        if phase in {'late', 'fine_tuning'} and autonomy >= self.minimum_autonomy:
            if self.best is None or score < self.best['validation_loss']:
                self.best = record.copy()
                torch.save(model.state_dict(), self.path)
        write_json(self.directory / f'selection_{self.stage}.json', dict(monitoring=self.monitoring, eligible=self.best))


def plot_history(rows, path, show):
    """Save training and autonomous-validation loss curves for the completed epochs."""
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(figsize=(9, 4))
    axes.plot(range(1, len(rows) + 1), [row['training_loss'] for row in rows], label='Training loss')
    axes.plot(range(1, len(rows) + 1), [row['validation_loss'] for row in rows], label='Autonomous validation MSE')
    axes.set(xlabel='Completed epoch (ESS followed by optional fine-tuning)', ylabel='Normalized loss')
    axes.grid(True)
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(figure)


def plot_training_trajectory(rows, path, show, title, autonomy, weighted_actions):
    """Plot the recorded training trajectory and actual sampling decisions.

    Current predictions and teacher targets produce next-sample actions; the
    applied-action panel shows the preceding action that advanced the plant.
    """
    import matplotlib.pyplot as plt
    rows = np.asarray(rows)
    time, source = rows[:, 0], rows[:, 7]
    figure, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    eligible = source >= 0
    fraction = np.mean(source[eligible] == 1) if eligible.any() else 0.0
    figure.suptitle(f'{title}\nANN probability: {autonomy:.1%}; observed ANN branch: {fraction:.1%}')
    axes[0].plot(time, rows[:, 1], label='Reference', color='black', linestyle='--')
    axes[0].plot(time, rows[:, 2], label='Process output')
    axes[0].set_ylabel('Process signal')
    axes[1].plot(time, rows[:, 6], label='Live PID target', color='tab:blue')
    axes[1].plot(time, rows[:, 5], label='ANN prediction', color='tab:orange', alpha=.85)
    axes[1].set_ylabel('Next action\n(physical units)')
    axes[2].plot(time, rows[:, 4], color='0.75', linewidth=.7)
    ann_label = 'ANN branch (PID/ANN blend)' if weighted_actions else 'ANN branch'
    for code, label, color in [(-1, 'Initial zero action', 'black'),
                                (0, 'PID branch', 'tab:blue'), (1, ann_label, 'tab:green')]:
        mask = source == code
        axes[2].scatter(time[mask], rows[mask, 4], s=6, label=label, color=color)
    axes[2].set_ylabel('Applied action\n(physical units)')
    axes[3].step(time, source, where='post', linewidth=.7, color='tab:green', label='Applied-action source')
    axes[3].set(yticks=[-1, 0, 1], yticklabels=['Initial', 'PID', 'ANN'],
                ylim=(-1.2, 1.2), ylabel='Selected branch', xlabel='Time (min)')
    for axis in axes:
        axis.grid(True, alpha=.3)
        axis.legend(loc='upper right')
    figure.tight_layout(rect=(0, 0, 1, .95))
    figure.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(figure)


def plot_validation_trajectory(rows, path, show, title, validation_target):
    """Plot an autonomous rollout and compare predictions with the configured target."""
    import matplotlib.pyplot as plt
    rows = np.asarray(rows)
    time = rows[:, 0]
    target_column = 6 if validation_target == 'live_pid' else 7
    target_label = 'Live PID target' if validation_target == 'live_pid' else 'Historical PID target'
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    figure.suptitle(f'{title} — autonomous ANN control')
    axes[0].plot(time, rows[:, 1], label='Reference', color='black', linestyle='--')
    axes[0].plot(time, rows[:, 2], label='Process output')
    axes[0].set_ylabel('Process signal')
    axes[1].plot(time, rows[:, target_column], label=target_label, color='tab:blue')
    axes[1].plot(time, rows[:, 5], label='ANN prediction (next action)', color='tab:orange')
    axes[1].plot(time, rows[:, 4], label='Applied ANN action', color='tab:green', alpha=.65, linestyle=':')
    axes[1].set_ylabel('Action\n(physical units)')
    axes[2].plot(time, rows[:, 3], label='Reference − process output', color='tab:red')
    axes[2].axhline(0, color='black', linewidth=.6)
    axes[2].set(ylabel='Tracking error', xlabel='Time (min)')
    for axis in axes:
        axis.grid(True, alpha=.3)
        axis.legend(loc='upper right')
    figure.tight_layout(rect=(0, 0, 1, .95))
    figure.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(figure)
