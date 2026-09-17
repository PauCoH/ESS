from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import uuid
import numpy as np
import torch
from .artifacts import (CheckpointSelector, export_weights, plot_history, write_history,
                        write_json, plot_training_trajectory, plot_validation_trajectory)
from .data import load_data
from .evaluation import evaluate, ROLL_OUT_COLUMNS
from .models import create_model
from .reproducibility import resolve_device, seed_everything
from .scheduling import ann_probability, phase_for, StageLRScheduler
from .training import train_epoch, TRAINING_COLUMNS


@dataclass
class RunResult:
    directory: Path
    history: list
    selection: dict


def run_experiment(config, repo_root=Path('.')):
    """Run fresh ESS training, optional fine-tuning, eligible selection and final export.

    Each run owns a new directory. Fine-tuning starts from the eligible ESS
    checkpoint with a new optimizer; it is not an interrupted-run resume mode.
    """
    config.validate()
    repo_root = Path(repo_root).resolve()
    seed_everything(config.seed, config.deterministic)
    device = resolve_device(config.device)
    data = load_data(config, repo_root)
    model = create_model(config.architecture, device)
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    directory = repo_root / config.output_dir / f'ess_cstr_{tag}'
    directory.mkdir(parents=True, exist_ok=False)
    for name in ('checkpoints', 'scalers', 'plots', 'trajectories'):
        (directory / name).mkdir()
    write_json(directory / 'config.json', config.to_dict())
    write_json(directory / 'environment.json', dict(device=str(device), seed=config.seed,
               deterministic=config.deterministic, versions={p: version(p) for p in
               ('torch', 'numpy', 'scipy', 'scikit-learn', 'control', 'h5py')}))
    for name, scaler in [('input', data.input_scaler), ('output', data.output_scaler)]:
        np.savetxt(directory / 'scalers' / f'{name}_mean.csv', scaler.mean_)
        np.savetxt(directory / 'scalers' / f'{name}_scale.csv', scaler.scale_)
    history = []

    def log(message):
        """Write each progress record to both the notebook and the run log."""
        print(message, flush=True)
        with (directory / 'training.log').open('a') as stream:
            stream.write(message + '\n')

    def train_stage(stage, epochs, initial_lr, selector):
        """Give each stage its own optimizer and coordinate all LR updates centrally."""
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=config.weight_decay)
        scheduler = StageLRScheduler(optimizer, config)
        for epoch in range(epochs):
            autonomy = config.fine_tuning.autonomy if stage == 'fine_tuning' else ann_probability(epoch, epochs, config.sampling_schedule)
            phase = phase_for(autonomy, config, fine_tuning=stage == 'fine_tuning')
            lr_used = scheduler.begin_phase(phase)
            noise = config.fine_tuning.action_noise_std if stage == 'fine_tuning' else config.action_noise_std
            plot_due = bool(config.plot_every and
                            (epoch == 0 or (epoch + 1) % config.plot_every == 0 or epoch + 1 == epochs))
            training_rows = [] if plot_due else None
            losses = train_epoch(model, optimizer, data, config, autonomy, noise, device, rollout_rows=training_rows)
            validation = evaluate(model, data, config, device, collect_rollout=plot_due)
            lr_next, scheduler_metric = scheduler.step(losses['training_loss'], validation.loss)
            row = dict(stage=stage, epoch=epoch + 1, phase=phase, autonomy=autonomy, lr_used=lr_used,
                       lr_next=lr_next, scheduler_metric_name=scheduler.metric_name,
                       scheduler_metric=scheduler_metric, **losses, validation_loss=validation.loss)
            history.append(row)
            write_history(directory / 'history.csv', history)
            selector.consider(model, validation.loss, epoch + 1, autonomy, phase)
            if config.checkpoint_every and (epoch + 1) % config.checkpoint_every == 0:
                torch.save(model.state_dict(), directory / 'checkpoints' / f'{stage}_epoch_{epoch+1}.pkl')
            log(f'{stage} {epoch+1}/{epochs} | {phase} | autonomy={autonomy:.4f} | '
                f'loss={losses["training_loss"]:.6g} | val={validation.loss:.6g} | '
                f'lr={lr_used:.3g}->{lr_next:.3g} | monitor={scheduler.metric_name}')
            if plot_due:
                prefix = f'{stage}_epoch_{epoch+1}'
                title = f'{stage.replace("_", " ").title()} epoch {epoch+1}/{epochs}'
                np.savetxt(directory / 'trajectories' / f'{prefix}_training.csv', training_rows,
                           delimiter=',', header=TRAINING_COLUMNS, comments='')
                np.savetxt(directory / 'trajectories' / f'{prefix}_validation.csv', validation.rollout,
                           delimiter=',', header=ROLL_OUT_COLUMNS, comments='')
                plot_training_trajectory(training_rows, directory / 'plots' / f'{prefix}_training.png',
                                         config.show_plots, title + ' — training', autonomy, config.weighted_actions)
                plot_validation_trajectory(validation.rollout, directory / 'plots' / f'{prefix}_validation.png',
                                           config.show_plots, title + ' — validation', config.validation_target)
                plot_history(history, directory / 'plots' / f'{prefix}.png', config.show_plots)
                log(f'Plots saved: {directory / "plots" / prefix}_*.png; trajectory CSVs in {directory / "trajectories"}')
        torch.save(model.state_dict(), directory / f'final_{stage}.pkl')

    log(f'Run directory: {directory}')
    ess_selector = CheckpointSelector(directory, 'ess', config.selection_min_autonomy)
    train_stage('ess', config.epochs, config.lr.early, ess_selector)
    if ess_selector.best is None:
        raise RuntimeError(f'No finite eligible ESS checkpoint. Inspect {directory}; early monitoring weights are not exported as the selected model.')
    model.load_state_dict(torch.load(ess_selector.path, map_location=device, weights_only=True))
    export_weights(model, directory / 'best_ess.h5')
    selected = dict(ess_selector.best, checkpoint=ess_selector.path.name)
    fine_selector = None
    if config.fine_tuning.enabled:
        fine_selector = CheckpointSelector(directory, 'fine_tuning', config.selection_min_autonomy)
        log(f'Fine-tuning starts from ESS epoch {ess_selector.best["epoch"]}; optimizer state is reset.')
        train_stage('fine_tuning', config.fine_tuning.epochs, config.fine_tuning.learning_rate, fine_selector)
        if fine_selector.best is not None:
            model.load_state_dict(torch.load(fine_selector.path, map_location=device, weights_only=True))
            export_weights(model, directory / 'best_fine_tuning.h5')
            if fine_selector.best['validation_loss'] < selected['validation_loss']:
                selected = dict(fine_selector.best, checkpoint=fine_selector.path.name)
    model.load_state_dict(torch.load(directory / selected['checkpoint'], map_location=device, weights_only=True))
    torch.save(model.state_dict(), directory / 'selected_model.pkl')
    export_weights(model, directory / 'selected_model.h5')
    selection = dict(selected=selected, ess=ess_selector.best,
                     fine_tuning=fine_selector.best if fine_selector else None)
    write_json(directory / 'selection.json', selection)
    validation = evaluate(model, data, config, device, collect_rollout=True)
    write_json(directory / 'validation_summary.json', dict(validation_loss=validation.loss, target=config.validation_target))
    if config.export_validation:
        np.savetxt(directory / 'validation_rollout.csv', validation.rollout, delimiter=',',
                   header=ROLL_OUT_COLUMNS, comments='')
    plot_history(history, directory / 'plots' / 'loss_history.png', config.show_plots)
    plot_validation_trajectory(validation.rollout, directory / 'plots' / 'selected_model_validation.png',
                               config.show_plots, f'Selected {selected["stage"]} epoch {selected["epoch"]}',
                               config.validation_target)
    log(f'Selected {selected["stage"]} epoch {selected["epoch"]}; validation loss={validation.loss:.6g}')
    return RunResult(directory, history, selection)
