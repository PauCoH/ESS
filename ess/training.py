from collections import deque
import random
import numpy as np
import torch
from .data import FeatureHistory
from .models import ControllerInput
from .processes import CSTRPlant, PIDController

TRAINING_COLUMNS = 'time_min,reference,process_output,error,applied_action,ann_action_next,live_pid_action_next,action_source'


def apply_accumulated_gradients(model, optimizer, count, clip_norm):
    """Average by the actual batch size, including the final partial batch, then clip."""
    if count < 1:
        raise ValueError('Gradient accumulation count must be positive')
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(count)
    if clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def train_epoch(model, optimizer, data, config, autonomy, noise_std, device, rollout_rows=None):
    """Train against a live PID on the sampled trajectory, resetting all simulation state.

    The sequence score retains the existing detached action-history definition.
    It contributes to reported loss but carries no gradient through previous actions.
    When supplied, rollout_rows receives detached physical-unit samples from the
    actual training trajectory. Source codes are -1 for initialization, 0 for
    the PID branch and 1 for the ANN branch (which may include blending/noise).
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    plant, teacher = CSTRPlant(config.sample_time_min), PIDController(**data.pid)
    features = FeatureHistory(config)
    controller = ControllerInput(config.architecture, config.lstm_window, device)
    past = deque(maxlen=config.sequence_length)
    previous_ann = previous_pid = 0.0
    sums = np.zeros(3)
    count = 0
    trajectory = data.training
    mean, scale = data.output_scaler.mean_[0], data.output_scaler.scale_[0]
    for i, reference in enumerate(trajectory.reference):
        use_ann = random.random() < autonomy
        if i == 0:
            action = 0.0
        else:
            normalized_action = previous_pid
            if use_ann:
                normalized_action = previous_ann + (np.random.normal(0, noise_std) if noise_std else 0.0)
                if config.weighted_actions:
                    normalized_action = (1 - autonomy) * previous_pid + autonomy * normalized_action
            action = float(normalized_action * scale + mean)
        measurement = plant.step(action, trajectory.dq[i], trajectory.dcai[i])
        target = float((teacher.update(float(reference), measurement) - mean) / scale)
        values = features.update(float(reference) - measurement, action)
        prediction = controller.predict(model, (values - data.input_scaler.mean_) / data.input_scaler.scale_)
        target_tensor = torch.as_tensor(target, dtype=torch.float32, device=device)
        mse = (prediction - target_tensor).square()
        sequence = torch.zeros((), device=device)
        if config.sequence_score_enabled and len(past) == config.sequence_length:
            pairs = torch.tensor(list(past), dtype=torch.float32, device=device)
            sequence = (pairs[:, 0] - pairs[:, 1]).square().mean()
        loss = mse + config.sequence_score_weight * sequence
        loss.backward()
        count += 1
        sums += [loss.item(), mse.item(), sequence.item()]
        previous_ann, previous_pid = prediction.item(), target_tensor.item()
        past.append((previous_ann, previous_pid))
        if rollout_rows is not None:
            rollout_rows.append([i * config.sample_time_min, float(reference), measurement,
                                 float(reference) - measurement, action,
                                 previous_ann * scale + mean, target * scale + mean,
                                 -1 if i == 0 else int(use_ann)])
        if count == config.accumulation_steps or i == len(trajectory.reference) - 1:
            apply_accumulated_gradients(model, optimizer, count, config.gradient_clip_norm)
            count = 0
    return dict(zip(('training_loss', 'action_mse', 'sequence_score'), sums / len(trajectory.reference)))
