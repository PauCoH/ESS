from dataclasses import dataclass
import numpy as np
import torch
from .data import FeatureHistory
from .models import ControllerInput
from .processes import CSTRPlant, PIDController

ROLL_OUT_COLUMNS = 'time_min,reference,process_output,error,previous_applied_action,ann_action_next,virtual_pid_action,historical_pid_action,dCAi,dQ'


@dataclass
class EvaluationResult:
    loss: float
    rollout: np.ndarray | None


def evaluate(model, data, config, device, collect_rollout=False):
    """Evaluate autonomous control; export uses this exact same trajectory and input protocol."""
    model.eval()
    plant, teacher = CSTRPlant(config.sample_time_min), PIDController(**data.pid)
    history = FeatureHistory(config)
    controller = ControllerInput(config.architecture, config.lstm_window, device)
    mean, scale = data.output_scaler.mean_[0], data.output_scaler.scale_[0]
    trajectory = data.validation
    action, total = 0.0, 0.0
    rows = []
    with torch.no_grad():
        for i, reference in enumerate(trajectory.reference):
            measurement = plant.step(action, trajectory.dq[i], trajectory.dcai[i])
            error = float(reference) - measurement
            virtual_pid = teacher.update(float(reference), measurement)
            features = history.update(error, action)
            prediction = controller.predict(model, (features - data.input_scaler.mean_) / data.input_scaler.scale_)
            target = virtual_pid if config.validation_target == 'live_pid' else trajectory.teacher_action[i]
            target_tensor = torch.as_tensor((target - mean) / scale, dtype=torch.float32, device=device)
            total += (prediction - target_tensor).square().item()
            next_action = prediction.item() * scale + mean
            if collect_rollout:
                rows.append([i * config.sample_time_min, reference, measurement, error, action,
                             next_action, virtual_pid, trajectory.teacher_action[i], trajectory.dcai[i], trajectory.dq[i]])
            action = next_action
    return EvaluationResult(total / len(trajectory.reference), np.asarray(rows) if collect_rollout else None)
