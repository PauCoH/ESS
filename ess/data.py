from collections import deque
from dataclasses import dataclass
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler


class FeatureHistory:
    def __init__(self, config):
        """Reset finite and exponentially retained error memory for one trajectory."""
        self.errors = deque(maxlen=config.error_window)
        self.retention = config.error_retention
        self.accumulated = 0.0

    def update(self, error, applied_action):
        """Return error, window sum, applied action and retained error in fixed order."""
        self.errors.append(error)
        self.accumulated = self.retention * self.accumulated + error
        return np.array([error, sum(self.errors), applied_action, self.accumulated])


@dataclass
class Trajectory:
    reference: np.ndarray
    teacher_action: np.ndarray
    features: np.ndarray
    dq: np.ndarray
    dcai: np.ndarray


@dataclass
class DataBundle:
    training: Trajectory
    validation: Trajectory
    input_scaler: StandardScaler
    output_scaler: StandardScaler
    pid: dict


def load_trajectory(path, excitation_path, config):
    """Load a leading data prefix and construct features using historical PID actions."""
    if path.suffix.lower() == '.csv':
        values = np.genfromtxt(path, delimiter=',', names=True)
        reference, error, action = [np.asarray(values[key]).reshape(-1) for key in ('r', 'e', 'uT')]
    else:
        values = loadmat(path)
        reference, error, action = [values[key].reshape(-1) for key in ('ref', 'error', 'u_PI')]
    n = max(1, int(len(reference) * config.dataset_fraction))
    reference, error, action = reference[:n], error[:n], action[:n]
    disturbances = loadmat(excitation_path)
    dq, dcai = [disturbances[key].reshape(-1)[:n] for key in ('dQ', 'dCAi')]
    if not all(len(x) == n for x in (reference, error, action, dq, dcai)):
        raise ValueError('Reference, error, actions and disturbances must cover the selected prefix')
    history = FeatureHistory(config)
    features = np.stack([history.update(float(error[i]), float(action[i-1]) if i else 0.0) for i in range(n)])
    return Trajectory(reference, action, features, dq, dcai)


def load_data(config, repo_root):
    """Fit scalers exclusively on training data and load the CSTR teacher parameters."""
    directory = repo_root / config.data_dir
    training = load_trajectory(directory / config.training_file, directory / config.training_excitation, config)
    validation = load_trajectory(directory / config.validation_file, directory / config.validation_excitation, config)
    parameters = loadmat(directory / config.pid_file)
    p = {key: float(parameters[key].item()) for key in ('Kp', 'Ti', 'Td', 'alpha', 'beta')}
    pid = dict(Kp=p['Kp'], Ti=p['Ti'], Td=p['Td'] / (1 + p['alpha'] * p['Td']), b=p['beta'], c=0, Ts=config.sample_time_min)
    return DataBundle(training, validation, StandardScaler().fit(training.features),
                      StandardScaler().fit(training.teacher_action.reshape(-1, 1)), pid)
