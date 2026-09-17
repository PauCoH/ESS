from collections import deque
import numpy as np
import torch
from .architectures import LeakyMLP4, MLP8, CustomLSTMModel


def create_model(architecture, device):
    """Instantiate the selected four-input architecture with dropout/batch norm off."""
    if architecture == 'LeakyMLP4':
        model = LeakyMLP4(inp_size=4, batch_norm=False, dropout_var=False)
    elif architecture == 'MLP8':
        model = MLP8(inp_size=4, batch_norm=False, dropout_var=False)
    elif architecture == 'LSTM':
        model = CustomLSTMModel(input_size=4, hidden_size1=100, hidden_size2=50,
                                mlp_hidden=25, output_size=1, dropout_rate=0)
    else:
        raise ValueError(f'Unknown architecture: {architecture}')
    return model.to(device)


class ControllerInput:
    def __init__(self, architecture, window, device):
        """Maintain one rolling input window, with zero padding at trajectory start."""
        self.architecture = architecture
        self.device = device
        self.history = deque([np.zeros(4, dtype=np.float32) for _ in range(window)], maxlen=window)

    def predict(self, model, normalized_features):
        """Use identical input handling in training, validation and export.

        LSTM evaluations process the full rolling window with a fresh hidden
        state; recurrent memory is not carried across overlapping windows.
        Gradients remain attached to the current prediction's model parameters.
        """
        features = np.asarray(normalized_features, dtype=np.float32).reshape(4)
        if self.architecture == 'LSTM':
            self.history.append(features.copy())
            values = np.stack(self.history)[None, :, :]
        else:
            values = features.reshape(1, 1, 4)
        output, _ = model(torch.as_tensor(values, device=self.device), None)
        return output.reshape(())
