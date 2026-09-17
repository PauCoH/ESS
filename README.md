# Enhanced Scheduled Sampling for CSTR control

Code and data for the CSTR example in *Mitigating Exposure Bias for Reliable
ANN-Based Control: The Enhanced Scheduled Sampling Framework* (ISA Transactions).
The repository includes ESS training, a pretrained LeakyMLP4 controller and
notebooks for measuring autonomous control performance. MATLAB is not required.

## Installation

Use Python 3.11. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name ess --display-name "ESS (Python 3.11)"
python -m jupyterlab
```

On Windows, activate with `.venv\Scripts\activate`. Select the **ESS (Python 3.11)**
kernel in Jupyter. Restart the kernel after editing Python modules.

## Notebooks

| Notebook | Use |
| --- | --- |
| [01_train_cstr.ipynb](notebooks/01_train_cstr.ipynb) | Train a new controller with ESS. |
| [02_evaluate_trained_model.ipynb](notebooks/02_evaluate_trained_model.ipynb) | Evaluate a completed run with the ANN controlling the process alone. |
| [03_evaluate_paper_model.ipynb](notebooks/03_evaluate_paper_model.ipynb) | Evaluate the included paper checkpoint without training. |

To try the pretrained controller, open **03** and run all cells. To train your
own model, run **01**, then **02**. Paths work from the repository root or the
`notebooks/` directory.

Both evaluation notebooks display reference tracking, control actions and
cumulative IAE for the training and validation trajectories. IAE integrates
`abs(reference - process_output)` in physical signal units over time in minutes.
Trapezoidal and rectangular estimates are reported. Evaluation uses the
linearized CSTR with no PID intervention, blending, noise or weight updates.

## Training

Notebook **01** defaults to LeakyMLP4, 308 epochs and the full datasets: 20,001
samples per trajectory at 0.01-minute intervals. Its configuration cell contains
annotations for every setting. The main choices are:

| Setting | Default | Purpose |
| --- | --- | --- |
| `architecture` | `LeakyMLP4` | Also supports `MLP8` and `LSTM`. |
| `epochs` | `308` | Number of ESS epochs. |
| `dataset_fraction` | `1.0` | Leading fraction of each dataset. |
| `device` | `auto` | Use CUDA when available, otherwise CPU. |
| `plot_every` | `20` | Plot every N epochs, plus the first and last. |
| `show_plots` | `True` | Display figures in the notebook as well as saving them. |
| `fine_tuning.enabled` | `False` | Add a high-autonomy stage from the best eligible ESS model. |

For a short setup check, use:

```python
epochs=3,
dataset_fraction=0.0064,
sampling_schedule='linear',
middle_autonomy=0.2,
late_autonomy=0.5,
selection_min_autonomy=0.5,
```

Restore the defaults for a full experiment. The lower thresholds let this short
run reach checkpoint eligibility. Full training includes plant simulation and
validation every epoch; progress messages appear after both finish.

Plots show reference tracking, PID targets versus ANN predictions, applied
actions colored by the sampled branch, and loss history. ANN-branch actions may
include PID blending. Set `plot_every=1` to see every epoch, or `show_plots=False`
to save figures without displaying them.

### Selecting and evaluating a model

`selected_model.pkl` contains the eligible checkpoint with the lowest autonomous
validation **action MSE**, not necessarily the lowest tracking IAE. Eligibility
requires a late-stage epoch and, by default, at least 95% ANN-selection probability.
`final_ess.pkl` contains the last ESS epoch. `selection.json` records the choice.

Notebook **02** automatically finds the newest completed run. To choose another:

```python
RUN_DIR = Path('outputs/training/ess_cstr_<timestamp>_<id>')
CHECKPOINT_NAME = 'selected_model.pkl'
```

Evaluation loads that run's settings and saved scalers without refitting them.
Notebook **03** instead loads the weights, scalers and metadata bundled under
`pretrained/cstr_leaky_mlp4/`.

## Paper-to-code map

The central loop is `train_epoch()`: select a preceding action, advance the CSTR,
obtain a live PID target on the resulting state, and train the ANN on that target.

| Component | Where to look |
| --- | --- |
| Scheduled sampling | [`ann_probability()`](ess/scheduling.py) defines the schedule; [`train_epoch()`](ess/training.py) draws each PID/ANN selection. |
| Weighted ANN actuation (WANNA) | [`train_epoch()`](ess/training.py) adds optional ANN noise and blends PID/ANN actions on the ANN branch using the sampling probability. |
| Online plant and teacher | [`CSTRPlant` and `PIDController`](ess/processes.py) provide the simulated trajectory and live targets. |
| Action loss and sequence term | [`train_epoch()`](ess/training.py) combines current action MSE with a score over preceding, detached ANN/PID action pairs. The history term does not backpropagate through past actions or the plant. |
| Dynamic learning rate | [`StageLRScheduler`](ess/scheduling.py) monitors training loss early and autonomous-validation MSE late, with phase caps and plateau reductions. |
| Fine-tuning | [`run_experiment()`](ess/runner.py) reloads the eligible best model for the optional high-autonomy stage. |

Feature construction is in [data.py](ess/data.py), network definitions in
[architectures.py](ess/architectures.py), and shared model-input handling in
[models.py](ess/models.py). Training uses a live PID teacher; autonomous
[validation](ess/evaluation.py) uses the PID only as a comparison target.

## Files and results

```text
notebooks/                  Training and evaluation entry points
ess/                        Python implementation
data/cstr/                  Reference trajectories, disturbances and PID parameters
pretrained/cstr_leaky_mlp4/  Paper weights, scalers and metadata
tests/                      Regression checks
outputs/                    Generated results; ignored by Git
```

Each training run creates `outputs/training/ess_cstr_<timestamp>_<id>/`, containing
settings, scalers, checkpoints, `training.log`, `history.csv`, plots and recorded
trajectories. Evaluations save metrics, trajectory CSVs and figures under
`outputs/evaluation/`. Keep scalers and model settings alongside the weights.

Run the checks from the repository root:

```bash
python -m unittest discover -s tests -v
```

For CUDA determinism errors, restart the kernel and run all cells. Import `ess`
before using CUDA; it configures the required cuBLAS workspace. Clear notebook
outputs before committing, and leave generated experiments out of version control.
