# CSTR datasets

These are the original MAT files with descriptive filenames. Their binary
contents and internal variable names have not been changed.

| File | Original filename | Contents |
| --- | --- | --- |
| `training_reference.mat` | `SimData_CSTR_excite.mat` | Training `ref`, `error`, `u_PI` |
| `validation_reference.mat` | `SimData_CSTR_excite2.mat` | Validation `ref`, `error`, `u_PI` |
| `training_disturbances.mat` | `excitation_CSTR_Lineal_excite.mat` | Aligned `dQ`, `dCAi` and excitation signals |
| `validation_disturbances.mat` | `excitation_CSTR_Lineal_excite2.mat` | Aligned validation disturbances |
| `pid_parameters.mat` | `PID_values_1dot6.mat` | `Kp`, `Ti`, `Td`, `alpha`, `beta` |

Each full reference trajectory has 20,001 samples at 0.01-minute spacing,
covering recorded times from 0 to 200 minutes. The original datasets were
obtained under PID control on the nonlinear CSTR. The Python controller training
and evaluation use the repository's linearized CSTR simulation.

Training normalization is fitted only on the selected training prefix. Evaluating
a saved model uses its saved normalization. During autonomous evaluation,
reference and disturbances drive the simulation; recorded PID actions do not
control the plant. Dataset files are read-only inputs to the notebooks.
