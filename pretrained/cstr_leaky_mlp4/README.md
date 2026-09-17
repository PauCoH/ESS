# Paper CSTR controller

Run [03_evaluate_paper_model.ipynb](../../notebooks/03_evaluate_paper_model.ipynb)
to evaluate this model without training.

- `weights.h5`: named PyTorch-compatible tensors, loaded with strict shape/key checks.
- `scalers/`: original input/output means and scales in physical units.
- `metadata.json`: architecture, input order, preprocessing, provenance and SHA256.
- `network.mat`: original MATLAB object, retained for reference; Python does not load it.

The original HDF5 checkpoint does not contain epoch or preprocessing metadata.
The metadata records the original CSTR settings used for this evaluation; it does
not assert that the exact saved epoch is known. Weights and normalization are
unchanged from the supplied research artifacts. Do not refit these scalers before
inference.

The notebook reports ANN-only tracking IAE on the training and validation
trajectories and displays tracking, applied-action and cumulative-IAE plots.
Results are saved under `outputs/evaluation/paper_model/` in the repository root.
