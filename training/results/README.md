# Classification result artifacts

Authoritative outputs for `synthetic-v2-700` are the three
`final-stratified-seed-*` directories and `final-stratified-aggregate-3seeds`.
Any plain `seed-*` directory, if present in a local worktree, is a preliminary
pre-audit run made before class×severity stratification and must not be reported
or committed as a final result.

Each run writes a new subdirectory here. The expected files are:

- `split_assignments.csv`: fixed outer train/test and inner gradient-train/validation membership
- `manifest_audit.json`, `split_audit.json`: integrity, counts, hashes, leakage checks, and split fingerprint
- `training_history.csv`: train/validation loss and accuracy plus selected epoch
- `model_final.pt`: selected validation checkpoint. The three authoritative
  `final-stratified-seed-*` checkpoints are explicit exceptions to the `*.pt`
  ignore rule and are intentionally tracked in this repository.
- `checkpoint_audit.json`: local checkpoint SHA-256
- `predictions.csv`: all 504 test predictions and confidence values
- `metrics_per_class.csv`: precision, recall, F1, support, TP, FP, FN
- `metrics_summary.json`: accuracy, macro/weighted metrics, confusion matrix, and limitations
- `confusion_matrix.csv`, `confusion_matrix.png`: raw and visual confusion matrices
- `run_metadata.json`, `config_snapshot.json`: environment and reproducibility record

The outer split is exactly 28 train-pool + 72 test samples per class. For every
class, mild/moderate/severe quotas are train `11/11/6` and test `29/29/14`.
The train pool is further split into 24 gradient-train + 4 validation samples
per class, giving 168/28/504 samples. Validation always has at least one sample
of each severity; its fourth sample alternates mild/moderate by class index.
Test samples are evaluated once after validation-only model selection.

These releases share one restored synthetic base component. Consequently, even a
high test score is only a synthetic same-base pipeline sanity result. It is not a
measurement of new physical specimens, lighting rigs, cameras, or real defects.
