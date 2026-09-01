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
  C0 `final-stratified-seed-*` checkpoints and three C2
  `v3-conditions-seed-*` checkpoints are explicit exceptions to the `*.pt`
  ignore rule and are intentionally tracked. C3 checkpoints are not tracked.
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

## v3 condition-augmentation transfer experiment

The `v3-conditions-seed-2700701`, `v3-conditions-seed-2700711`, and
`v3-conditions-seed-2700721` directories repeat the same ResNet-18 transfer
recipe while appending the six train-only v3 lighting/camera variants for each
of the 168 gradient-train parents. Their aggregate is
`v3-conditions-aggregate-3seeds`.

- Base 3-seed macro-F1 mean: `0.971091`
- v3 condition 3-seed macro-F1 mean: `0.984178` (`+0.013087`)
- Base mild recall mean: `0.937603`
- v3 condition mild recall mean: `0.962233` (`+0.024631`)
- Discoloration recall changed from `0.995370` to `0.990741`
- Completed C2 optimizer updates by seed: `1,050 / 840 / 1,134`
  (C0 and C3 fixed comparison budget: `180`)

These values are a synthetic same-base ablation only. The simple append recipe
also gives every train parent six extra condition variants and therefore uses a
larger optimizer-update budget than the base run. It is a useful C2 feasibility
result, not a final method selection. None of these metrics is a real-lighting,
independent-specimen, or production accuracy result.

## C3 parent-balanced equal-update control

The three `c3-family-balanced-seed-*` runs select one base-or-condition image
per parent and epoch. They keep the C0-sized budget at exactly 168 draws and six
optimizer updates per epoch, or 180 updates over 30 epochs. Their aggregate is
`c3-family-balanced-aggregate-3seeds`.

- C3 macro-F1 mean: `0.964601` (C0: `-0.006490`, C2: `-0.019577`)
- C3 mild recall mean: `0.919540` (C0: `-0.018062`, C2: `-0.042693`)
- C3 class recall minima: body-chip `0.935185`, contamination `0.944444`
- Split/lineage gate: PASS; validation 28 and test 504 remain base-v2 only
- Planned/completed budget: 180/180 optimizer updates in every run

This controlled result rejects the current C3 recipe as the preferred
classifier. It also shows that the C2 gain cannot yet be attributed to condition
diversity alone: C2 used many more optimizer updates, while C3 did not preserve
its gain at the C0-sized budget. A future comparison must give C0 and a
family-balanced condition recipe the same larger fixed update budget. The C3
checkpoints remain covered by the general `*.pt` ignore rule; metrics, sampling
plans, hashes and one independent re-evaluation are retained.

## C4 equal-weight soft-voting ensemble

`v3-conditions-soft-voting-ensemble` averages the seven class probabilities
from the three SHA-pinned v3 condition checkpoints, then applies one argmax.
On the same fixed 504-image synthetic test set, accuracy is `0.986111` and
macro-F1 is `0.986186`. The corresponding three single-model means are
`0.984127` and `0.984178`, so the ensemble deltas are `+0.001984` accuracy and
`+0.002008` macro-F1. The folder includes member/ensemble probabilities,
class and severity recall, the confusion matrix, checkpoint gates, and artifact
hashes.

The ensemble is not uniformly better: body-crack and discoloration recall are
respectively `-0.009259` and `-0.004630` below their three-member means. This is
still a synthetic same-base sanity result. Confidence-threshold and HOLD/unknown
calibration are explicitly `NOT VERIFIED` because no separately reserved
validation set containing representative real OK and real defect specimens is
available.
