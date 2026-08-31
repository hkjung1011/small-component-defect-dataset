# Synthetic v2 classifier commands

The commands below assume the repository root is the current directory and a
Python environment containing the packages in `requirements-synthetic.txt`:

```powershell
$py = 'python'
```

Integrity and deterministic split check (does not import PyTorch):

```powershell
& $py -B training\scripts\train_eval_classifier.py --check-only
```

Default GPU run. This loads `resnet18-f37072fd.pth` only from the local
torchvision cache; it never downloads model weights:

```powershell
& $py -B training\scripts\train_eval_classifier.py --device cuda
```

Three stochastic training repeats use the identical immutable dataset split:

```powershell
& $py -B training\scripts\train_eval_classifier.py --device cuda --training-seed 2700701 --output training\results\final-stratified-seed-2700701
& $py -B training\scripts\train_eval_classifier.py --device cuda --training-seed 2700711 --output training\results\final-stratified-seed-2700711
& $py -B training\scripts\train_eval_classifier.py --device cuda --training-seed 2700721 --output training\results\final-stratified-seed-2700721
```

Aggregate mean and sample standard deviation across the repeated runs:

```powershell
& $py -B training\scripts\aggregate_seed_results.py --runs training\results\final-stratified-seed-2700701 training\results\final-stratified-seed-2700711 training\results\final-stratified-seed-2700721 --output training\results\final-stratified-aggregate-3seeds
```

Independent checkpoint re-evaluation:

```powershell
& $py -B training\scripts\evaluate_classifier.py --device cuda --checkpoint training\results\final-stratified-seed-2700701\model_final.pt --output training\results\final-stratified-seed-2700701-reeval
```

The fixed component ROI is `[96, 64, 384, 416]` in the 512×512 source image,
then resized to 224×224. It is identical for all classes and partitions and does
not use the label, defect mask, or defect bounding box.

The outer split is 196 train-pool / 504 test. Model fitting uses 168 gradient
train / 28 validation / 504 final test samples. Validation selects the checkpoint;
the 504-image test set is evaluated only after selection.

The split is deterministic within every class×severity cell: each class uses
mild/moderate/severe outer train quotas `11/11/6` and test quotas `29/29/14`.
Validation contains each severity and alternates its fourth mild/moderate sample
by class index. Aggregation rejects duplicate run directories or training seeds,
and rejects any change in manifest, split fingerprint, evaluation scope, model
architecture/weights SHA-256, or planned training hyperparameters.

Verified local environment on 2026-08-31: Python 3.12.13,
`torch 2.11.0+cu128`, `torchvision 0.26.0+cu128`, `Pillow 12.3.0`, CUDA enabled
on NVIDIA GeForce RTX 5060 Laptop GPU. The requirements file intentionally does
not pin the `+cu128` builds because CUDA wheel sources are environment-specific;
every run records the actual library/device versions in `run_metadata.json`.
