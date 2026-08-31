"""Train once on the fixed 196/504 synthetic split, then evaluate once.

The held-out set is intentionally not consulted for early stopping or model
selection. Because train and test share the same restored synthetic base, its
metrics are a pipeline/synthetic sanity check, not evidence of real-world
generalization.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from classification_common import (
    PipelineError,
    build_model,
    choose_device,
    create_dataset_class,
    deterministic_split,
    evaluate_model,
    load_and_validate_manifest,
    load_config,
    load_ml_dependencies,
    seed_everything,
    sha256_file,
    split_samples,
    write_evaluation_artifacts,
    write_json,
    write_split_artifacts,
)


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "synthetic_v2_700_classifier.json"
DEFAULT_RESULTS_ROOT = SCRIPT_PATH.parents[1] / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic 7-class training/evaluation for synthetic-v2-700 "
            "(28 train + 72 test per class)."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate manifest, SHA-256, counts, ROI, and split without importing torch.",
    )
    parser.add_argument(
        "--write-split",
        type=Path,
        help="With --check-only, optionally write manifest/split audit artifacts here.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Run output directory (default: timestamped directory under training/results).",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, help="Override fixed epoch count.")
    parser.add_argument(
        "--training-seed",
        type=int,
        help="Override only model/data-order/augmentation RNG; dataset split stays fixed.",
    )
    parser.add_argument(
        "--weights-mode",
        choices=("none", "local_path", "torchvision_cache"),
        help="Override model.weights.mode. Network downloads are never attempted.",
    )
    parser.add_argument(
        "--local-weights",
        type=Path,
        help="Local torchvision ResNet-18 state_dict for weights-mode=local_path.",
    )
    return parser.parse_args()


def _make_optimizer(
    model: Any,
    architecture: str,
    torch_module: Any,
    training_config: dict[str, Any],
    backbone_trainable: bool,
) -> Any:
    weight_decay = float(training_config["weight_decay"])
    head_lr = float(training_config["head_learning_rate"])
    fine_tune_lr = float(training_config["fine_tune_learning_rate"])
    if architecture == "resnet18":
        head_parameters = list(model.fc.parameters())
        if backbone_trainable:
            backbone_parameters = [
                parameter
                for name, parameter in model.named_parameters()
                if not name.startswith("fc.")
            ]
            return torch_module.optim.AdamW(
                [
                    {"params": backbone_parameters, "lr": fine_tune_lr},
                    {"params": head_parameters, "lr": head_lr},
                ],
                weight_decay=weight_decay,
            )
        return torch_module.optim.AdamW(
            head_parameters,
            lr=head_lr,
            weight_decay=weight_decay,
        )
    return torch_module.optim.AdamW(
        model.parameters(),
        lr=fine_tune_lr,
        weight_decay=weight_decay,
    )


def _set_resnet_trainable(model: Any, backbone_trainable: bool) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = backbone_trainable or name.startswith("fc.")


def _set_training_mode(model: Any, architecture: str, backbone_trainable: bool) -> None:
    if architecture == "resnet18" and not backbone_trainable:
        # Keep frozen BatchNorm running statistics unchanged during head warm-up.
        model.eval()
        model.fc.train()
    else:
        model.train()


def _save_history(path: Path, history: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "epoch",
                "phase",
                "backbone_trainable",
                "train_loss",
                "train_accuracy",
                "validation_loss",
                "validation_accuracy",
                "selected_as_best",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(history)


def _evaluate_loss_accuracy(
    model: Any,
    loader: Any,
    criterion: Any,
    device: Any,
    torch_module: Any,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0
    with torch_module.no_grad():
        for inputs, targets, _sample_ids in loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            logits = model(inputs)
            loss = criterion(logits, targets)
            batch_count = targets.size(0)
            total_loss += float(loss.item()) * batch_count
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            seen += batch_count
    if seen == 0:
        raise PipelineError("validation loader is empty")
    return total_loss / seen, correct / seen


def main() -> int:
    args = parse_args()
    config, config_path, repository_root = load_config(args.config)
    samples, manifest_audit = load_and_validate_manifest(config, repository_root)
    records, split_audit = deterministic_split(samples, config)

    check_summary = {
        "status": "PASS",
        "release": config["release"],
        "manifest": config["manifest"],
        "manifest_sample_count": manifest_audit["sample_count"],
        "class_counts": manifest_audit["class_counts"],
        "manifest_severity_counts": manifest_audit["severity_counts"],
        "manifest_class_severity_counts": manifest_audit[
            "class_severity_counts"
        ],
        "split_counts": split_audit["counts"],
        "split_class_counts": split_audit["class_counts"],
        "split_severity_counts": split_audit["severity_counts"],
        "split_class_severity_counts": split_audit[
            "class_severity_counts"
        ],
        "model_counts": split_audit["model_counts"],
        "model_class_counts": split_audit["model_class_counts"],
        "model_severity_counts": split_audit["model_severity_counts"],
        "model_class_severity_counts": split_audit[
            "model_class_severity_counts"
        ],
        "split_fingerprint_sha256": split_audit["fingerprint_sha256"],
        "fixed_component_roi_xyxy": manifest_audit[
            "fixed_component_roi_xyxy"
        ],
        "base_group_overlap": split_audit["base_group_overlap"],
        "evaluation_scope": split_audit["evaluation_scope"],
        "warnings": sorted(
            set(manifest_audit["warnings"] + split_audit["warnings"])
        ),
    }
    if args.check_only:
        if args.write_split:
            write_split_artifacts(
                args.write_split.resolve(), records, manifest_audit, split_audit
            )
        print(json.dumps(check_summary, ensure_ascii=False, indent=2))
        return 0

    # Must be set before the first CUDA context is created.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    architecture = str(config["model"]["architecture"])
    torch_module, image_module, image_ops_module = load_ml_dependencies(
        require_torchvision=architecture == "resnet18"
    )
    training_config = config["training"]
    training_seed = int(
        args.training_seed
        if args.training_seed is not None
        else training_config["seed"]
    )
    seed_everything(training_seed, torch_module)
    device = choose_device(args.device, torch_module)

    if int(training_config["num_workers"]) != 0:
        raise PipelineError(
            "num_workers must remain 0 for the strict deterministic Windows run"
        )
    epochs = int(args.epochs if args.epochs is not None else training_config["epochs"])
    if epochs <= 0:
        raise PipelineError("epochs must be positive")
    freeze_backbone_epochs = int(training_config["freeze_backbone_epochs"])
    if architecture != "resnet18":
        freeze_backbone_epochs = 0
    if not 0 <= freeze_backbone_epochs < epochs:
        raise PipelineError("freeze_backbone_epochs must be >= 0 and < epochs")

    output_directory = args.output
    if output_directory is None:
        timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_directory = DEFAULT_RESULTS_ROOT / (
            f"{config['release']}-{architecture}-seed{training_seed}-{timestamp}"
        )
    output_directory = output_directory.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise PipelineError(f"output directory is not empty: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    write_split_artifacts(output_directory, records, manifest_audit, split_audit)
    write_json(output_directory / "config_snapshot.json", config)

    classes: list[str] = list(config["classes"])
    class_to_index = {class_name: index for index, class_name in enumerate(classes)}
    normalization = config["model"]["normalization"]
    mean = normalization["mean"]
    std = normalization["std"]
    if not (
        isinstance(mean, list)
        and isinstance(std, list)
        and len(mean) == 3
        and len(std) == 3
    ):
        raise PipelineError("model.normalization mean/std must each have 3 values")
    input_size = int(config["model"]["input_size"])
    fixed_roi = config["model"]["fixed_component_roi_xyxy"]
    dataset_class = create_dataset_class(
        torch_module, image_module, image_ops_module
    )
    train_samples = split_samples(records, "gradient_train")
    validation_samples = split_samples(records, "validation")
    test_samples = split_samples(records, "test")
    augmentation = training_config["augmentation"]
    augmentation_seed = int(training_config["augmentation_seed"]) + (
        training_seed - int(training_config["seed"])
    )
    train_dataset = dataset_class(
        train_samples,
        class_to_index,
        input_size,
        mean,
        std,
        fixed_roi,
        training=True,
        augmentation=augmentation,
        augmentation_seed=augmentation_seed,
    )
    validation_dataset = dataset_class(
        validation_samples,
        class_to_index,
        input_size,
        mean,
        std,
        fixed_roi,
        training=False,
    )
    test_dataset = dataset_class(
        test_samples,
        class_to_index,
        input_size,
        mean,
        std,
        fixed_roi,
        training=False,
    )
    data_generator = torch_module.Generator()
    data_generator.manual_seed(training_seed)
    batch_size = int(training_config["batch_size"])
    train_loader = torch_module.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=data_generator,
    )
    test_loader = torch_module.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    validation_loader = torch_module.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    local_weights = args.local_weights.resolve() if args.local_weights else None
    weights_mode = args.weights_mode
    if local_weights is not None and weights_mode is None:
        weights_mode = "local_path"
    model, model_audit = build_model(
        config,
        len(classes),
        torch_module,
        weights_mode_override=weights_mode,
        local_weights_override=local_weights,
    )
    model.to(device)
    backbone_trainable = freeze_backbone_epochs == 0
    if architecture == "resnet18":
        _set_resnet_trainable(model, backbone_trainable)
    optimizer = _make_optimizer(
        model,
        architecture,
        torch_module,
        training_config,
        backbone_trainable,
    )
    criterion = torch_module.nn.CrossEntropyLoss()
    early_stopping = training_config["early_stopping"]
    early_stopping_enabled = bool(early_stopping["enabled"])
    early_stopping_patience = int(early_stopping["patience"])
    early_stopping_min_delta = float(early_stopping["min_delta"])
    if early_stopping_patience <= 0:
        raise PipelineError("early_stopping.patience must be positive")

    history: list[dict[str, Any]] = []
    best_validation_loss = float("inf")
    best_epoch: int | None = None
    best_model_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    stopped_early = False
    for epoch_index in range(epochs):
        if architecture == "resnet18" and epoch_index == freeze_backbone_epochs:
            backbone_trainable = True
            _set_resnet_trainable(model, True)
            optimizer = _make_optimizer(
                model,
                architecture,
                torch_module,
                training_config,
                True,
            )
        _set_training_mode(model, architecture, backbone_trainable)
        train_dataset.set_epoch(epoch_index)
        total_loss = 0.0
        correct = 0
        seen = 0
        for inputs, targets, _sample_ids in train_loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            if not torch_module.isfinite(loss):
                raise PipelineError(
                    f"non-finite loss at epoch {epoch_index + 1}: {loss.item()}"
                )
            loss.backward()
            optimizer.step()
            batch_count = targets.size(0)
            total_loss += float(loss.item()) * batch_count
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            seen += batch_count

        phase = "fine_tune" if backbone_trainable else "head_warmup"
        validation_loss, validation_accuracy = _evaluate_loss_accuracy(
            model,
            validation_loader,
            criterion,
            device,
            torch_module,
        )
        selected_as_best = False
        # Head-warmup checkpoints are not eligible: selection begins only after
        # the pretrained backbone is unfrozen.
        selection_eligible = backbone_trainable
        if selection_eligible and (
            validation_loss < best_validation_loss - early_stopping_min_delta
        ):
            best_validation_loss = validation_loss
            best_epoch = epoch_index + 1
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            selected_as_best = True
        elif selection_eligible:
            epochs_without_improvement += 1
        row = {
            "epoch": epoch_index + 1,
            "phase": phase,
            "backbone_trainable": "YES" if backbone_trainable else "NO",
            "train_loss": f"{total_loss / seen:.10f}",
            "train_accuracy": f"{correct / seen:.10f}",
            "validation_loss": f"{validation_loss:.10f}",
            "validation_accuracy": f"{validation_accuracy:.10f}",
            "selected_as_best": "YES" if selected_as_best else "NO",
        }
        history.append(row)
        print(
            f"epoch={epoch_index + 1:03d}/{epochs:03d} phase={phase} "
            f"loss={total_loss / seen:.6f} train_accuracy={correct / seen:.6f}",
            f"validation_loss={validation_loss:.6f} "
            f"validation_accuracy={validation_accuracy:.6f}",
            flush=True,
        )
        if (
            early_stopping_enabled
            and selection_eligible
            and epochs_without_improvement >= early_stopping_patience
        ):
            stopped_early = True
            print(
                f"early_stop epoch={epoch_index + 1} best_epoch={best_epoch} "
                f"best_validation_loss={best_validation_loss:.6f}",
                flush=True,
            )
            break

    if best_model_state is None or best_epoch is None:
        raise PipelineError("no fine-tuning epoch was eligible for model selection")
    model.load_state_dict(best_model_state)

    _save_history(output_directory / "training_history.csv", history)
    checkpoint = {
        "schema_version": "1.0",
        "release": config["release"],
        "classes": classes,
        "class_to_index": class_to_index,
        "model": model_audit,
        "model_state": model.state_dict(),
        "input_size": input_size,
        "fixed_component_roi_xyxy": fixed_roi,
        "normalization": normalization,
        "split_fingerprint_sha256": split_audit["fingerprint_sha256"],
        "manifest_sha256": manifest_audit["manifest_sha256"],
        "training_seed": training_seed,
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "selected_epoch": best_epoch,
        "selection_metric": "validation_loss",
    }
    checkpoint_path = output_directory / "model_final.pt"
    torch_module.save(checkpoint, checkpoint_path)

    # The test set is evaluated exactly once after the fixed training schedule.
    samples_by_id = {sample.sample_id: sample for sample in test_samples}
    predictions, matrix, metrics = evaluate_model(
        model,
        test_loader,
        samples_by_id,
        classes,
        device,
        torch_module,
    )
    try:
        import torchvision

        torchvision_version = torchvision.__version__
    except ModuleNotFoundError:
        torchvision_version = None
    metadata = {
        "release": config["release"],
        "evaluation_scope": config["evaluation"]["scope"],
        "test_set_used_for_model_selection": False,
        "selection_set": "validation (4/class carved from 28/class training pool)",
        "selection_metric": "validation_loss",
        "selected_epoch": best_epoch,
        "early_stopping_enabled": early_stopping_enabled,
        "stopped_early": stopped_early,
        "split_fingerprint_sha256": split_audit["fingerprint_sha256"],
        "manifest_sha256": manifest_audit["manifest_sha256"],
        "base_group_overlap": split_audit["base_group_overlap"],
        "source_specimen_group_overlap": split_audit[
            "source_specimen_group_overlap"
        ],
        "fixed_component_roi_xyxy": fixed_roi,
        "input_size": input_size,
        "normalization": normalization,
        "roi_source": "single class-independent config constant",
        "roi_uses_label_mask_or_bbox": False,
        "model": model_audit,
        "training": {
            "seed": training_seed,
            "epochs_requested": epochs,
            "epochs_completed": len(history),
            "freeze_backbone_epochs": freeze_backbone_epochs,
            "batch_size": batch_size,
            "sample_counts": split_audit["model_counts"],
            "samples_per_class": split_audit["model_class_counts"],
            "sample_severity_counts": split_audit[
                "model_severity_counts"
            ],
            "samples_per_class_severity": split_audit[
                "model_class_severity_counts"
            ],
            "head_learning_rate": training_config["head_learning_rate"],
            "fine_tune_learning_rate": training_config[
                "fine_tune_learning_rate"
            ],
            "weight_decay": training_config["weight_decay"],
            "early_stopping": early_stopping,
            "augmentation": augmentation,
            "augmentation_seed": augmentation_seed,
            "augmentation_scope": "gradient_train only",
            "color_jitter_used": False,
            "validation_or_test_augmentation_used": False,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch_module.__version__,
            "torchvision": torchvision_version,
            "device": str(device),
            "cuda_available": torch_module.cuda.is_available(),
            "cuda_device_name": (
                torch_module.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": torch_module.are_deterministic_algorithms_enabled(),
        },
        "warnings": sorted(
            set(manifest_audit["warnings"] + split_audit["warnings"])
        ),
    }
    write_evaluation_artifacts(
        output_directory,
        classes,
        predictions,
        matrix,
        metrics,
        metadata,
    )
    write_json(output_directory / "run_metadata.json", metadata)
    checkpoint_audit = {
        "path": checkpoint_path.name,
        "sha256": sha256_file(checkpoint_path),
        "note": (
            "Verify this SHA-256 before use; only explicitly allowlisted release "
            "checkpoints are tracked."
        ),
    }
    write_json(output_directory / "checkpoint_audit.json", checkpoint_audit)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output_directory),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_avg"]["f1"],
                "weighted_f1": metrics["weighted_avg"]["f1"],
                "evaluation_scope": config["evaluation"]["scope"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
