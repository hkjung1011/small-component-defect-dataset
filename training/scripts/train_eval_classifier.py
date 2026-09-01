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
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from classification_common import (
    FAMILY_BALANCED_PROFILE_ORDER,
    PipelineError,
    build_model,
    build_family_balanced_sampling_plan,
    choose_device,
    create_dataset_class,
    deterministic_split,
    evaluate_model,
    load_and_validate_manifest,
    load_and_validate_auxiliary_condition_manifest,
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
FAMILY_SAMPLING_SEED_OFFSET = 3109


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
        "--auxiliary-condition-manifest",
        type=Path,
        help=(
            "Optional v3 condition manifest. It must contain exactly six train-only "
            "variants for each of the 168 base gradient-train parents."
        ),
    )
    parser.add_argument(
        "--condition-sampling",
        choices=("append", "family-balanced"),
        default="append",
        help=(
            "How to consume an auxiliary condition manifest. 'append' preserves "
            "the C2 behavior; 'family-balanced' draws one base-or-variant image "
            "per parent and epoch for C3."
        ),
    )
    parser.add_argument(
        "--optimizer-update-budget",
        type=int,
        help=(
            "Optional explicit assertion for family-balanced "
            "runs. It must equal epochs * ceil(168 / batch_size); early stopping "
            "is disabled so the exact budget is completed."
        ),
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
                "train_sample_count",
                "optimizer_updates_this_epoch",
                "optimizer_updates_cumulative",
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
            if not bool(torch_module.isfinite(logits).all().item()):
                raise PipelineError("validation produced non-finite logits")
            loss = criterion(logits, targets)
            if not bool(torch_module.isfinite(loss).item()):
                raise PipelineError("validation produced non-finite loss")
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
    auxiliary_samples = []
    auxiliary_audit = None
    if args.auxiliary_condition_manifest is not None:
        auxiliary_samples, auxiliary_audit = (
            load_and_validate_auxiliary_condition_manifest(
                args.auxiliary_condition_manifest,
                config,
                repository_root,
                samples,
                records,
                manifest_audit,
                split_audit,
            )
        )

    if args.condition_sampling == "family-balanced" and auxiliary_audit is None:
        raise PipelineError(
            "--condition-sampling family-balanced requires "
            "--auxiliary-condition-manifest"
        )
    if args.optimizer_update_budget is not None and (
        args.condition_sampling != "family-balanced"
    ):
        raise PipelineError(
            "--optimizer-update-budget is supported only with "
            "--condition-sampling family-balanced"
        )

    training_config = config["training"]
    planned_epochs = int(
        args.epochs if args.epochs is not None else training_config["epochs"]
    )
    if planned_epochs <= 0:
        raise PipelineError("epochs must be positive")
    planned_architecture = str(config["model"]["architecture"])
    planned_freeze_backbone_epochs = int(
        training_config["freeze_backbone_epochs"]
    )
    if planned_architecture != "resnet18":
        planned_freeze_backbone_epochs = 0
    if not 0 <= planned_freeze_backbone_epochs < planned_epochs:
        raise PipelineError("freeze_backbone_epochs must be >= 0 and < epochs")
    batch_size = int(training_config["batch_size"])
    if batch_size <= 0:
        raise PipelineError("training.batch_size must be positive")
    planned_training_seed = int(
        args.training_seed
        if args.training_seed is not None
        else training_config["seed"]
    )
    base_train_samples = split_samples(records, "gradient_train")
    validation_samples = split_samples(records, "validation")
    test_samples = split_samples(records, "test")
    family_epoch_samples: list[list[Any]] | None = None
    family_sampling_audit: dict[str, Any] | None = None
    if args.condition_sampling == "family-balanced":
        family_sampling_seed = (
            planned_training_seed + FAMILY_SAMPLING_SEED_OFFSET
        )
        default_update_budget = planned_epochs * math.ceil(
            len(base_train_samples) / batch_size
        )
        optimizer_update_budget = int(
            args.optimizer_update_budget
            if args.optimizer_update_budget is not None
            else default_update_budget
        )
        family_epoch_samples, family_sampling_audit = (
            build_family_balanced_sampling_plan(
                base_train_samples,
                auxiliary_samples,
                config["classes"],
                validation_samples + test_samples,
                planned_epochs,
                batch_size,
                family_sampling_seed,
                optimizer_update_budget,
            )
        )
        optimizer_update_plan = {
            "mode": "fixed_family_balanced_budget",
            "samples_per_epoch": len(base_train_samples),
            "optimizer_updates_per_epoch": family_sampling_audit[
                "optimizer_updates_per_epoch"
            ],
            "epochs": planned_epochs,
            "planned_optimizer_updates": family_sampling_audit[
                "planned_optimizer_update_count"
            ],
            "optimizer_update_budget": optimizer_update_budget,
            "early_stopping_effective": False,
            "budget_gate": "PASS",
        }
    else:
        append_train_count = len(base_train_samples) + len(auxiliary_samples)
        updates_per_epoch = math.ceil(append_train_count / batch_size)
        optimizer_update_plan = {
            "mode": "epoch_schedule_with_configured_early_stopping",
            "samples_per_epoch": append_train_count,
            "optimizer_updates_per_epoch": updates_per_epoch,
            "epochs": planned_epochs,
            "planned_optimizer_updates_if_no_early_stop": (
                planned_epochs * updates_per_epoch
            ),
            "optimizer_update_budget": None,
            "early_stopping_effective": bool(
                training_config["early_stopping"]["enabled"]
            ),
        }

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
        "condition_sampling": args.condition_sampling,
        "optimizer_update_plan": optimizer_update_plan,
        "warnings": sorted(
            set(manifest_audit["warnings"] + split_audit["warnings"])
        ),
    }
    if auxiliary_audit is not None:
        check_summary["auxiliary_condition_manifest_audit"] = auxiliary_audit
        check_summary["effective_gradient_train_sample_count"] = auxiliary_audit[
            "effective_gradient_train_count"
        ]
        check_summary["warnings"] = sorted(
            set(check_summary["warnings"] + auxiliary_audit["warnings"])
        )
    if family_sampling_audit is not None:
        family_check_summary = copy.deepcopy(family_sampling_audit)
        family_check_summary.pop("per_epoch", None)
        family_check_summary["per_epoch_detail_artifact"] = (
            "family_balanced_sampling_plan.json when --write-split is used"
        )
        check_summary["family_balanced_sampling_audit"] = family_check_summary
        check_summary["effective_gradient_train_sample_count"] = (
            family_sampling_audit["samples_per_epoch"]
        )
        check_summary["gradient_train_candidate_pool_sample_count"] = (
            family_sampling_audit["candidate_pool_sample_count"]
        )
        check_summary["warnings"] = sorted(
            set(
                check_summary["warnings"]
                + family_sampling_audit["warnings"]
            )
        )
    if args.check_only:
        if args.write_split:
            write_split_artifacts(
                args.write_split.resolve(),
                records,
                manifest_audit,
                split_audit,
                auxiliary_audit,
            )
            write_json(
                args.write_split.resolve() / "optimizer_update_plan.json",
                optimizer_update_plan,
            )
            if family_sampling_audit is not None:
                write_json(
                    args.write_split.resolve()
                    / "family_balanced_sampling_plan.json",
                    family_sampling_audit,
                )
        print(json.dumps(check_summary, ensure_ascii=False, indent=2))
        return 0

    # Must be set before the first CUDA context is created.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    architecture = planned_architecture
    torch_module, image_module, image_ops_module = load_ml_dependencies(
        require_torchvision=architecture == "resnet18"
    )
    training_seed = planned_training_seed
    seed_everything(training_seed, torch_module)
    device = choose_device(args.device, torch_module)

    if int(training_config["num_workers"]) != 0:
        raise PipelineError(
            "num_workers must remain 0 for the strict deterministic Windows run"
        )
    epochs = planned_epochs
    freeze_backbone_epochs = planned_freeze_backbone_epochs

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
    write_split_artifacts(
        output_directory,
        records,
        manifest_audit,
        split_audit,
        auxiliary_audit,
    )
    write_json(output_directory / "config_snapshot.json", config)
    write_json(
        output_directory / "optimizer_update_plan.json",
        optimizer_update_plan,
    )
    if family_sampling_audit is not None:
        write_json(
            output_directory / "family_balanced_sampling_plan.json",
            family_sampling_audit,
        )

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
    if family_epoch_samples is not None:
        train_samples = family_epoch_samples[0]
    else:
        train_samples = base_train_samples + auxiliary_samples
    effective_model_counts = split_audit["model_counts"]
    effective_model_class_counts = split_audit["model_class_counts"]
    effective_model_severity_counts = split_audit["model_severity_counts"]
    effective_model_class_severity_counts = split_audit[
        "model_class_severity_counts"
    ]
    if auxiliary_audit is not None and family_sampling_audit is None:
        class_counter = Counter(sample.label for sample in train_samples)
        severity_counter = Counter(sample.severity for sample in train_samples)
        class_severity_counter = Counter(
            (sample.label, sample.severity) for sample in train_samples
        )
        effective_model_counts = copy.deepcopy(split_audit["model_counts"])
        effective_model_counts["gradient_train"] = len(train_samples)
        effective_model_class_counts = copy.deepcopy(
            split_audit["model_class_counts"]
        )
        effective_model_class_counts["gradient_train"] = {
            class_name: class_counter[class_name] for class_name in classes
        }
        effective_model_severity_counts = copy.deepcopy(
            split_audit["model_severity_counts"]
        )
        effective_model_severity_counts["gradient_train"] = {
            severity_name: severity_counter[severity_name]
            for severity_name in config["severities"]
        }
        effective_model_class_severity_counts = copy.deepcopy(
            split_audit["model_class_severity_counts"]
        )
        effective_model_class_severity_counts["gradient_train"] = {
            class_name: {
                severity_name: class_severity_counter[
                    (class_name, severity_name)
                ]
                for severity_name in config["severities"]
            }
            for class_name in classes
        }
    augmentation = training_config["augmentation"]
    augmentation_metadata = augmentation
    if auxiliary_audit is not None:
        augmentation_metadata = copy.deepcopy(augmentation)
        augmentation_metadata["offline_condition_manifest_sha256"] = (
            auxiliary_audit["manifest_sha256"]
        )
        augmentation_metadata["offline_condition_sample_count"] = (
            auxiliary_audit["sample_count"]
        )
    if family_sampling_audit is not None:
        augmentation_metadata["condition_sampling_mode"] = (
            "family_balanced_parent_variant"
        )
        augmentation_metadata["family_sampling_seed_strategy"] = (
            f"training_seed_plus_{FAMILY_SAMPLING_SEED_OFFSET}"
        )
        augmentation_metadata["family_candidates_per_parent"] = (
            family_sampling_audit["candidates_per_family"]
        )
        augmentation_metadata["family_samples_per_epoch"] = (
            family_sampling_audit["samples_per_epoch"]
        )
        augmentation_metadata["fixed_optimizer_update_budget"] = (
            family_sampling_audit["optimizer_update_budget"]
        )
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
    early_stopping_configured = bool(early_stopping["enabled"])
    early_stopping_enabled = (
        early_stopping_configured and family_sampling_audit is None
    )
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
    optimizer_updates_completed = 0
    actual_family_epoch_audits: list[dict[str, Any]] = []
    family_sample_identity: dict[str, tuple[str, str, str]] = {}
    if family_sampling_audit is not None:
        for sample in base_train_samples:
            family_sample_identity[sample.sample_id] = (
                sample.sample_id,
                "base",
                sample.label,
            )
        for sample in auxiliary_samples:
            family_sample_identity[sample.sample_id] = (
                sample.parent_sample_id,
                sample.condition_profile,
                sample.label,
            )
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
        if family_epoch_samples is not None:
            train_dataset.set_samples(family_epoch_samples[epoch_index])
        train_dataset.set_epoch(epoch_index)
        total_loss = 0.0
        correct = 0
        seen = 0
        epoch_optimizer_updates = 0
        epoch_sample_ids: list[str] = []
        for inputs, targets, _sample_ids in train_loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            if not bool(torch_module.isfinite(logits).all().item()):
                raise PipelineError(
                    f"non-finite logits at epoch {epoch_index + 1}"
                )
            loss = criterion(logits, targets)
            if not bool(torch_module.isfinite(loss).item()):
                raise PipelineError(
                    f"non-finite loss at epoch {epoch_index + 1}: {loss.item()}"
                )
            loss.backward()
            optimizer.step()
            optimizer_updates_completed += 1
            epoch_optimizer_updates += 1
            batch_count = targets.size(0)
            total_loss += float(loss.item()) * batch_count
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            seen += batch_count
            epoch_sample_ids.extend(str(sample_id) for sample_id in _sample_ids)

        if family_sampling_audit is not None:
            expected_samples = family_epoch_samples[epoch_index]
            expected_ids = Counter(sample.sample_id for sample in expected_samples)
            actual_ids = Counter(epoch_sample_ids)
            if actual_ids != expected_ids:
                missing = sorted((expected_ids - actual_ids).elements())
                unexpected = sorted((actual_ids - expected_ids).elements())
                raise PipelineError(
                    f"family-balanced epoch {epoch_index + 1} loader draw mismatch; "
                    f"missing={missing[:10]} unexpected={unexpected[:10]}"
                )
            if epoch_optimizer_updates != family_sampling_audit[
                "optimizer_updates_per_epoch"
            ]:
                raise PipelineError(
                    f"family-balanced epoch {epoch_index + 1} optimizer update "
                    f"count mismatch: expected "
                    f"{family_sampling_audit['optimizer_updates_per_epoch']}, "
                    f"got {epoch_optimizer_updates}"
                )
            parent_counts: Counter[str] = Counter()
            profile_counts: Counter[str] = Counter()
            class_counts: Counter[str] = Counter()
            class_profile_counts: Counter[tuple[str, str]] = Counter()
            actual_lines: list[str] = []
            for sample_id in sorted(epoch_sample_ids):
                parent_id, profile, class_name = family_sample_identity[sample_id]
                parent_counts[parent_id] += 1
                profile_counts[profile] += 1
                class_counts[class_name] += 1
                class_profile_counts[(class_name, profile)] += 1
                actual_lines.append(
                    f"{epoch_index + 1}\0{parent_id}\0{sample_id}\0"
                    f"{class_name}\0{profile}\n"
                )
            if len(parent_counts) != family_sampling_audit["family_count"] or any(
                count != 1 for count in parent_counts.values()
            ):
                raise PipelineError(
                    f"family-balanced epoch {epoch_index + 1} did not consume "
                    "each parent exactly once"
                )
            actual_family_epoch_audits.append(
                {
                    "epoch": epoch_index + 1,
                    "draw_count": len(epoch_sample_ids),
                    "unique_parent_count": len(parent_counts),
                    "optimizer_update_count": epoch_optimizer_updates,
                    "class_counts": {
                        name: class_counts[name] for name in classes
                    },
                    "profile_counts": {
                        name: profile_counts[name]
                        for name in FAMILY_BALANCED_PROFILE_ORDER
                    },
                    "class_profile_counts": {
                        class_name: {
                            profile: class_profile_counts[
                                (class_name, profile)
                            ]
                            for profile in FAMILY_BALANCED_PROFILE_ORDER
                        }
                        for class_name in classes
                    },
                    "draw_set_fingerprint_sha256": hashlib.sha256(
                        "".join(actual_lines).encode("utf-8")
                    ).hexdigest(),
                }
            )

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
            "train_sample_count": seen,
            "optimizer_updates_this_epoch": epoch_optimizer_updates,
            "optimizer_updates_cumulative": optimizer_updates_completed,
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

    if family_sampling_audit is not None:
        expected_update_budget = family_sampling_audit[
            "optimizer_update_budget"
        ]
        if optimizer_updates_completed != expected_update_budget:
            raise PipelineError(
                "family-balanced optimizer update budget was not completed: "
                f"expected {expected_update_budget}, got "
                f"{optimizer_updates_completed}"
            )
        actual_profile_totals: Counter[str] = Counter()
        actual_class_totals: Counter[str] = Counter()
        for epoch_audit in actual_family_epoch_audits:
            actual_profile_totals.update(epoch_audit["profile_counts"])
            actual_class_totals.update(epoch_audit["class_counts"])
        actual_family_audit = {
            "schema_version": "1.0",
            "status": "PASS",
            "mode": family_sampling_audit["mode"],
            "sampling_plan_fingerprint_sha256": family_sampling_audit[
                "sampling_plan_fingerprint_sha256"
            ],
            "epochs_completed": len(actual_family_epoch_audits),
            "draw_count": sum(
                row["draw_count"] for row in actual_family_epoch_audits
            ),
            "unique_family_count_per_epoch": family_sampling_audit[
                "family_count"
            ],
            "optimizer_update_budget": expected_update_budget,
            "optimizer_updates_completed": optimizer_updates_completed,
            "planned_draw_sets_verified": True,
            "every_parent_drawn_once_per_epoch_verified": True,
            "profile_counts": {
                name: actual_profile_totals[name]
                for name in FAMILY_BALANCED_PROFILE_ORDER
            },
            "class_counts": {
                name: actual_class_totals[name] for name in classes
            },
            "per_epoch": actual_family_epoch_audits,
        }
        write_json(
            output_directory / "family_balanced_actual_draws.json",
            actual_family_audit,
        )
    else:
        actual_family_audit = None

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
        "condition_sampling": args.condition_sampling,
        "optimizer_updates_completed": optimizer_updates_completed,
        "planned_optimizer_updates_if_no_early_stop": (
            optimizer_update_plan.get("planned_optimizer_updates")
            or optimizer_update_plan.get(
                "planned_optimizer_updates_if_no_early_stop"
            )
        ),
    }
    if auxiliary_audit is not None:
        checkpoint.update(
            {
                "auxiliary_condition_manifest": auxiliary_audit["manifest"],
                "auxiliary_condition_manifest_sha256": auxiliary_audit[
                    "manifest_sha256"
                ],
                "auxiliary_condition_sample_count": auxiliary_audit[
                    "sample_count"
                ],
                "auxiliary_condition_parent_count": auxiliary_audit[
                    "parent_count"
                ],
                "auxiliary_condition_lineage_fingerprint_sha256": auxiliary_audit[
                    "lineage_fingerprint_sha256"
                ],
                "base_gradient_train_sample_count": len(base_train_samples),
                "effective_gradient_train_sample_count": len(train_samples),
            }
        )
    if family_sampling_audit is not None:
        checkpoint.update(
            {
                "family_sampling_mode": family_sampling_audit["mode"],
                "family_sampling_seed": family_sampling_audit[
                    "sampling_seed"
                ],
                "family_sampling_plan_fingerprint_sha256": (
                    family_sampling_audit[
                        "sampling_plan_fingerprint_sha256"
                    ]
                ),
                "optimizer_update_budget": family_sampling_audit[
                    "optimizer_update_budget"
                ],
                "candidate_pool_sample_count": family_sampling_audit[
                    "candidate_pool_sample_count"
                ],
                "gradient_train_samples_per_epoch": family_sampling_audit[
                    "samples_per_epoch"
                ],
            }
        )
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
        "early_stopping_configured": early_stopping_configured,
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
            "condition_sampling": args.condition_sampling,
            "optimizer_updates_per_full_epoch": optimizer_update_plan[
                "optimizer_updates_per_epoch"
            ],
            "planned_optimizer_updates_if_no_early_stop": (
                optimizer_update_plan.get("planned_optimizer_updates")
                or optimizer_update_plan.get(
                    "planned_optimizer_updates_if_no_early_stop"
                )
            ),
            "optimizer_update_budget": optimizer_update_plan.get(
                "optimizer_update_budget"
            ),
            "optimizer_updates_completed": optimizer_updates_completed,
            "fixed_optimizer_update_budget_enforced": (
                family_sampling_audit is not None
            ),
            "sample_counts": effective_model_counts,
            "samples_per_class": effective_model_class_counts,
            "sample_severity_counts": effective_model_severity_counts,
            "samples_per_class_severity": effective_model_class_severity_counts,
            "head_learning_rate": training_config["head_learning_rate"],
            "fine_tune_learning_rate": training_config[
                "fine_tune_learning_rate"
            ],
            "weight_decay": training_config["weight_decay"],
            "early_stopping": early_stopping,
            "early_stopping_effective_enabled": early_stopping_enabled,
            "augmentation": augmentation_metadata,
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
            set(
                manifest_audit["warnings"]
                + split_audit["warnings"]
                + (auxiliary_audit["warnings"] if auxiliary_audit else [])
                + (
                    family_sampling_audit["warnings"]
                    if family_sampling_audit
                    else []
                )
            )
        ),
    }
    if auxiliary_audit is not None:
        metadata["auxiliary_condition_manifest"] = {
            "path": auxiliary_audit["manifest"],
            "sha256": auxiliary_audit["manifest_sha256"],
            "sample_count": auxiliary_audit["sample_count"],
            "parent_count": auxiliary_audit["parent_count"],
            "variants_per_parent": auxiliary_audit["variants_per_parent"],
            "lineage_fingerprint_sha256": auxiliary_audit[
                "lineage_fingerprint_sha256"
            ],
            "training_use": auxiliary_audit["training_use"],
            "evaluation_eligible": auxiliary_audit["evaluation_eligible"],
        }
        metadata["training"].update(
            {
                "base_gradient_train_sample_count": len(base_train_samples),
                "auxiliary_condition_sample_count": len(auxiliary_samples),
                "auxiliary_condition_manifest_sha256": auxiliary_audit[
                    "manifest_sha256"
                ],
                "effective_gradient_train_sample_count": len(train_samples),
                "auxiliary_scope": "gradient_train only",
                "validation_sample_count_unchanged": len(validation_samples),
                "test_sample_count_unchanged": len(test_samples),
            }
        )
    if family_sampling_audit is not None:
        metadata["family_balanced_sampling"] = {
            "mode": family_sampling_audit["mode"],
            "sampling_seed": family_sampling_audit["sampling_seed"],
            "family_count": family_sampling_audit["family_count"],
            "family_counts_per_class": family_sampling_audit[
                "family_counts_per_class"
            ],
            "candidates_per_family": family_sampling_audit[
                "candidates_per_family"
            ],
            "profile_order": family_sampling_audit["profile_order"],
            "samples_per_epoch": family_sampling_audit[
                "samples_per_epoch"
            ],
            "optimizer_updates_per_epoch": family_sampling_audit[
                "optimizer_updates_per_epoch"
            ],
            "optimizer_update_budget": family_sampling_audit[
                "optimizer_update_budget"
            ],
            "optimizer_updates_completed": optimizer_updates_completed,
            "sampling_plan_fingerprint_sha256": family_sampling_audit[
                "sampling_plan_fingerprint_sha256"
            ],
            "leakage_gate": family_sampling_audit["leakage_gate"],
            "rotation_gate": family_sampling_audit["rotation_gate"],
            "plan_artifact": "family_balanced_sampling_plan.json",
            "actual_draw_artifact": "family_balanced_actual_draws.json",
        }
        metadata["training"].update(
            {
                "candidate_pool_sample_count": family_sampling_audit[
                    "candidate_pool_sample_count"
                ],
                "gradient_train_samples_per_epoch": family_sampling_audit[
                    "samples_per_epoch"
                ],
                "effective_gradient_train_sample_count": (
                    family_sampling_audit["samples_per_epoch"]
                ),
                "family_profile_counts_actual": actual_family_audit[
                    "profile_counts"
                ],
                "family_class_counts_actual": actual_family_audit[
                    "class_counts"
                ],
            }
        )
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
