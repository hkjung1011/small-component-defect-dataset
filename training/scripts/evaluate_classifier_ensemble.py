"""Hard-gated equal-weight soft voting for the three v3 condition models.

This script deliberately performs argmax classification only. It does not tune
confidence thresholds and does not enable a HOLD/unknown decision because the
repository has no independent validation set containing real OK and real defect
specimens.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from classification_common import (
    PipelineError,
    build_model,
    calculate_metrics,
    choose_device,
    confusion_matrix,
    create_dataset_class,
    deterministic_split,
    load_and_validate_manifest,
    load_config,
    load_ml_dependencies,
    resolve_repository_path,
    sha256_file,
    split_samples,
    write_evaluation_artifacts,
    write_json,
    write_split_artifacts,
)


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_ENSEMBLE_CONFIG = (
    SCRIPT_PATH.parents[1]
    / "configs"
    / "v3_conditions_soft_voting_ensemble.json"
)
SEVERITIES = ("mild", "moderate", "severe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate three SHA-pinned v3 condition checkpoints by equal-weight "
            "per-class probability soft voting on the immutable 504-image test set."
        )
    )
    parser.add_argument(
        "--ensemble-config", type=Path, default=DEFAULT_ENSEMBLE_CONFIG
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def _read_json(path: Path, context: str) -> dict[str, Any]:
    if not path.is_file():
        raise PipelineError(f"{context} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PipelineError(f"cannot read {context} {path}: {error}") from error
    if not isinstance(value, dict):
        raise PipelineError(f"{context} must contain a JSON object: {path}")
    return value


def _required(mapping: Any, key: str, context: str) -> Any:
    if not isinstance(mapping, dict) or key not in mapping:
        raise PipelineError(f"missing required field: {context}.{key}")
    return mapping[key]


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _target_fingerprint(samples: Sequence[Any]) -> str:
    targets: dict[str, tuple[str, str]] = {}
    for sample in samples:
        if sample.sample_id in targets:
            raise PipelineError(f"duplicate test sample_id: {sample.sample_id}")
        targets[sample.sample_id] = (sample.label, sample.severity)
    canonical = "".join(
        f"{sample_id}\0{targets[sample_id][0]}\0{targets[sample_id][1]}\n"
        for sample_id in sorted(targets)
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_ensemble_contract(
    ensemble_config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    ensemble_config_path = ensemble_config_path.resolve()
    if len(ensemble_config_path.parents) < 3:
        raise PipelineError(
            f"cannot infer repository root from ensemble config: {ensemble_config_path}"
        )
    repository_root = ensemble_config_path.parents[2]
    spec = _read_json(ensemble_config_path, "ensemble config")
    if spec.get("schema_version") != "1.0":
        raise PipelineError("ensemble config schema_version must be '1.0'")
    if spec.get("method") != "equal_weight_per_class_probability_soft_voting":
        raise PipelineError("unsupported ensemble method")

    data_config_entry = _required(spec, "data_config", "ensemble config")
    data_config_relative = _required(data_config_entry, "path", "data_config")
    data_config_sha = _required(data_config_entry, "sha256", "data_config")
    if not isinstance(data_config_relative, str) or not data_config_relative:
        raise PipelineError("data_config.path must be a non-empty repository path")
    if not _valid_sha256(data_config_sha):
        raise PipelineError("data_config.sha256 is invalid")
    data_config_path = resolve_repository_path(
        repository_root, data_config_relative
    )
    actual_data_config_sha = sha256_file(data_config_path)
    if actual_data_config_sha != data_config_sha:
        raise PipelineError(
            "data config SHA mismatch: "
            f"expected={data_config_sha} actual={actual_data_config_sha}"
        )
    data_config, _resolved_data_config, data_repository_root = load_config(
        data_config_path
    )
    if data_repository_root != repository_root:
        raise PipelineError("ensemble config and data config resolve to different roots")
    return spec, data_config, repository_root, ensemble_config_path


def _validate_test_contract(
    spec: dict[str, Any],
    data_config: dict[str, Any],
    samples: Sequence[Any],
    manifest_audit: dict[str, Any],
    split_audit: dict[str, Any],
) -> dict[str, Any]:
    expected = _required(spec, "expected_identity", "ensemble config")
    classes = list(data_config["classes"])
    identity_checks = {
        "release": data_config["release"],
        "classes": classes,
        "manifest_sha256": manifest_audit["manifest_sha256"],
        "split_fingerprint_sha256": split_audit["fingerprint_sha256"],
        "model_architecture": data_config["model"]["architecture"],
    }
    for key, actual in identity_checks.items():
        expected_value = _required(expected, key, "expected_identity")
        if actual != expected_value:
            raise PipelineError(
                f"ensemble identity mismatch at {key}: "
                f"expected={expected_value!r} actual={actual!r}"
            )

    expected_count = int(
        _required(expected, "test_sample_count", "expected_identity")
    )
    if len(samples) != expected_count or expected_count != 504:
        raise PipelineError(
            f"test sample count must be exactly 504, got {len(samples)}"
        )
    target_fingerprint = _target_fingerprint(samples)
    expected_target_fingerprint = _required(
        expected, "test_target_fingerprint_sha256", "expected_identity"
    )
    if target_fingerprint != expected_target_fingerprint:
        raise PipelineError(
            "test target fingerprint mismatch: "
            f"expected={expected_target_fingerprint} actual={target_fingerprint}"
        )

    support = Counter((sample.label, sample.severity) for sample in samples)
    quotas = data_config["split"]["severity_quotas"]
    for class_name in classes:
        for severity in SEVERITIES:
            expected_support = int(quotas[severity]["test"])
            actual_support = support[(class_name, severity)]
            if actual_support != expected_support:
                raise PipelineError(
                    f"test support mismatch for {class_name}/{severity}: "
                    f"expected={expected_support} actual={actual_support}"
                )
    return {
        "status": "PASS",
        "sample_count": len(samples),
        "target_fingerprint_sha256": target_fingerprint,
        "support_per_class_severity": {
            class_name: {
                severity: support[(class_name, severity)]
                for severity in SEVERITIES
            }
            for class_name in classes
        },
    }


def _load_checkpoint(path: Path, torch_module: Any) -> dict[str, Any]:
    try:
        checkpoint = torch_module.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch_module.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise PipelineError(f"unsupported checkpoint structure: {path}")
    return checkpoint


def _checkpoint_identity(
    checkpoint: dict[str, Any],
    expected: dict[str, Any],
    data_config: dict[str, Any],
    manifest_audit: dict[str, Any],
    split_audit: dict[str, Any],
) -> dict[str, Any]:
    classes = list(data_config["classes"])
    class_to_index = {name: index for index, name in enumerate(classes)}
    model_config = data_config["model"]
    expected_fields = {
        "release": data_config["release"],
        "classes": classes,
        "class_to_index": class_to_index,
        "manifest_sha256": manifest_audit["manifest_sha256"],
        "split_fingerprint_sha256": split_audit["fingerprint_sha256"],
        "input_size": model_config["input_size"],
        "fixed_component_roi_xyxy": model_config["fixed_component_roi_xyxy"],
        "normalization": model_config["normalization"],
        "auxiliary_condition_manifest_sha256": _required(
            expected,
            "auxiliary_condition_manifest_sha256",
            "expected_identity",
        ),
        "auxiliary_condition_lineage_fingerprint_sha256": _required(
            expected,
            "auxiliary_condition_lineage_fingerprint_sha256",
            "expected_identity",
        ),
    }
    for key, expected_value in expected_fields.items():
        actual_value = checkpoint.get(key)
        if actual_value != expected_value:
            raise PipelineError(
                f"checkpoint identity mismatch at {key}: "
                f"expected={expected_value!r} actual={actual_value!r}"
            )

    model_identity = checkpoint.get("model")
    if not isinstance(model_identity, dict):
        raise PipelineError("checkpoint model identity must be an object")
    if model_identity.get("architecture") != model_config["architecture"]:
        raise PipelineError("checkpoint model architecture differs from data config")
    expected_weights_sha = _required(
        expected, "pretrained_weights_sha256", "expected_identity"
    )
    if model_identity.get("weights_sha256") != expected_weights_sha:
        raise PipelineError("checkpoint pretrained weights SHA differs from contract")
    return {
        "release": checkpoint["release"],
        "classes": checkpoint["classes"],
        "class_to_index": checkpoint["class_to_index"],
        "manifest_sha256": checkpoint["manifest_sha256"],
        "split_fingerprint_sha256": checkpoint["split_fingerprint_sha256"],
        "model": model_identity,
        "input_size": checkpoint["input_size"],
        "fixed_component_roi_xyxy": checkpoint["fixed_component_roi_xyxy"],
        "normalization": checkpoint["normalization"],
        "auxiliary_condition_manifest_sha256": checkpoint[
            "auxiliary_condition_manifest_sha256"
        ],
        "auxiliary_condition_lineage_fingerprint_sha256": checkpoint[
            "auxiliary_condition_lineage_fingerprint_sha256"
        ],
    }


def _validate_and_load_members(
    spec: dict[str, Any],
    repository_root: Path,
    data_config: dict[str, Any],
    manifest_audit: dict[str, Any],
    split_audit: dict[str, Any],
    torch_module: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    members = _required(spec, "members", "ensemble config")
    if not isinstance(members, list) or len(members) != 3:
        raise PipelineError("ensemble config must pin exactly three members")
    expected = _required(spec, "expected_identity", "ensemble config")
    loaded: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    member_ids: set[str] = set()
    checkpoint_hashes: set[str] = set()
    training_seeds: set[int] = set()
    reference_identity: dict[str, Any] | None = None

    for index, member in enumerate(members):
        context = f"members[{index}]"
        if not isinstance(member, dict):
            raise PipelineError(f"{context} must be an object")
        member_id = _required(member, "member_id", context)
        run_relative = _required(member, "run_directory", context)
        checkpoint_name = _required(member, "checkpoint", context)
        expected_sha = _required(member, "checkpoint_sha256", context)
        expected_seed = _required(member, "training_seed", context)
        if not isinstance(member_id, str) or not member_id:
            raise PipelineError(f"{context}.member_id must be non-empty")
        if member_id in member_ids:
            raise PipelineError(f"duplicate ensemble member_id: {member_id}")
        if not isinstance(run_relative, str) or not run_relative:
            raise PipelineError(f"{context}.run_directory must be non-empty")
        if not isinstance(checkpoint_name, str) or Path(checkpoint_name).name != checkpoint_name:
            raise PipelineError(f"{context}.checkpoint must be a filename")
        if not _valid_sha256(expected_sha):
            raise PipelineError(f"{context}.checkpoint_sha256 is invalid")
        if not isinstance(expected_seed, int):
            raise PipelineError(f"{context}.training_seed must be an integer")
        if expected_sha in checkpoint_hashes:
            raise PipelineError("ensemble members must have distinct checkpoint SHA values")
        if expected_seed in training_seeds:
            raise PipelineError("ensemble members must have distinct training seeds")

        run_directory = resolve_repository_path(repository_root, run_relative)
        checkpoint_path = (run_directory / checkpoint_name).resolve()
        try:
            checkpoint_path.relative_to(run_directory)
        except ValueError as error:
            raise PipelineError(f"checkpoint escapes run directory: {context}") from error
        if not checkpoint_path.is_file():
            raise PipelineError(f"checkpoint not found: {checkpoint_path}")
        actual_sha = sha256_file(checkpoint_path)
        if actual_sha != expected_sha:
            raise PipelineError(
                f"checkpoint SHA mismatch for {member_id}: "
                f"expected={expected_sha} actual={actual_sha}"
            )

        audit = _read_json(run_directory / "checkpoint_audit.json", "checkpoint audit")
        if audit.get("path") != checkpoint_name or audit.get("sha256") != expected_sha:
            raise PipelineError(
                f"checkpoint audit does not match pinned member {member_id}"
            )
        checkpoint = _load_checkpoint(checkpoint_path, torch_module)
        if checkpoint.get("training_seed") != expected_seed:
            raise PipelineError(
                f"training seed mismatch for {member_id}: "
                f"expected={expected_seed} actual={checkpoint.get('training_seed')!r}"
            )
        identity = _checkpoint_identity(
            checkpoint, expected, data_config, manifest_audit, split_audit
        )
        if reference_identity is None:
            reference_identity = identity
        elif identity != reference_identity:
            raise PipelineError(
                f"ensemble checkpoint identity differs for member {member_id}"
            )

        member_ids.add(member_id)
        checkpoint_hashes.add(expected_sha)
        training_seeds.add(expected_seed)
        loaded.append(
            {
                "member_id": member_id,
                "run_directory": Path(run_relative).as_posix(),
                "checkpoint": checkpoint_name,
                "checkpoint_sha256": actual_sha,
                "training_seed": expected_seed,
                "checkpoint_payload": checkpoint,
            }
        )
        gates.append(
            {
                "member_id": member_id,
                "run_directory": Path(run_relative).as_posix(),
                "checkpoint": checkpoint_name,
                "checkpoint_sha256": actual_sha,
                "training_seed": expected_seed,
                "spec_sha_match": True,
                "adjacent_audit_sha_match": True,
                "class_order_match": True,
                "manifest_sha_match": True,
                "split_fingerprint_match": True,
                "checkpoint_identity_match": True,
                "status": "PASS",
            }
        )
    return loaded, gates


def _infer_probabilities(
    model: Any,
    loader: Any,
    classes: Sequence[str],
    device: Any,
    torch_module: Any,
) -> tuple[list[str], list[int], Any]:
    model.eval()
    sample_ids: list[str] = []
    true_indices: list[int] = []
    probability_batches: list[Any] = []
    context = (
        torch_module.inference_mode()
        if hasattr(torch_module, "inference_mode")
        else torch_module.no_grad()
    )
    with context:
        for inputs, targets, batch_sample_ids in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            if logits.ndim != 2 or logits.shape[1] != len(classes):
                raise PipelineError(
                    f"model output shape must be [N,{len(classes)}], got {tuple(logits.shape)}"
                )
            probabilities = torch_module.softmax(logits, dim=1)
            if not bool(torch_module.isfinite(probabilities).all().item()):
                raise PipelineError("model emitted non-finite probabilities")
            row_sums = probabilities.sum(dim=1)
            if not bool(
                torch_module.allclose(
                    row_sums,
                    torch_module.ones_like(row_sums),
                    rtol=1e-5,
                    atol=1e-6,
                )
            ):
                raise PipelineError("model probability rows do not sum to one")
            sample_ids.extend(str(value) for value in batch_sample_ids)
            true_indices.extend(int(value) for value in targets.tolist())
            probability_batches.append(probabilities.detach().cpu().to(torch_module.float64))
    if not probability_batches:
        raise PipelineError("test loader produced no batches")
    return sample_ids, true_indices, torch_module.cat(probability_batches, dim=0)


def _severity_recall(
    samples: Sequence[Any],
    true_indices: Sequence[int],
    predicted_indices: Sequence[int],
    classes: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not (len(samples) == len(true_indices) == len(predicted_indices)):
        raise PipelineError("severity metric inputs have different lengths")
    per_class_severity: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(classes):
        for severity in SEVERITIES:
            positions = [
                index
                for index, sample in enumerate(samples)
                if sample.label == class_name and sample.severity == severity
            ]
            support = len(positions)
            correct = sum(
                1
                for index in positions
                if true_indices[index] == class_index
                and predicted_indices[index] == class_index
            )
            per_class_severity.append(
                {
                    "class": class_name,
                    "severity": severity,
                    "recall": correct / support if support else 0.0,
                    "support": support,
                    "correct": correct,
                }
            )

    overall: list[dict[str, Any]] = []
    for severity in SEVERITIES:
        positions = [
            index
            for index, sample in enumerate(samples)
            if sample.severity == severity
        ]
        support = len(positions)
        correct = sum(
            1
            for index in positions
            if true_indices[index] == predicted_indices[index]
        )
        overall.append(
            {
                "severity": severity,
                "recall": correct / support if support else 0.0,
                "support": support,
                "correct": correct,
            }
        )
    return per_class_severity, overall


def _mean_member_comparison(
    ensemble_metrics: dict[str, Any],
    member_metrics: Sequence[dict[str, Any]],
    ensemble_class_severity: Sequence[dict[str, Any]],
    member_class_severity: Sequence[Sequence[dict[str, Any]]],
    ensemble_overall_severity: Sequence[dict[str, Any]],
    member_overall_severity: Sequence[Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    scalar_fields = {
        "accuracy": lambda metric: metric["accuracy"],
        "macro_f1": lambda metric: metric["macro_avg"]["f1"],
        "macro_precision": lambda metric: metric["macro_avg"]["precision"],
        "macro_recall": lambda metric: metric["macro_avg"]["recall"],
    }
    scalar_comparison: dict[str, Any] = {}
    for name, getter in scalar_fields.items():
        member_values = [float(getter(metric)) for metric in member_metrics]
        ensemble_value = float(getter(ensemble_metrics))
        member_mean = statistics.mean(member_values)
        scalar_comparison[name] = {
            "ensemble": ensemble_value,
            "single_model_3seed_mean": member_mean,
            "ensemble_minus_single_model_3seed_mean": ensemble_value - member_mean,
            "member_values": member_values,
        }

    per_class_recall: list[dict[str, Any]] = []
    for class_index, ensemble_row in enumerate(ensemble_metrics["per_class"]):
        member_values = [
            float(metric["per_class"][class_index]["recall"])
            for metric in member_metrics
        ]
        member_mean = statistics.mean(member_values)
        ensemble_value = float(ensemble_row["recall"])
        per_class_recall.append(
            {
                "class": ensemble_row["class"],
                "ensemble_recall": ensemble_value,
                "single_model_3seed_mean_recall": member_mean,
                "delta": ensemble_value - member_mean,
                "member_values": member_values,
            }
        )

    class_severity_comparison: list[dict[str, Any]] = []
    for row_index, ensemble_row in enumerate(ensemble_class_severity):
        member_values = [
            float(rows[row_index]["recall"]) for rows in member_class_severity
        ]
        member_mean = statistics.mean(member_values)
        ensemble_value = float(ensemble_row["recall"])
        class_severity_comparison.append(
            {
                "class": ensemble_row["class"],
                "severity": ensemble_row["severity"],
                "ensemble_recall": ensemble_value,
                "single_model_3seed_mean_recall": member_mean,
                "delta": ensemble_value - member_mean,
                "support": ensemble_row["support"],
            }
        )

    overall_severity_comparison: list[dict[str, Any]] = []
    for row_index, ensemble_row in enumerate(ensemble_overall_severity):
        member_values = [
            float(rows[row_index]["recall"]) for rows in member_overall_severity
        ]
        member_mean = statistics.mean(member_values)
        ensemble_value = float(ensemble_row["recall"])
        overall_severity_comparison.append(
            {
                "severity": ensemble_row["severity"],
                "ensemble_recall": ensemble_value,
                "single_model_3seed_mean_recall": member_mean,
                "delta": ensemble_value - member_mean,
                "support": ensemble_row["support"],
            }
        )
    return {
        "definition": (
            "Ensemble metrics are computed from one argmax after averaging the "
            "three members' per-class probabilities. The comparator is the "
            "arithmetic mean of three independently argmaxed single-model metrics."
        ),
        "scalar_metrics": scalar_comparison,
        "per_class_recall": per_class_recall,
        "recall_by_class_severity": class_severity_comparison,
        "overall_recall_by_severity": overall_severity_comparison,
    }


def _write_probability_artifacts(
    output: Path,
    samples: Sequence[Any],
    classes: Sequence[str],
    member_records: Sequence[dict[str, Any]],
    ensemble_probabilities: Any,
) -> None:
    probability_columns = [f"probability_{class_name}" for class_name in classes]
    with (output / "member_probabilities.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fieldnames = [
            "member_id",
            "training_seed",
            "sample_id",
            "true_class",
            "severity",
            "predicted_class",
            "confidence",
            *probability_columns,
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for member in member_records:
            probabilities = member["probabilities"]
            predicted = probabilities.argmax(dim=1).tolist()
            confidence = probabilities.max(dim=1).values.tolist()
            for row_index, sample in enumerate(samples):
                row: dict[str, Any] = {
                    "member_id": member["member_id"],
                    "training_seed": member["training_seed"],
                    "sample_id": sample.sample_id,
                    "true_class": sample.label,
                    "severity": sample.severity,
                    "predicted_class": classes[predicted[row_index]],
                    "confidence": f"{confidence[row_index]:.12f}",
                }
                for class_index, column in enumerate(probability_columns):
                    row[column] = f"{probabilities[row_index, class_index].item():.12f}"
                writer.writerow(row)

    ensemble_predicted = ensemble_probabilities.argmax(dim=1).tolist()
    ensemble_confidence = ensemble_probabilities.max(dim=1).values.tolist()
    with (output / "ensemble_probabilities.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fieldnames = [
            "sample_id",
            "image_path",
            "true_class",
            "severity",
            "predicted_class",
            "correct",
            "confidence",
            *probability_columns,
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row_index, sample in enumerate(samples):
            predicted_class = classes[ensemble_predicted[row_index]]
            row = {
                "sample_id": sample.sample_id,
                "image_path": sample.image_path,
                "true_class": sample.label,
                "severity": sample.severity,
                "predicted_class": predicted_class,
                "correct": "YES" if predicted_class == sample.label else "NO",
                "confidence": f"{ensemble_confidence[row_index]:.12f}",
            }
            for class_index, column in enumerate(probability_columns):
                row[column] = (
                    f"{ensemble_probabilities[row_index, class_index].item():.12f}"
                )
            writer.writerow(row)


def _write_severity_csv(
    output: Path,
    per_class_severity: Sequence[dict[str, Any]],
    overall_severity: Sequence[dict[str, Any]],
) -> None:
    with (output / "recall_by_severity.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fieldnames = ["scope", "class", "severity", "recall", "support", "correct"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in per_class_severity:
            writer.writerow(
                {
                    "scope": "class_severity",
                    "class": row["class"],
                    "severity": row["severity"],
                    "recall": f"{float(row['recall']):.10f}",
                    "support": row["support"],
                    "correct": row["correct"],
                }
            )
        for row in overall_severity:
            writer.writerow(
                {
                    "scope": "overall_severity",
                    "class": "__all__",
                    "severity": row["severity"],
                    "recall": f"{float(row['recall']):.10f}",
                    "support": row["support"],
                    "correct": row["correct"],
                }
            )


def _artifact_hashes(output: Path) -> dict[str, Any]:
    files = sorted(
        path for path in output.iterdir() if path.is_file() and path.name != "artifact_hashes.json"
    )
    return {
        "schema_version": "1.0",
        "hash_algorithm": "SHA-256",
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    }


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise PipelineError(f"output directory is not empty: {output}")

    spec, data_config, repository_root, ensemble_config_path = (
        _load_ensemble_contract(args.ensemble_config)
    )
    samples, manifest_audit = load_and_validate_manifest(
        data_config, repository_root
    )
    records, split_audit = deterministic_split(samples, data_config)
    test_samples = split_samples(records, "test")
    test_gate = _validate_test_contract(
        spec,
        data_config,
        test_samples,
        manifest_audit,
        split_audit,
    )

    decision_policy = _required(spec, "decision_policy", "ensemble config")
    if (
        decision_policy.get("prediction")
        != "argmax_of_equal_weight_mean_probability"
        or decision_policy.get("member_weights") != "equal"
        or decision_policy.get("threshold_calibration") != "NOT VERIFIED"
        or decision_policy.get("hold_policy") != "NOT VERIFIED"
    ):
        raise PipelineError(
            "decision policy must be equal-weight argmax with threshold/HOLD NOT VERIFIED"
        )

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    architecture = str(data_config["model"]["architecture"])
    torch_module, image_module, image_ops_module = load_ml_dependencies(
        require_torchvision=architecture == "resnet18"
    )
    torch_module.use_deterministic_algorithms(True)
    if hasattr(torch_module.backends, "cudnn"):
        torch_module.backends.cudnn.deterministic = True
        torch_module.backends.cudnn.benchmark = False
    device = choose_device(args.device, torch_module)

    loaded_members, checkpoint_gates = _validate_and_load_members(
        spec,
        repository_root,
        data_config,
        manifest_audit,
        split_audit,
        torch_module,
    )

    classes = list(data_config["classes"])
    class_to_index = {name: index for index, name in enumerate(classes)}
    model_config = data_config["model"]
    normalization = model_config["normalization"]
    dataset_class = create_dataset_class(
        torch_module, image_module, image_ops_module
    )
    test_dataset = dataset_class(
        test_samples,
        class_to_index,
        int(model_config["input_size"]),
        normalization["mean"],
        normalization["std"],
        model_config["fixed_component_roi_xyxy"],
        training=False,
    )
    test_loader = torch_module.utils.data.DataLoader(
        test_dataset,
        batch_size=int(data_config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    member_records: list[dict[str, Any]] = []
    reference_sample_ids: list[str] | None = None
    reference_targets: list[int] | None = None
    for member in loaded_members:
        model, _model_audit = build_model(
            data_config,
            len(classes),
            torch_module,
            weights_mode_override="none",
        )
        model.load_state_dict(member["checkpoint_payload"]["model_state"], strict=True)
        model.to(device)
        sample_ids, targets, probabilities = _infer_probabilities(
            model, test_loader, classes, device, torch_module
        )
        if reference_sample_ids is None:
            reference_sample_ids = sample_ids
            reference_targets = targets
        elif sample_ids != reference_sample_ids or targets != reference_targets:
            raise PipelineError(
                f"inference target order changed for member {member['member_id']}"
            )
        member_records.append(
            {
                "member_id": member["member_id"],
                "training_seed": member["training_seed"],
                "checkpoint_sha256": member["checkpoint_sha256"],
                "probabilities": probabilities,
            }
        )
        del model
        if device.type == "cuda":
            torch_module.cuda.empty_cache()

    if reference_sample_ids is None or reference_targets is None:
        raise PipelineError("no ensemble members were evaluated")
    expected_sample_ids = [sample.sample_id for sample in test_samples]
    expected_targets = [class_to_index[sample.label] for sample in test_samples]
    if reference_sample_ids != expected_sample_ids or reference_targets != expected_targets:
        raise PipelineError("test loader order/targets differ from deterministic split")

    probability_stack = torch_module.stack(
        [member["probabilities"] for member in member_records], dim=0
    )
    ensemble_probabilities = probability_stack.mean(dim=0)
    if not bool(
        torch_module.allclose(
            ensemble_probabilities.sum(dim=1),
            torch_module.ones(len(test_samples), dtype=torch_module.float64),
            # Member softmax values originate as float32. The accumulation is
            # float64, but it cannot recover precision discarded upstream.
            rtol=1e-6,
            atol=1e-6,
        )
    ):
        raise PipelineError("averaged ensemble probability rows do not sum to one")

    ensemble_predicted = [int(value) for value in ensemble_probabilities.argmax(dim=1).tolist()]
    matrix = confusion_matrix(reference_targets, ensemble_predicted, len(classes))
    metrics = calculate_metrics(matrix, classes)
    per_class_severity, overall_severity = _severity_recall(
        test_samples,
        reference_targets,
        ensemble_predicted,
        classes,
    )

    member_metrics: list[dict[str, Any]] = []
    member_class_severity: list[list[dict[str, Any]]] = []
    member_overall_severity: list[list[dict[str, Any]]] = []
    for member in member_records:
        predicted = [int(value) for value in member["probabilities"].argmax(dim=1).tolist()]
        member_matrix = confusion_matrix(reference_targets, predicted, len(classes))
        calculated = calculate_metrics(member_matrix, classes)
        class_severity_rows, overall_rows = _severity_recall(
            test_samples, reference_targets, predicted, classes
        )
        member_metrics.append(calculated)
        member_class_severity.append(class_severity_rows)
        member_overall_severity.append(overall_rows)
    comparison = _mean_member_comparison(
        metrics,
        member_metrics,
        per_class_severity,
        member_class_severity,
        overall_severity,
        member_overall_severity,
    )

    confidences = ensemble_probabilities.max(dim=1).values.tolist()
    predictions = []
    for index, sample in enumerate(test_samples):
        prediction = ensemble_predicted[index]
        target = reference_targets[index]
        predictions.append(
            {
                "sample_id": sample.sample_id,
                "image_path": sample.image_path,
                "true_class": classes[target],
                "severity": sample.severity,
                "predicted_class": classes[prediction],
                "correct": "YES" if target == prediction else "NO",
                "confidence": f"{confidences[index]:.10f}",
                "split": "test",
            }
        )

    calibration_status = {
        "status": "NOT VERIFIED",
        "threshold_calibration_performed": False,
        "hold_policy_enabled": False,
        "decision_used_for_this_result": "argmax only",
        "reason": decision_policy["reason"],
        "required_next_evidence": (
            "A separately reserved validation set with representative real OK, "
            "real defect, lighting, camera, lot, and specimen groups."
        ),
    }
    portable_members = [
        {
            key: member[key]
            for key in (
                "member_id",
                "run_directory",
                "checkpoint",
                "checkpoint_sha256",
                "training_seed",
            )
        }
        for member in loaded_members
    ]
    metadata = {
        "schema_version": "1.0",
        "release": data_config["release"],
        "method": spec["method"],
        "method_definition": (
            "For each sample and class, take the arithmetic mean of the three "
            "member softmax probabilities, then apply one argmax."
        ),
        "evaluation_scope": data_config["evaluation"]["scope"],
        "test_set_used_for_model_selection": False,
        "member_count": len(loaded_members),
        "members": portable_members,
        "class_order": classes,
        "manifest_sha256": manifest_audit["manifest_sha256"],
        "split_fingerprint_sha256": split_audit["fingerprint_sha256"],
        "test_target_fingerprint_sha256": test_gate[
            "target_fingerprint_sha256"
        ],
        "data_config": spec["data_config"],
        "ensemble_config": ensemble_config_path.name,
        "fixed_component_roi_xyxy": model_config["fixed_component_roi_xyxy"],
        "input_size": model_config["input_size"],
        "normalization": normalization,
        "checkpoint_hard_gate": "PASS",
        "test_target_gate": test_gate,
        "calibration": calibration_status,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch_module.__version__,
            "device": str(device),
        },
        "warnings": sorted(
            set(
                list(manifest_audit["warnings"])
                + list(split_audit["warnings"])
                + [
                    "All ensemble members and test images originate from the same synthetic base component.",
                    "Soft voting on the shared synthetic test set is not real-domain or independent-specimen validation.",
                    "Threshold and HOLD calibration are NOT VERIFIED and were not performed.",
                ]
            )
        ),
    }

    output.mkdir(parents=True, exist_ok=True)
    write_split_artifacts(output, records, manifest_audit, split_audit)
    write_evaluation_artifacts(
        output,
        classes,
        predictions,
        matrix,
        metrics,
        metadata,
    )
    _write_probability_artifacts(
        output, test_samples, classes, member_records, ensemble_probabilities
    )
    _write_severity_csv(output, per_class_severity, overall_severity)

    serializable_member_metrics = []
    for member, calculated, class_rows, overall_rows in zip(
        portable_members,
        member_metrics,
        member_class_severity,
        member_overall_severity,
    ):
        serializable_member_metrics.append(
            {
                **member,
                "metrics": calculated,
                "recall_by_class_severity": class_rows,
                "overall_recall_by_severity": overall_rows,
            }
        )
    write_json(
        output / "checkpoint_gate.json",
        {
            "schema_version": "1.0",
            "overall_status": "PASS",
            "data_config_sha_match": True,
            "test_target_gate": test_gate,
            "members": checkpoint_gates,
        },
    )
    write_json(output / "member_metrics.json", {"members": serializable_member_metrics})
    write_json(output / "comparison_to_single_model_mean.json", comparison)
    write_json(output / "calibration_status.json", calibration_status)
    write_json(output / "ensemble_config_snapshot.json", spec)
    write_json(output / "run_metadata.json", metadata)

    summary_path = output / "metrics_summary.json"
    summary = _read_json(summary_path, "metrics summary")
    summary["schema_version"] = "1.1"
    summary["method"] = spec["method"]
    summary["recall_by_class_severity"] = per_class_severity
    summary["overall_recall_by_severity"] = overall_severity
    summary["comparison_to_single_model_3seed_mean"] = comparison
    summary["calibration"] = calibration_status
    write_json(summary_path, summary)
    write_json(output / "artifact_hashes.json", _artifact_hashes(output))

    print(
        json.dumps(
            {
                "status": "PASS",
                "method": spec["method"],
                "member_count": len(loaded_members),
                "test_sample_count": metrics["sample_count"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_avg"]["f1"],
                "single_model_3seed_mean_macro_f1": comparison["scalar_metrics"][
                    "macro_f1"
                ]["single_model_3seed_mean"],
                "macro_f1_delta": comparison["scalar_metrics"]["macro_f1"][
                    "ensemble_minus_single_model_3seed_mean"
                ],
                "threshold_hold_calibration": "NOT VERIFIED",
                "evaluation_scope": data_config["evaluation"]["scope"],
                "output": str(output),
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
