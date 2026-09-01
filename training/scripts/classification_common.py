"""Shared, deterministic utilities for the synthetic 7-class classifier.

The integrity and split code deliberately uses only the Python standard library.
This keeps ``--check-only`` usable before PyTorch is installed or before a model
training environment is provisioned.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


class PipelineError(RuntimeError):
    """Raised when data or configuration cannot satisfy a hard pipeline gate."""


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_path: str
    image_absolute_path: Path
    label: str
    severity: str
    image_sha256: str
    sample_seed: str
    base_group_id: str
    source_specimen_group: str
    domain: str
    qc_status: str
    human_verified: str
    evaluation_eligible: str
    width: int
    height: int
    parent_sample_id: str = ""
    family_split_id: str = ""
    augmentation_family_id: str = ""
    condition_profile: str = ""
    variant_index: int = -1
    derivation_depth: int = 0


@dataclass(frozen=True)
class SplitRecord:
    sample: Sample
    split: str
    model_split: str
    class_severity_rank: int
    split_key_sha256: str
    validation_key_sha256: str


CONDITION_PROFILE_ORDER = (
    "underexposure",
    "overexposure",
    "warm_directional",
    "cool_directional",
    "soft_shadow_vignette",
    "specular_sensor",
)
FAMILY_BALANCED_PROFILE_ORDER = ("base",) + CONDITION_PROFILE_ORDER


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PipelineError(f"config field must be an object: {name}")
    return value


def load_config(config_path: Path) -> tuple[dict[str, Any], Path, Path]:
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise PipelineError(f"config not found: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PipelineError(f"cannot read config {config_path}: {error}") from error
    if not isinstance(config, dict):
        raise PipelineError("top-level config must be an object")

    # training/configs/<name>.json -> repository root
    if len(config_path.parents) < 3:
        raise PipelineError(f"cannot infer repository root from {config_path}")
    repository_root = config_path.parents[2]
    classes = config.get("classes")
    if not isinstance(classes, list) or len(classes) != 7:
        raise PipelineError("config classes must contain exactly 7 entries")
    if any(not isinstance(name, str) or not name for name in classes):
        raise PipelineError("every class name must be a non-empty string")
    if len(set(classes)) != len(classes):
        raise PipelineError("config classes contain duplicates")

    severities = config.get("severities")
    if severities != ["mild", "moderate", "severe"]:
        raise PipelineError(
            "config severities must be ordered as mild, moderate, severe"
        )
    expected_severity_per_class = _require_mapping(
        config.get("expected_severity_per_class"),
        "expected_severity_per_class",
    )
    required_manifest_severity = {"mild": 40, "moderate": 40, "severe": 20}
    if expected_severity_per_class != required_manifest_severity:
        raise PipelineError(
            "expected_severity_per_class must be mild=40, moderate=40, severe=20"
        )

    split_config = _require_mapping(config.get("split"), "split")
    train_per_class = int(split_config.get("train_per_class", -1))
    test_per_class = int(split_config.get("test_per_class", -1))
    expected_per_class = int(config.get("expected_samples_per_class", -1))
    if train_per_class != 28 or test_per_class != 72:
        raise PipelineError("this release requires exactly 28 train and 72 test per class")
    if train_per_class + test_per_class != expected_per_class:
        raise PipelineError(
            "train_per_class + test_per_class must equal expected_samples_per_class"
        )
    if split_config.get("strategy") != "sha256_rank_within_class_severity":
        raise PipelineError("unsupported split strategy")
    if not isinstance(split_config.get("seed"), int):
        raise PipelineError("split.seed must be an integer")
    if int(split_config.get("validation_per_class", -1)) != 4:
        raise PipelineError(
            "the 28/class training pool requires exactly 4/class validation samples"
        )
    if not isinstance(split_config.get("validation_seed"), int):
        raise PipelineError("split.validation_seed must be an integer")
    severity_quotas = _require_mapping(
        split_config.get("severity_quotas"), "split.severity_quotas"
    )
    required_split_quotas = {
        "mild": {"train": 11, "test": 29},
        "moderate": {"train": 11, "test": 29},
        "severe": {"train": 6, "test": 14},
    }
    if severity_quotas != required_split_quotas:
        raise PipelineError(
            "split severity quotas must be mild=11/29, moderate=11/29, "
            "severe=6/14 (train/test)"
        )
    validation_policy = _require_mapping(
        split_config.get("validation_severity_policy"),
        "split.validation_severity_policy",
    )
    if int(validation_policy.get("base_per_severity", -1)) != 1:
        raise PipelineError("validation must include one base sample per severity")
    parity_policy = _require_mapping(
        validation_policy.get("extra_by_class_index_parity"),
        "split.validation_severity_policy.extra_by_class_index_parity",
    )
    if parity_policy != {"even": "mild", "odd": "moderate"}:
        raise PipelineError(
            "validation extra severity must alternate even=mild, odd=moderate"
        )

    _require_mapping(config.get("integrity"), "integrity")
    _require_mapping(config.get("model"), "model")
    training_config = _require_mapping(config.get("training"), "training")
    augmentation = _require_mapping(
        training_config.get("augmentation"), "training.augmentation"
    )
    if float(augmentation.get("rotation_degrees", 0.0)) > 5.0:
        raise PipelineError("train rotation augmentation must not exceed 5 degrees")
    if float(augmentation.get("translation_fraction", 0.0)) > 0.04:
        raise PipelineError("train translation augmentation must not exceed 0.04")
    flip_probability = float(
        augmentation.get("horizontal_flip_probability", 0.0)
    )
    if not 0.0 <= flip_probability <= 0.5:
        raise PipelineError("horizontal flip probability must be between 0 and 0.5")
    if "color_jitter" in augmentation:
        raise PipelineError("ColorJitter is prohibited for this defect-class pipeline")
    _require_mapping(config.get("evaluation"), "evaluation")
    return config, config_path, repository_root


def resolve_repository_path(repository_root: Path, relative_path: str) -> Path:
    candidate = (repository_root / Path(relative_path)).resolve()
    try:
        candidate.relative_to(repository_root)
    except ValueError as error:
        raise PipelineError(f"path escapes repository root: {relative_path}") from error
    return candidate


def load_and_validate_manifest(
    config: dict[str, Any], repository_root: Path
) -> tuple[list[Sample], dict[str, Any]]:
    manifest_value = config.get("manifest")
    if not isinstance(manifest_value, str) or not manifest_value:
        raise PipelineError("config manifest must be a non-empty relative path")
    manifest_path = resolve_repository_path(repository_root, manifest_value)
    if not manifest_path.is_file():
        raise PipelineError(
            "release manifest not found (release generation may still be pending): "
            f"{manifest_path}"
        )

    required_columns = {
        "sample_id",
        "image_path",
        "primary_class",
        "severity",
        "image_sha256",
        "sample_seed",
        "base_group_id",
        "source_specimen_group",
        "domain",
        "qc_status",
        "human_verified",
        "evaluation_eligible",
        "width",
        "height",
    }
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = set(reader.fieldnames or [])
            missing = sorted(required_columns - fieldnames)
            if missing:
                raise PipelineError(
                    "manifest missing required columns: " + ", ".join(missing)
                )
            raw_rows = list(reader)
    except OSError as error:
        raise PipelineError(f"cannot read manifest {manifest_path}: {error}") from error

    classes: list[str] = list(config["classes"])
    class_set = set(classes)
    severities: list[str] = list(config["severities"])
    severity_set = set(severities)
    expected_severity_per_class: dict[str, int] = config[
        "expected_severity_per_class"
    ]
    expected_per_class = int(config["expected_samples_per_class"])
    expected_total = expected_per_class * len(classes)
    if len(raw_rows) != expected_total:
        raise PipelineError(
            f"manifest row count mismatch: expected {expected_total}, got {len(raw_rows)}"
        )

    integrity = config["integrity"]
    verify_hashes = bool(integrity.get("verify_image_sha256", True))
    require_unique_hashes = bool(integrity.get("require_unique_image_sha256", True))
    required_qc_status = str(integrity.get("required_qc_status", "AUTO_PASS"))
    expected_width = int(integrity.get("expected_width", 512))
    expected_height = int(integrity.get("expected_height", 512))
    roi = config["model"].get("fixed_component_roi_xyxy")
    if (
        not isinstance(roi, list)
        or len(roi) != 4
        or any(not isinstance(value, int) for value in roi)
    ):
        raise PipelineError("model.fixed_component_roi_xyxy must contain 4 integers")
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    if not (0 <= roi_x1 < roi_x2 <= expected_width):
        raise PipelineError("fixed component ROI x bounds are invalid")
    if not (0 <= roi_y1 < roi_y2 <= expected_height):
        raise PipelineError("fixed component ROI y bounds are invalid")
    release_root = manifest_path.parents[1]

    samples: list[Sample] = []
    ids: set[str] = set()
    paths: set[str] = set()
    hashes: set[str] = set()
    errors: list[str] = []
    warnings: list[str] = []
    class_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    class_severity_counts: Counter[tuple[str, str]] = Counter()
    domain_counts: Counter[str] = Counter()
    qc_counts: Counter[str] = Counter()
    human_verified_counts: Counter[str] = Counter()
    evaluation_eligible_counts: Counter[str] = Counter()

    for row_index, row in enumerate(raw_rows, start=2):
        sample_id = (row.get("sample_id") or "").strip()
        image_relative = (row.get("image_path") or "").strip()
        label = (row.get("primary_class") or "").strip()
        severity = (row.get("severity") or "").strip()
        image_sha = (row.get("image_sha256") or "").strip().lower()
        sample_seed = (row.get("sample_seed") or "").strip()
        qc_status = (row.get("qc_status") or "").strip()
        human_verified = (row.get("human_verified") or "").strip()
        evaluation_eligible = (row.get("evaluation_eligible") or "").strip()
        try:
            width = int((row.get("width") or "").strip())
            height = int((row.get("height") or "").strip())
        except ValueError:
            width = -1
            height = -1

        row_errors: list[str] = []
        if not sample_id:
            row_errors.append("empty sample_id")
        elif sample_id in ids:
            row_errors.append(f"duplicate sample_id {sample_id}")
        if not image_relative:
            row_errors.append("empty image_path")
            image_absolute = repository_root
        else:
            try:
                image_absolute = resolve_repository_path(repository_root, image_relative)
                image_absolute.relative_to(release_root)
            except (PipelineError, ValueError):
                row_errors.append(f"image path outside release: {image_relative}")
                image_absolute = repository_root
            if image_relative in paths:
                row_errors.append(f"duplicate image_path {image_relative}")
            if not image_absolute.is_file():
                row_errors.append(f"missing image {image_relative}")
        if label not in class_set:
            row_errors.append(f"unexpected class {label!r}")
        if severity not in severity_set:
            row_errors.append(f"unexpected severity {severity!r}")
        if len(image_sha) != 64 or any(ch not in "0123456789abcdef" for ch in image_sha):
            row_errors.append("invalid image_sha256")
        elif require_unique_hashes and image_sha in hashes:
            row_errors.append(f"duplicate image_sha256 {image_sha}")
        if not sample_seed:
            row_errors.append("empty sample_seed")
        if qc_status != required_qc_status:
            row_errors.append(
                f"qc_status is {qc_status!r}, expected {required_qc_status!r}"
            )
        if width != expected_width or height != expected_height:
            row_errors.append(
                f"image dimensions are {width}x{height}, expected "
                f"{expected_width}x{expected_height}"
            )
        if verify_hashes and image_absolute.is_file() and len(image_sha) == 64:
            actual_sha = sha256_file(image_absolute)
            if actual_sha != image_sha:
                row_errors.append(
                    f"image SHA mismatch expected={image_sha} actual={actual_sha}"
                )

        if row_errors:
            errors.extend(f"row {row_index} ({sample_id or 'UNKNOWN'}): {item}" for item in row_errors)
            continue

        ids.add(sample_id)
        paths.add(image_relative)
        hashes.add(image_sha)
        class_counts[label] += 1
        severity_counts[severity] += 1
        class_severity_counts[(label, severity)] += 1
        domain_counts[(row.get("domain") or "").strip()] += 1
        qc_counts[qc_status] += 1
        human_verified_counts[human_verified] += 1
        evaluation_eligible_counts[evaluation_eligible] += 1
        samples.append(
            Sample(
                sample_id=sample_id,
                image_path=image_relative,
                image_absolute_path=image_absolute,
                label=label,
                severity=severity,
                image_sha256=image_sha,
                sample_seed=sample_seed,
                base_group_id=(row.get("base_group_id") or "").strip(),
                source_specimen_group=(row.get("source_specimen_group") or "").strip(),
                domain=(row.get("domain") or "").strip(),
                qc_status=qc_status,
                human_verified=human_verified,
                evaluation_eligible=evaluation_eligible,
                width=width,
                height=height,
            )
        )

    if errors:
        preview = "\n".join(errors[:30])
        suffix = "" if len(errors) <= 30 else f"\n... {len(errors) - 30} more errors"
        raise PipelineError(f"manifest integrity gate failed:\n{preview}{suffix}")

    for class_name in classes:
        count = class_counts[class_name]
        if count != expected_per_class:
            errors.append(
                f"class count mismatch {class_name}: expected {expected_per_class}, got {count}"
            )
        for severity_name in severities:
            expected_severity_count = int(
                expected_severity_per_class[severity_name]
            )
            actual_severity_count = class_severity_counts[
                (class_name, severity_name)
            ]
            if actual_severity_count != expected_severity_count:
                errors.append(
                    f"class×severity count mismatch {class_name}/{severity_name}: "
                    f"expected {expected_severity_count}, got {actual_severity_count}"
                )
    if errors:
        raise PipelineError("manifest class gate failed:\n" + "\n".join(errors))

    base_groups = sorted({sample.base_group_id for sample in samples})
    specimen_groups = sorted({sample.source_specimen_group for sample in samples})
    if len(base_groups) == 1:
        warnings.append(
            "All samples share one synthetic base_group_id; train/test are not "
            "independent-specimen splits."
        )
    if human_verified_counts and set(human_verified_counts) != {"YES"}:
        warnings.append(
            "Some or all manifest rows are not human-verified; AUTO_PASS is generator self-QC."
        )
    if evaluation_eligible_counts and set(evaluation_eligible_counts) != {"YES"}:
        warnings.append(
            "Manifest rows are not marked evaluation-eligible; test metrics are synthetic "
            "same-base sanity metrics only."
        )

    audit = {
        "manifest": manifest_value,
        "manifest_sha256": sha256_file(manifest_path),
        "sample_count": len(samples),
        "class_counts": {name: class_counts[name] for name in classes},
        "severity_counts": {
            severity_name: severity_counts[severity_name]
            for severity_name in severities
        },
        "class_severity_counts": {
            class_name: {
                severity_name: class_severity_counts[
                    (class_name, severity_name)
                ]
                for severity_name in severities
            }
            for class_name in classes
        },
        "domain_counts": dict(sorted(domain_counts.items())),
        "qc_status_counts": dict(sorted(qc_counts.items())),
        "human_verified_counts": dict(sorted(human_verified_counts.items())),
        "evaluation_eligible_counts": dict(
            sorted(evaluation_eligible_counts.items())
        ),
        "unique_image_sha256_count": len(hashes),
        "base_group_ids": base_groups,
        "source_specimen_groups": specimen_groups,
        "image_hashes_verified": verify_hashes,
        "expected_image_size": [expected_width, expected_height],
        "fixed_component_roi_xyxy": roi,
        "fixed_component_roi_is_class_independent": True,
        "warnings": warnings,
    }
    return samples, audit


def deterministic_split(
    samples: Sequence[Sample], config: dict[str, Any]
) -> tuple[list[SplitRecord], dict[str, Any]]:
    classes: list[str] = list(config["classes"])
    severities: list[str] = list(config["severities"])
    split_config = config["split"]
    split_seed = int(split_config["seed"])
    validation_per_class = int(split_config["validation_per_class"])
    validation_seed = int(split_config["validation_seed"])
    severity_quotas: dict[str, dict[str, int]] = split_config[
        "severity_quotas"
    ]
    validation_policy = split_config["validation_severity_policy"]
    parity_policy = validation_policy["extra_by_class_index_parity"]
    base_validation_quota = int(validation_policy["base_per_severity"])
    grouped: dict[tuple[str, str], list[tuple[str, Sample]]] = defaultdict(list)

    for sample in samples:
        payload = (
            f"{split_seed}\0{sample.label}\0{sample.severity}\0"
            f"{sample.sample_id}\0{sample.sample_seed}"
        ).encode("utf-8")
        key = hashlib.sha256(payload).hexdigest()
        grouped[(sample.label, sample.severity)].append((key, sample))

    records: list[SplitRecord] = []
    for class_index, class_name in enumerate(classes):
        parity = "even" if class_index % 2 == 0 else "odd"
        extra_validation_severity = parity_policy[parity]
        validation_quotas = {
            severity_name: base_validation_quota
            + (1 if severity_name == extra_validation_severity else 0)
            for severity_name in severities
        }
        if sum(validation_quotas.values()) != validation_per_class:
            raise PipelineError(
                f"validation quota mismatch for {class_name}: {validation_quotas}"
            )
        for severity_name in severities:
            quota = severity_quotas[severity_name]
            train_quota = int(quota["train"])
            test_quota = int(quota["test"])
            ranked = sorted(
                grouped[(class_name, severity_name)],
                key=lambda item: (item[0], item[1].sample_id),
            )
            expected = train_quota + test_quota
            if len(ranked) != expected:
                raise PipelineError(
                    f"cannot split {class_name}/{severity_name}: "
                    f"expected {expected}, got {len(ranked)}"
                )
            outer_train = ranked[:train_quota]
            validation_key_by_id = {
                sample.sample_id: hashlib.sha256(
                    (
                        f"{validation_seed}\0{class_name}\0{severity_name}\0"
                        f"{sample.sample_id}\0"
                        f"{sample.sample_seed}"
                    ).encode("utf-8")
                ).hexdigest()
                for _key, sample in outer_train
            }
            validation_ranked = sorted(
                outer_train,
                key=lambda item: (
                    validation_key_by_id[item[1].sample_id],
                    item[1].sample_id,
                ),
            )
            validation_ids = {
                sample.sample_id
                for _key, sample in validation_ranked[
                    : validation_quotas[severity_name]
                ]
            }
            for rank, (key, sample) in enumerate(ranked):
                split_name = "train" if rank < train_quota else "test"
                if split_name == "test":
                    model_split = "test"
                    validation_key = ""
                else:
                    model_split = (
                        "validation"
                        if sample.sample_id in validation_ids
                        else "gradient_train"
                    )
                    validation_key = validation_key_by_id[sample.sample_id]
                records.append(
                    SplitRecord(
                        sample=sample,
                        split=split_name,
                        model_split=model_split,
                        class_severity_rank=rank,
                        split_key_sha256=key,
                        validation_key_sha256=validation_key,
                    )
                )

    canonical = "".join(
        f"{record.sample.sample_id},{record.sample.severity},"
        f"{record.split},{record.model_split}\n"
        for record in sorted(records, key=lambda item: item.sample.sample_id)
    )
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    train_records = [record for record in records if record.split == "train"]
    test_records = [record for record in records if record.split == "test"]
    gradient_records = [
        record for record in records if record.model_split == "gradient_train"
    ]
    validation_records = [
        record for record in records if record.model_split == "validation"
    ]
    train_ids = {record.sample.sample_id for record in train_records}
    test_ids = {record.sample.sample_id for record in test_records}
    train_hashes = {record.sample.image_sha256 for record in train_records}
    test_hashes = {record.sample.image_sha256 for record in test_records}
    if train_ids & test_ids:
        raise PipelineError("sample_id leakage between train and test")
    if train_hashes & test_hashes:
        raise PipelineError("exact image SHA leakage between train and test")
    model_split_id_sets = {
        name: {
            record.sample.sample_id
            for record in records
            if record.model_split == name
        }
        for name in ("gradient_train", "validation", "test")
    }
    model_split_hash_sets = {
        name: {
            record.sample.image_sha256
            for record in records
            if record.model_split == name
        }
        for name in ("gradient_train", "validation", "test")
    }
    for left, right in (
        ("gradient_train", "validation"),
        ("gradient_train", "test"),
        ("validation", "test"),
    ):
        if model_split_id_sets[left] & model_split_id_sets[right]:
            raise PipelineError(f"sample_id leakage between {left} and {right}")
        if model_split_hash_sets[left] & model_split_hash_sets[right]:
            raise PipelineError(f"exact image SHA leakage between {left} and {right}")

    train_base_groups = {record.sample.base_group_id for record in train_records}
    test_base_groups = {record.sample.base_group_id for record in test_records}
    train_specimen_groups = {
        record.sample.source_specimen_group for record in train_records
    }
    test_specimen_groups = {
        record.sample.source_specimen_group for record in test_records
    }
    base_overlap = sorted(train_base_groups & test_base_groups)
    specimen_overlap = sorted(train_specimen_groups & test_specimen_groups)

    expected_base_overlap = bool(
        config["evaluation"].get("expected_same_base_overlap", True)
    )
    if expected_base_overlap and not base_overlap:
        raise PipelineError(
            "config expects same-base overlap, but train/test base_group_id sets are disjoint"
        )

    def counts_for(split_name: str, field: str = "split") -> dict[str, int]:
        counter = Counter(
            record.sample.label
            for record in records
            if getattr(record, field) == split_name
        )
        return {name: counter[name] for name in classes}

    def severity_counts_for(
        split_name: str, field: str = "split"
    ) -> dict[str, int]:
        counter = Counter(
            record.sample.severity
            for record in records
            if getattr(record, field) == split_name
        )
        return {name: counter[name] for name in severities}

    def class_severity_counts_for(
        split_name: str, field: str = "split"
    ) -> dict[str, dict[str, int]]:
        counter = Counter(
            (record.sample.label, record.sample.severity)
            for record in records
            if getattr(record, field) == split_name
        )
        return {
            class_name: {
                severity_name: counter[(class_name, severity_name)]
                for severity_name in severities
            }
            for class_name in classes
        }

    outer_class_severity = {
        "train": class_severity_counts_for("train"),
        "test": class_severity_counts_for("test"),
    }
    model_class_severity = {
        "gradient_train": class_severity_counts_for(
            "gradient_train", "model_split"
        ),
        "validation": class_severity_counts_for("validation", "model_split"),
        "test": class_severity_counts_for("test", "model_split"),
    }
    for class_index, class_name in enumerate(classes):
        parity = "even" if class_index % 2 == 0 else "odd"
        extra_validation_severity = parity_policy[parity]
        for severity_name in severities:
            quota = severity_quotas[severity_name]
            if outer_class_severity["train"][class_name][severity_name] != int(
                quota["train"]
            ):
                raise PipelineError(
                    f"outer train severity quota failed for {class_name}/{severity_name}"
                )
            if outer_class_severity["test"][class_name][severity_name] != int(
                quota["test"]
            ):
                raise PipelineError(
                    f"outer test severity quota failed for {class_name}/{severity_name}"
                )
            expected_validation = base_validation_quota + (
                1 if severity_name == extra_validation_severity else 0
            )
            if (
                model_class_severity["validation"][class_name][severity_name]
                != expected_validation
            ):
                raise PipelineError(
                    f"validation severity quota failed for {class_name}/{severity_name}"
                )

    warnings = list(config["evaluation"].get("disclaimers", []))
    if base_overlap:
        warnings.append(
            "base_group_id overlaps between train and test; metrics cannot estimate "
            "independent-specimen or real-domain generalization."
        )
    split_audit = {
        "strategy": split_config["strategy"],
        "seed": split_seed,
        "validation_seed": validation_seed,
        "severity_quotas": severity_quotas,
        "validation_severity_policy": validation_policy,
        "fingerprint_sha256": fingerprint,
        "counts": {
            "train": len(train_records),
            "test": len(test_records),
            "total": len(records),
        },
        "class_counts": {
            "train": counts_for("train"),
            "test": counts_for("test"),
        },
        "severity_counts": {
            "train": severity_counts_for("train"),
            "test": severity_counts_for("test"),
        },
        "class_severity_counts": outer_class_severity,
        "model_counts": {
            "gradient_train": len(gradient_records),
            "validation": len(validation_records),
            "test": len(test_records),
        },
        "model_class_counts": {
            "gradient_train": counts_for("gradient_train", "model_split"),
            "validation": counts_for("validation", "model_split"),
            "test": counts_for("test", "model_split"),
        },
        "model_severity_counts": {
            "gradient_train": severity_counts_for(
                "gradient_train", "model_split"
            ),
            "validation": severity_counts_for("validation", "model_split"),
            "test": severity_counts_for("test", "model_split"),
        },
        "model_class_severity_counts": model_class_severity,
        "sample_id_overlap_count": len(train_ids & test_ids),
        "exact_image_sha256_overlap_count": len(train_hashes & test_hashes),
        "model_split_pairwise_sample_id_overlap_count": 0,
        "model_split_pairwise_exact_image_sha256_overlap_count": 0,
        "base_group_overlap": base_overlap,
        "source_specimen_group_overlap": specimen_overlap,
        "evaluation_scope": config["evaluation"]["scope"],
        "warnings": warnings,
    }
    return records, split_audit


def load_and_validate_auxiliary_condition_manifest(
    manifest_value: Path | str,
    config: dict[str, Any],
    repository_root: Path,
    base_samples: Sequence[Sample],
    base_records: Sequence[SplitRecord],
    base_manifest_audit: dict[str, Any],
    base_split_audit: dict[str, Any],
) -> tuple[list[Sample], dict[str, Any]]:
    """Validate the optional v3 condition release without changing the base split.

    The auxiliary rows are accepted only when they are six condition variants of
    every base ``gradient_train`` parent.  Validation and test parents are therefore
    structurally unable to enter the effective training set through this path.
    """

    manifest_path = resolve_repository_path(repository_root, str(manifest_value))
    if not manifest_path.is_file():
        raise PipelineError(f"auxiliary condition manifest not found: {manifest_path}")
    auxiliary_release_root = manifest_path.parents[1]

    required_columns = {
        "sample_id",
        "image_path",
        "mask_path",
        "domain",
        "split",
        "model_split",
        "training_use",
        "evaluation_eligible",
        "primary_class",
        "visible_multilabel",
        "severity",
        "base_image_id",
        "source_release",
        "parent_sample_id",
        "parent_image_path",
        "parent_mask_path",
        "parent_image_sha256",
        "parent_mask_sha256",
        "parent_sample_seed",
        "base_group_id",
        "source_specimen_group",
        "view",
        "lineage_group_id",
        "family_split_id",
        "defect_instance_id",
        "augmentation_family_id",
        "derivation_depth",
        "variant_index",
        "condition_profile",
        "sample_seed",
        "condition_seed",
        "generator_version",
        "qc_gate_version",
        "config_sha256",
        "image_sha256",
        "mask_sha256",
        "width",
        "height",
        "qc_status",
        "human_verified",
    }
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = set(reader.fieldnames or [])
            missing = sorted(required_columns - fieldnames)
            if missing:
                raise PipelineError(
                    "auxiliary manifest missing required columns: "
                    + ", ".join(missing)
                )
            raw_rows = list(reader)
    except OSError as error:
        raise PipelineError(
            f"cannot read auxiliary condition manifest {manifest_path}: {error}"
        ) from error

    manifest_sha256 = sha256_file(manifest_path)
    release_path = manifest_path.parent / "release.json"
    if not release_path.is_file():
        raise PipelineError(
            f"auxiliary release metadata not found beside manifest: {release_path}"
        )
    try:
        release_metadata = json.loads(release_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PipelineError(
            f"cannot read auxiliary release metadata {release_path}: {error}"
        ) from error
    if not isinstance(release_metadata, dict):
        raise PipelineError("auxiliary release.json must contain an object")
    release_expectations = {
        "release": "synthetic-v3-conditions",
        "source_release": config["release"],
        "source_manifest_sha256": base_manifest_audit["manifest_sha256"],
        "manifest_sha256": manifest_sha256,
        "training_use": "TRAIN_ONLY_CONDITION_SYNTHETIC",
        "evaluation_eligible": "NO",
        "sample_count": 1008,
        "parent_count": 168,
        "variants_per_parent": 6,
    }
    release_errors = [
        f"release.json {name} mismatch: "
        f"{release_metadata.get(name)!r} != {expected!r}"
        for name, expected in release_expectations.items()
        if release_metadata.get(name) != expected
    ]

    auxiliary_config_value = release_metadata.get("config_path")
    if not isinstance(auxiliary_config_value, str) or not auxiliary_config_value:
        release_errors.append("release.json config_path must be non-empty")
        auxiliary_config_path = repository_root
        auxiliary_config: dict[str, Any] = {}
        auxiliary_config_sha256 = ""
    else:
        try:
            auxiliary_config_path = resolve_repository_path(
                repository_root, auxiliary_config_value
            )
        except PipelineError as error:
            release_errors.append(str(error))
            auxiliary_config_path = repository_root
        if not auxiliary_config_path.is_file():
            release_errors.append(
                f"auxiliary config not found: {auxiliary_config_value}"
            )
            auxiliary_config = {}
            auxiliary_config_sha256 = ""
        else:
            auxiliary_config_sha256 = sha256_file(auxiliary_config_path)
            try:
                auxiliary_config = json.loads(
                    auxiliary_config_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as error:
                release_errors.append(
                    f"cannot read auxiliary config {auxiliary_config_path}: {error}"
                )
                auxiliary_config = {}
            if not isinstance(auxiliary_config, dict):
                release_errors.append("auxiliary config must contain an object")
                auxiliary_config = {}
    if release_metadata.get("config_sha256") != auxiliary_config_sha256:
        release_errors.append(
            "release.json config_sha256 does not match the actual auxiliary config"
        )

    expected_generator_version = "3.0.0"
    expected_qc_gate_version = "condition-replay-post-jpeg-512-224-v1"
    expected_profile_order = CONDITION_PROFILE_ORDER
    auxiliary_config_expectations = {
        "release": "synthetic-v3-conditions",
        "generator_version": expected_generator_version,
        "qc_gate_version": expected_qc_gate_version,
        "split": "train",
        "model_split": "gradient_train_auxiliary",
        "training_use": "TRAIN_ONLY_CONDITION_SYNTHETIC",
        "evaluation_eligible": "NO",
        "profiles": list(expected_profile_order),
    }
    for name, expected in auxiliary_config_expectations.items():
        if auxiliary_config.get(name) != expected:
            release_errors.append(
                f"auxiliary config {name} mismatch: "
                f"{auxiliary_config.get(name)!r} != {expected!r}"
            )
    if release_metadata.get("generator_version") != expected_generator_version:
        release_errors.append(
            "release.json generator_version does not match the required version"
        )

    auxiliary_source = auxiliary_config.get("source")
    if not isinstance(auxiliary_source, dict):
        release_errors.append("auxiliary config source must be an object")
        auxiliary_source = {}
    source_expectations = {
        "release": config["release"],
        "manifest_path": config["manifest"],
        "manifest_sha256": base_manifest_audit["manifest_sha256"],
        "required_parent_model_split": "gradient_train",
        "expected_parent_count": 168,
        "expected_parent_count_per_class": 24,
    }
    for name, expected in source_expectations.items():
        if auxiliary_source.get(name) != expected:
            release_errors.append(
                f"auxiliary config source.{name} mismatch: "
                f"{auxiliary_source.get(name)!r} != {expected!r}"
            )

    source_config_value = auxiliary_source.get("config_path")
    source_config_sha = auxiliary_source.get("config_sha256")
    if not isinstance(source_config_value, str) or not source_config_value:
        release_errors.append("auxiliary config source.config_path must be non-empty")
    else:
        try:
            source_config_path = resolve_repository_path(
                repository_root, source_config_value
            )
            actual_source_config_sha = sha256_file(source_config_path)
        except (PipelineError, OSError) as error:
            release_errors.append(f"cannot verify source config: {error}")
        else:
            if source_config_sha != actual_source_config_sha:
                release_errors.append(
                    "auxiliary source config SHA does not match the actual file"
                )
            if release_metadata.get("source_config_sha256") != actual_source_config_sha:
                release_errors.append(
                    "release.json source_config_sha256 does not match the actual file"
                )

    split_assignments_value = auxiliary_source.get("split_assignments_path")
    split_assignments_sha = auxiliary_source.get("split_assignments_sha256")
    if not isinstance(split_assignments_value, str) or not split_assignments_value:
        release_errors.append(
            "auxiliary config source.split_assignments_path must be non-empty"
        )
    else:
        try:
            split_assignments_path = resolve_repository_path(
                repository_root, split_assignments_value
            )
            actual_split_assignments_sha = sha256_file(split_assignments_path)
        except (PipelineError, OSError) as error:
            release_errors.append(f"cannot verify source split assignments: {error}")
        else:
            if split_assignments_sha != actual_split_assignments_sha:
                release_errors.append(
                    "auxiliary source split assignment SHA does not match the actual file"
                )
            if (
                release_metadata.get("source_split_assignments_sha256")
                != actual_split_assignments_sha
            ):
                release_errors.append(
                    "release.json source_split_assignments_sha256 does not match the actual file"
                )
    if release_errors:
        raise PipelineError(
            "auxiliary release metadata gate failed:\n" + "\n".join(release_errors)
        )

    expected_profiles = set(expected_profile_order)
    expected_variants_per_parent = len(expected_profiles)
    expected_parent_count = 168
    expected_parent_count_per_class = 24
    expected_auxiliary_per_class = 144
    expected_total = expected_parent_count * expected_variants_per_parent
    if len(raw_rows) != expected_total:
        raise PipelineError(
            "auxiliary manifest row count mismatch: "
            f"expected {expected_total}, got {len(raw_rows)}"
        )

    gradient_parents = {
        record.sample.sample_id: record.sample
        for record in base_records
        if record.model_split == "gradient_train"
    }
    non_gradient_parent_ids = {
        record.sample.sample_id
        for record in base_records
        if record.model_split != "gradient_train"
    }
    if len(gradient_parents) != expected_parent_count:
        raise PipelineError(
            "base gradient_train parent count changed: "
            f"expected {expected_parent_count}, got {len(gradient_parents)}"
        )
    classes: list[str] = list(config["classes"])
    severities: list[str] = list(config["severities"])
    configured_profiles = auxiliary_config.get("profiles")
    if (
        not isinstance(configured_profiles, list)
        or len(configured_profiles) != len(expected_profiles)
        or set(configured_profiles) != expected_profiles
    ):
        raise PipelineError(
            "auxiliary config profiles must contain the six required profiles exactly once"
        )
    expected_release_class_counts = {
        class_name: expected_auxiliary_per_class for class_name in classes
    }
    expected_release_profile_counts = {
        profile: expected_parent_count for profile in expected_profiles
    }
    expected_release_class_profile_counts = {
        class_name: {
            profile: expected_parent_count_per_class
            for profile in expected_profiles
        }
        for class_name in classes
    }
    for name, expected in (
        ("class_counts", expected_release_class_counts),
        ("profile_counts", expected_release_profile_counts),
        ("class_profile_counts", expected_release_class_profile_counts),
    ):
        if release_metadata.get(name) != expected:
            raise PipelineError(
                f"auxiliary release metadata count mismatch for {name}"
            )
    parent_class_counts = Counter(parent.label for parent in gradient_parents.values())
    for class_name in classes:
        if parent_class_counts[class_name] != expected_parent_count_per_class:
            raise PipelineError(
                f"base gradient_train parent count for {class_name} changed: "
                f"expected {expected_parent_count_per_class}, "
                f"got {parent_class_counts[class_name]}"
            )

    base_manifest_value = config.get("manifest")
    if not isinstance(base_manifest_value, str) or not base_manifest_value:
        raise PipelineError("config manifest must be a non-empty relative path")
    base_manifest_path = resolve_repository_path(repository_root, base_manifest_value)
    try:
        with base_manifest_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as stream:
            base_reader = csv.DictReader(stream)
            base_fieldnames = set(base_reader.fieldnames or [])
            required_base_columns = {
                "sample_id",
                "image_path",
                "mask_path",
                "image_sha256",
                "mask_sha256",
                "base_image_id",
                "view",
            }
            missing_base = sorted(required_base_columns - base_fieldnames)
            if missing_base:
                raise PipelineError(
                    "base manifest lacks auxiliary lineage columns: "
                    + ", ".join(missing_base)
                )
            base_rows = {
                (row.get("sample_id") or "").strip(): row for row in base_reader
            }
    except OSError as error:
        raise PipelineError(f"cannot reread base manifest {base_manifest_path}: {error}") from error

    base_ids = {sample.sample_id for sample in base_samples}
    base_hashes = {sample.image_sha256 for sample in base_samples}
    ids: set[str] = set()
    image_paths: set[str] = set()
    mask_paths: set[str] = set()
    image_hashes: set[str] = set()
    condition_seeds: set[str] = set()
    config_hashes: set[str] = set()
    samples: list[Sample] = []
    errors: list[str] = []
    class_counts: Counter[str] = Counter()
    profile_counts: Counter[str] = Counter()
    class_profile_counts: Counter[tuple[str, str]] = Counter()
    qc_counts: Counter[str] = Counter()
    human_verified_counts: Counter[str] = Counter()
    parent_profiles: defaultdict[str, list[str]] = defaultdict(list)
    parent_variant_indices: defaultdict[str, set[int]] = defaultdict(set)
    parent_index_profiles: defaultdict[str, dict[int, str]] = defaultdict(dict)
    parent_families: defaultdict[str, set[str]] = defaultdict(set)
    family_to_parents: defaultdict[str, set[str]] = defaultdict(set)
    verified_hashes: dict[Path, str] = {}

    def valid_sha256(value: str) -> bool:
        return len(value) == 64 and all(
            character in "0123456789abcdef" for character in value
        )

    def cached_sha256(path: Path) -> str:
        if path not in verified_hashes:
            verified_hashes[path] = sha256_file(path)
        return verified_hashes[path]

    for row_index, row in enumerate(raw_rows, start=2):
        value = lambda name: (row.get(name) or "").strip()  # noqa: E731
        sample_id = value("sample_id")
        image_relative = value("image_path")
        mask_relative = value("mask_path")
        label = value("primary_class")
        severity = value("severity")
        profile = value("condition_profile")
        parent_id = value("parent_sample_id")
        image_sha = value("image_sha256").lower()
        mask_sha = value("mask_sha256").lower()
        parent_image_sha = value("parent_image_sha256").lower()
        parent_mask_sha = value("parent_mask_sha256").lower()
        condition_seed = value("condition_seed")
        config_sha = value("config_sha256").lower()
        family_id = value("augmentation_family_id")
        row_errors: list[str] = []

        if not sample_id:
            row_errors.append("empty sample_id")
        elif sample_id in ids:
            row_errors.append(f"duplicate sample_id {sample_id}")
        elif sample_id in base_ids:
            row_errors.append(f"sample_id overlaps base manifest: {sample_id}")

        image_absolute = repository_root
        if not image_relative:
            row_errors.append("empty image_path")
        else:
            try:
                image_absolute = resolve_repository_path(
                    repository_root, image_relative
                )
                image_absolute.relative_to(auxiliary_release_root)
            except (PipelineError, ValueError):
                row_errors.append(
                    f"image path outside auxiliary release: {image_relative}"
                )
                image_absolute = repository_root
            if image_relative in image_paths:
                row_errors.append(f"duplicate image_path {image_relative}")
            if not image_absolute.is_file():
                row_errors.append(f"missing image {image_relative}")

        mask_absolute = repository_root
        if not mask_relative:
            row_errors.append("empty mask_path")
        else:
            try:
                mask_absolute = resolve_repository_path(repository_root, mask_relative)
                mask_absolute.relative_to(auxiliary_release_root)
            except (PipelineError, ValueError):
                row_errors.append(
                    f"mask path outside auxiliary release: {mask_relative}"
                )
                mask_absolute = repository_root
            if mask_relative in mask_paths:
                row_errors.append(f"duplicate mask_path {mask_relative}")
            if not mask_absolute.is_file():
                row_errors.append(f"missing mask {mask_relative}")

        if label not in classes:
            row_errors.append(f"unexpected class {label!r}")
        if value("visible_multilabel") != label:
            row_errors.append("visible_multilabel must equal primary_class")
        if severity not in severities:
            row_errors.append(f"unexpected severity {severity!r}")
        if profile not in expected_profiles:
            row_errors.append(f"unexpected condition_profile {profile!r}")
        if value("domain") != "synthetic_conditioned_from_v2":
            row_errors.append("domain must be synthetic_conditioned_from_v2")
        if value("split") != "train":
            row_errors.append("split must be train")
        if value("model_split") != "gradient_train_auxiliary":
            row_errors.append("model_split must be gradient_train_auxiliary")
        if value("training_use") != "TRAIN_ONLY_CONDITION_SYNTHETIC":
            row_errors.append(
                "training_use must be TRAIN_ONLY_CONDITION_SYNTHETIC"
            )
        if value("evaluation_eligible") != "NO":
            row_errors.append("evaluation_eligible must be NO")
        if value("source_release") != str(config["release"]):
            row_errors.append(
                f"source_release must be {config['release']}"
            )
        if value("qc_status") != "AUTO_PASS_CONDITION_POST_JPEG_512_224":
            row_errors.append(
                "qc_status must be AUTO_PASS_CONDITION_POST_JPEG_512_224"
            )
        if value("human_verified") not in {"YES", "NO"}:
            row_errors.append("human_verified must be YES or NO")
        if value("generator_version") != expected_generator_version:
            row_errors.append(
                f"generator_version must be {expected_generator_version}"
            )
        if value("qc_gate_version") != expected_qc_gate_version:
            row_errors.append(
                f"qc_gate_version must be {expected_qc_gate_version}"
            )

        try:
            width = int(value("width"))
            height = int(value("height"))
        except ValueError:
            width = -1
            height = -1
        if (width, height) != (512, 512):
            row_errors.append(
                f"image dimensions are {width}x{height}, expected 512x512"
            )
        try:
            derivation_depth = int(value("derivation_depth"))
        except ValueError:
            derivation_depth = -1
        if derivation_depth != 1:
            row_errors.append("derivation_depth must be 1")
        try:
            variant_index = int(value("variant_index"))
        except ValueError:
            variant_index = -1
        if not 0 <= variant_index < expected_variants_per_parent:
            row_errors.append(
                f"variant_index must be 0..{expected_variants_per_parent - 1}"
            )

        for name, sha_value in (
            ("image_sha256", image_sha),
            ("mask_sha256", mask_sha),
            ("parent_image_sha256", parent_image_sha),
            ("parent_mask_sha256", parent_mask_sha),
            ("config_sha256", config_sha),
        ):
            if not valid_sha256(sha_value):
                row_errors.append(f"invalid {name}")
        if valid_sha256(image_sha):
            if image_sha in image_hashes:
                row_errors.append(f"duplicate image_sha256 {image_sha}")
            if image_sha in base_hashes:
                row_errors.append(f"image SHA overlaps base manifest: {image_sha}")
        if not condition_seed:
            row_errors.append("empty condition_seed")
        elif condition_seed in condition_seeds:
            row_errors.append(f"duplicate condition_seed {condition_seed}")
        if not value("sample_seed"):
            row_errors.append("empty sample_seed")
        elif value("sample_seed") != condition_seed:
            row_errors.append("sample_seed must equal condition_seed")
        if not family_id:
            row_errors.append("empty augmentation_family_id")

        parent = gradient_parents.get(parent_id)
        if parent is None:
            if parent_id in non_gradient_parent_ids:
                row_errors.append(
                    f"parent belongs to validation/test, not gradient_train: {parent_id}"
                )
            else:
                row_errors.append(f"unknown parent_sample_id {parent_id!r}")
        else:
            base_row = base_rows.get(parent_id)
            if base_row is None:
                row_errors.append(f"parent absent from base manifest: {parent_id}")
            else:
                expected_parent_mask_path = (
                    base_row.get("mask_path") or ""
                ).strip()
                expected_parent_mask_sha = (
                    base_row.get("mask_sha256") or ""
                ).strip().lower()
                expected_parent_mask_absolute = repository_root
                try:
                    expected_parent_mask_absolute = resolve_repository_path(
                        repository_root, expected_parent_mask_path
                    )
                    expected_parent_mask_absolute.relative_to(
                        base_manifest_path.parents[1]
                    )
                except (PipelineError, ValueError):
                    row_errors.append(
                        "parent mask path escapes the base release"
                    )
                    expected_parent_mask_absolute = repository_root
                if not expected_parent_mask_absolute.is_file():
                    row_errors.append(
                        f"missing parent mask {expected_parent_mask_path}"
                    )
                elif valid_sha256(expected_parent_mask_sha):
                    actual_parent_mask_sha = cached_sha256(
                        expected_parent_mask_absolute
                    )
                    if actual_parent_mask_sha != expected_parent_mask_sha:
                        row_errors.append(
                            "parent mask SHA mismatch "
                            f"expected={expected_parent_mask_sha} "
                            f"actual={actual_parent_mask_sha}"
                        )
                lineage_checks = {
                    "primary_class": (label, parent.label),
                    "severity": (severity, parent.severity),
                    "parent_image_path": (
                        value("parent_image_path"),
                        parent.image_path,
                    ),
                    "parent_image_sha256": (
                        parent_image_sha,
                        parent.image_sha256,
                    ),
                    "parent_mask_path": (
                        value("parent_mask_path"),
                        expected_parent_mask_path,
                    ),
                    "parent_mask_sha256": (
                        parent_mask_sha,
                        expected_parent_mask_sha,
                    ),
                    "parent_sample_seed": (
                        value("parent_sample_seed"),
                        parent.sample_seed,
                    ),
                    "base_group_id": (
                        value("base_group_id"),
                        parent.base_group_id,
                    ),
                    "source_specimen_group": (
                        value("source_specimen_group"),
                        parent.source_specimen_group,
                    ),
                    "base_image_id": (
                        value("base_image_id"),
                        (base_row.get("base_image_id") or "").strip(),
                    ),
                    "view": (
                        value("view"),
                        (base_row.get("view") or "").strip(),
                    ),
                }
                for name, (actual, expected) in lineage_checks.items():
                    if actual != expected:
                        row_errors.append(
                            f"{name} does not match parent: {actual!r} != {expected!r}"
                        )
                for name in (
                    "lineage_group_id",
                    "family_split_id",
                    "defect_instance_id",
                ):
                    if value(name) != parent_id:
                        row_errors.append(f"{name} must equal parent_sample_id")
                if valid_sha256(mask_sha) and mask_sha != expected_parent_mask_sha:
                    row_errors.append("condition mask SHA must equal parent mask SHA")

        if image_absolute.is_file() and valid_sha256(image_sha):
            actual_image_sha = cached_sha256(image_absolute)
            if actual_image_sha != image_sha:
                row_errors.append(
                    "image SHA mismatch "
                    f"expected={image_sha} actual={actual_image_sha}"
                )
        if mask_absolute.is_file() and valid_sha256(mask_sha):
            actual_mask_sha = cached_sha256(mask_absolute)
            if actual_mask_sha != mask_sha:
                row_errors.append(
                    "mask SHA mismatch "
                    f"expected={mask_sha} actual={actual_mask_sha}"
                )

        if row_errors:
            errors.extend(
                f"row {row_index} ({sample_id or 'UNKNOWN'}): {item}"
                for item in row_errors
            )
            continue

        ids.add(sample_id)
        image_paths.add(image_relative)
        mask_paths.add(mask_relative)
        image_hashes.add(image_sha)
        condition_seeds.add(condition_seed)
        config_hashes.add(config_sha)
        class_counts[label] += 1
        profile_counts[profile] += 1
        class_profile_counts[(label, profile)] += 1
        qc_counts[value("qc_status")] += 1
        human_verified_counts[value("human_verified")] += 1
        parent_profiles[parent_id].append(profile)
        parent_variant_indices[parent_id].add(variant_index)
        parent_index_profiles[parent_id][variant_index] = profile
        parent_families[parent_id].add(family_id)
        family_to_parents[family_id].add(parent_id)
        samples.append(
            Sample(
                sample_id=sample_id,
                image_path=image_relative,
                image_absolute_path=image_absolute,
                label=label,
                severity=severity,
                image_sha256=image_sha,
                sample_seed=value("sample_seed"),
                base_group_id=value("base_group_id"),
                source_specimen_group=value("source_specimen_group"),
                domain=value("domain"),
                qc_status=value("qc_status"),
                human_verified=value("human_verified"),
                evaluation_eligible=value("evaluation_eligible"),
                width=width,
                height=height,
                parent_sample_id=parent_id,
                family_split_id=value("family_split_id"),
                augmentation_family_id=family_id,
                condition_profile=profile,
                variant_index=variant_index,
                derivation_depth=derivation_depth,
            )
        )

    if errors:
        preview = "\n".join(errors[:30])
        suffix = "" if len(errors) <= 30 else f"\n... {len(errors) - 30} more errors"
        raise PipelineError(
            f"auxiliary manifest integrity gate failed:\n{preview}{suffix}"
        )

    expected_parent_ids = set(gradient_parents)
    actual_parent_ids = set(parent_profiles)
    if actual_parent_ids != expected_parent_ids:
        missing = sorted(expected_parent_ids - actual_parent_ids)
        unexpected = sorted(actual_parent_ids - expected_parent_ids)
        errors.append(
            "auxiliary parent set does not exactly match base gradient_train; "
            f"missing={missing[:10]} unexpected={unexpected[:10]}"
        )
    for parent_id in sorted(expected_parent_ids):
        profiles = parent_profiles[parent_id]
        if len(profiles) != expected_variants_per_parent or set(profiles) != expected_profiles:
            errors.append(
                f"parent {parent_id} must have each of the six profiles exactly once"
            )
        if parent_variant_indices[parent_id] != set(
            range(expected_variants_per_parent)
        ):
            errors.append(
                f"parent {parent_id} variant_index set must be 0..5"
            )
        for variant_index, expected_profile in enumerate(expected_profile_order):
            actual_profile = parent_index_profiles[parent_id].get(variant_index)
            if actual_profile != expected_profile:
                errors.append(
                    f"parent {parent_id} variant_index {variant_index} must map to "
                    f"{expected_profile}, got {actual_profile!r}"
                )
        if len(parent_families[parent_id]) != 1:
            errors.append(
                f"parent {parent_id} must map to exactly one augmentation family"
            )
    reused_families = {
        family: parents
        for family, parents in family_to_parents.items()
        if len(parents) != 1
    }
    if reused_families:
        errors.append(
            "augmentation_family_id reused across parents: "
            + ", ".join(sorted(reused_families)[:10])
        )
    for class_name in classes:
        if class_counts[class_name] != expected_auxiliary_per_class:
            errors.append(
                f"auxiliary class count mismatch {class_name}: "
                f"expected {expected_auxiliary_per_class}, got {class_counts[class_name]}"
            )
        for profile in sorted(expected_profiles):
            if class_profile_counts[(class_name, profile)] != expected_parent_count_per_class:
                errors.append(
                    f"auxiliary class×profile count mismatch {class_name}/{profile}: "
                    f"expected {expected_parent_count_per_class}, "
                    f"got {class_profile_counts[(class_name, profile)]}"
                )
    for profile in expected_profiles:
        if profile_counts[profile] != expected_parent_count:
            errors.append(
                f"auxiliary profile count mismatch {profile}: "
                f"expected {expected_parent_count}, got {profile_counts[profile]}"
            )
    if config_hashes != {auxiliary_config_sha256}:
        errors.append(
            "auxiliary row config_sha256 values must equal the release-pinned "
            f"config SHA {auxiliary_config_sha256}"
        )
    if errors:
        raise PipelineError("auxiliary manifest count/lineage gate failed:\n" + "\n".join(errors))

    canonical_lineage = "".join(
        f"{value_row['sample_id']},{value_row['parent_sample_id']},"
        f"{value_row['condition_profile']},{value_row['image_sha256']}\n"
        for value_row in sorted(raw_rows, key=lambda item: item["sample_id"])
    )
    manifest_relative = manifest_path.relative_to(repository_root).as_posix()
    audit = {
        "manifest": manifest_relative,
        "manifest_sha256": manifest_sha256,
        "release_metadata": release_path.relative_to(repository_root).as_posix(),
        "release_metadata_sha256": sha256_file(release_path),
        "source_release": config["release"],
        "source_manifest_sha256": base_manifest_audit["manifest_sha256"],
        "source_split_fingerprint_sha256": base_split_audit["fingerprint_sha256"],
        "required_parent_model_split": "gradient_train",
        "sample_count": len(samples),
        "parent_count": len(parent_profiles),
        "augmentation_family_count": len(family_to_parents),
        "variants_per_parent": expected_variants_per_parent,
        "effective_gradient_train_count": len(gradient_parents) + len(samples),
        "class_counts": {name: class_counts[name] for name in classes},
        "parent_class_counts": {
            name: parent_class_counts[name] for name in classes
        },
        "profile_counts": {
            name: profile_counts[name] for name in sorted(expected_profiles)
        },
        "class_profile_counts": {
            class_name: {
                profile: class_profile_counts[(class_name, profile)]
                for profile in sorted(expected_profiles)
            }
            for class_name in classes
        },
        "qc_status_counts": dict(sorted(qc_counts.items())),
        "human_verified_counts": dict(sorted(human_verified_counts.items())),
        "unique_image_sha256_count": len(image_hashes),
        "image_and_mask_hashes_verified": True,
        "parent_image_and_mask_lineage_verified": True,
        "sample_id_overlap_with_base_count": len(ids & base_ids),
        "image_sha256_overlap_with_base_count": len(image_hashes & base_hashes),
        "validation_or_test_parent_count": len(actual_parent_ids & non_gradient_parent_ids),
        "config": auxiliary_config_path.relative_to(repository_root).as_posix(),
        "config_sha256": auxiliary_config_sha256,
        "generator_version": expected_generator_version,
        "qc_gate_version": expected_qc_gate_version,
        "lineage_fingerprint_sha256": hashlib.sha256(
            canonical_lineage.encode("utf-8")
        ).hexdigest(),
        "evaluation_eligible": "NO",
        "training_use": "TRAIN_ONLY_CONDITION_SYNTHETIC",
        "warnings": [
            "Auxiliary condition variants add no independent physical specimen or defect morphology.",
            "The auxiliary release is gradient-train-only and is excluded from validation/test metrics.",
        ],
    }
    return samples, audit


def build_family_balanced_sampling_plan(
    base_train_samples: Sequence[Sample],
    auxiliary_samples: Sequence[Sample],
    classes: Sequence[str],
    validation_test_samples: Sequence[Sample],
    epoch_count: int,
    batch_size: int,
    sampling_seed: int,
    optimizer_update_budget: int,
) -> tuple[list[list[Sample]], dict[str, Any]]:
    """Build and audit deterministic one-draw-per-parent epoch plans.

    Every base training parent contributes exactly one image per epoch: either the
    base image or one of its six v3 condition variants. The seven profile choices
    rotate across parents and classes so each epoch has exactly 24 draws per
    profile and 24 draws per class. A seven-epoch cycle exposes every parent to
    every profile exactly once without treating derivatives as new specimens.

    This function intentionally depends only on the standard library so the full
    family/leakage/update-budget gate also runs under ``--check-only``.
    """

    if epoch_count <= 0:
        raise PipelineError("family-balanced epoch_count must be positive")
    if batch_size <= 0:
        raise PipelineError("family-balanced batch_size must be positive")
    if optimizer_update_budget <= 0:
        raise PipelineError(
            "family-balanced optimizer_update_budget must be positive"
        )
    expected_classes = list(classes)
    if len(expected_classes) != 7 or len(set(expected_classes)) != 7:
        raise PipelineError(
            "family-balanced sampling requires the configured seven unique classes"
        )

    base_by_id: dict[str, Sample] = {}
    for sample in base_train_samples:
        if sample.sample_id in base_by_id:
            raise PipelineError(
                f"duplicate base gradient-train parent: {sample.sample_id}"
            )
        base_by_id[sample.sample_id] = sample
    if len(base_by_id) != 168:
        raise PipelineError(
            "family-balanced sampling requires exactly 168 base gradient-train "
            f"parents, got {len(base_by_id)}"
        )

    evaluation_ids = {sample.sample_id for sample in validation_test_samples}
    evaluation_hashes = {
        sample.image_sha256 for sample in validation_test_samples
    }
    base_parent_overlap = set(base_by_id) & evaluation_ids
    if base_parent_overlap:
        raise PipelineError(
            "family-balanced base parents overlap validation/test: "
            + ", ".join(sorted(base_parent_overlap)[:10])
        )

    base_class_counts = Counter(sample.label for sample in base_by_id.values())
    unexpected_base_classes = sorted(set(base_class_counts) - set(expected_classes))
    if unexpected_base_classes:
        raise PipelineError(
            "family-balanced base parents contain unexpected classes: "
            + ", ".join(unexpected_base_classes)
        )
    for class_name in expected_classes:
        if base_class_counts[class_name] != 24:
            raise PipelineError(
                f"family-balanced parent count for {class_name} must be 24, "
                f"got {base_class_counts[class_name]}"
            )

    variants_by_parent: defaultdict[str, dict[str, Sample]] = defaultdict(dict)
    family_ids_by_parent: defaultdict[str, set[str]] = defaultdict(set)
    auxiliary_ids: set[str] = set()
    auxiliary_hashes: set[str] = set()
    expected_variant_index = {
        profile: index for index, profile in enumerate(CONDITION_PROFILE_ORDER)
    }
    for sample in auxiliary_samples:
        if sample.sample_id in auxiliary_ids:
            raise PipelineError(
                f"duplicate auxiliary sample in family plan: {sample.sample_id}"
            )
        auxiliary_ids.add(sample.sample_id)
        auxiliary_hashes.add(sample.image_sha256)
        parent_id = sample.parent_sample_id
        parent = base_by_id.get(parent_id)
        if parent is None:
            if parent_id in evaluation_ids:
                raise PipelineError(
                    "family-balanced auxiliary parent belongs to validation/test: "
                    f"{parent_id}"
                )
            raise PipelineError(
                f"family-balanced auxiliary sample has unknown parent: {parent_id!r}"
            )
        if sample.family_split_id != parent_id:
            raise PipelineError(
                f"family_split_id must equal parent for {sample.sample_id}"
            )
        if sample.condition_profile not in CONDITION_PROFILE_ORDER:
            raise PipelineError(
                f"unexpected family condition profile for {sample.sample_id}: "
                f"{sample.condition_profile!r}"
            )
        if sample.condition_profile in variants_by_parent[parent_id]:
            raise PipelineError(
                f"duplicate parent/profile candidate: {parent_id}/"
                f"{sample.condition_profile}"
            )
        if sample.variant_index != expected_variant_index[sample.condition_profile]:
            raise PipelineError(
                f"variant index/profile mismatch for {sample.sample_id}"
            )
        if sample.derivation_depth != 1:
            raise PipelineError(
                f"auxiliary derivation depth must be 1 for {sample.sample_id}"
            )
        lineage_values = (
            ("label", sample.label, parent.label),
            ("severity", sample.severity, parent.severity),
            ("base_group_id", sample.base_group_id, parent.base_group_id),
            (
                "source_specimen_group",
                sample.source_specimen_group,
                parent.source_specimen_group,
            ),
        )
        for name, actual, expected in lineage_values:
            if actual != expected:
                raise PipelineError(
                    f"family-balanced {name} lineage mismatch for "
                    f"{sample.sample_id}: {actual!r} != {expected!r}"
                )
        variants_by_parent[parent_id][sample.condition_profile] = sample
        family_ids_by_parent[parent_id].add(sample.augmentation_family_id)

    if len(auxiliary_samples) != 1008:
        raise PipelineError(
            "family-balanced sampling requires exactly 1,008 auxiliary variants, "
            f"got {len(auxiliary_samples)}"
        )
    if auxiliary_ids & evaluation_ids:
        raise PipelineError(
            "family-balanced auxiliary sample IDs overlap validation/test"
        )
    if auxiliary_hashes & evaluation_hashes:
        raise PipelineError(
            "family-balanced auxiliary image hashes overlap validation/test"
        )

    candidates_by_parent: dict[str, dict[str, Sample]] = {}
    augmentation_family_ids: set[str] = set()
    expected_profiles = set(CONDITION_PROFILE_ORDER)
    for parent_id, parent in sorted(base_by_id.items()):
        variants = variants_by_parent.get(parent_id, {})
        if set(variants) != expected_profiles:
            missing = sorted(expected_profiles - set(variants))
            unexpected = sorted(set(variants) - expected_profiles)
            raise PipelineError(
                f"parent {parent_id} does not have the exact six variants; "
                f"missing={missing} unexpected={unexpected}"
            )
        family_ids = family_ids_by_parent[parent_id]
        if len(family_ids) != 1 or "" in family_ids:
            raise PipelineError(
                f"parent {parent_id} must have one non-empty augmentation family"
            )
        family_id = next(iter(family_ids))
        if family_id in augmentation_family_ids:
            raise PipelineError(
                f"augmentation family reused across parents: {family_id}"
            )
        augmentation_family_ids.add(family_id)
        candidates_by_parent[parent_id] = {"base": parent, **variants}

    ordered_parents_by_class: dict[str, list[Sample]] = {}
    for class_name in expected_classes:
        class_parents = [
            sample
            for sample in base_by_id.values()
            if sample.label == class_name
        ]
        ordered_parents_by_class[class_name] = sorted(
            class_parents,
            key=lambda sample: hashlib.sha256(
                (
                    f"{sampling_seed}\0family-balanced\0{class_name}\0"
                    f"{sample.sample_id}"
                ).encode("utf-8")
            ).hexdigest(),
        )

    profile_order = list(FAMILY_BALANCED_PROFILE_ORDER)
    updates_per_epoch = math.ceil(len(base_by_id) / batch_size)
    planned_optimizer_updates = epoch_count * updates_per_epoch
    if optimizer_update_budget != planned_optimizer_updates:
        raise PipelineError(
            "family-balanced optimizer update budget must equal "
            "epoch_count * ceil(family_count / batch_size): "
            f"expected {planned_optimizer_updates}, got {optimizer_update_budget}"
        )

    epoch_samples: list[list[Sample]] = []
    per_epoch: list[dict[str, Any]] = []
    plan_lines: list[str] = []
    total_profile_counts: Counter[str] = Counter()
    total_class_counts: Counter[str] = Counter()
    total_class_profile_counts: Counter[tuple[str, str]] = Counter()
    for epoch_index in range(epoch_count):
        selected: list[Sample] = []
        parent_ids: set[str] = set()
        profile_counts: Counter[str] = Counter()
        class_counts: Counter[str] = Counter()
        class_profile_counts: Counter[tuple[str, str]] = Counter()
        for class_index, class_name in enumerate(expected_classes):
            parents = ordered_parents_by_class[class_name]
            for position, parent in enumerate(parents):
                # Each class has 24 parents: three complete seven-profile blocks
                # plus three draws. Offsetting the seven classes makes those
                # three extras cancel globally, yielding 24 draws/profile/epoch.
                profile = profile_order[
                    (position + class_index + epoch_index) % len(profile_order)
                ]
                sample = candidates_by_parent[parent.sample_id][profile]
                if parent.sample_id in parent_ids:
                    raise PipelineError(
                        f"parent drawn more than once in epoch {epoch_index + 1}: "
                        f"{parent.sample_id}"
                    )
                parent_ids.add(parent.sample_id)
                selected.append(sample)
                profile_counts[profile] += 1
                class_counts[class_name] += 1
                class_profile_counts[(class_name, profile)] += 1
                plan_lines.append(
                    f"{epoch_index + 1}\0{parent.sample_id}\0{sample.sample_id}\0"
                    f"{class_name}\0{profile}\n"
                )
        if parent_ids != set(base_by_id):
            raise PipelineError(
                f"epoch {epoch_index + 1} does not draw every parent exactly once"
            )
        if len(selected) != 168:
            raise PipelineError(
                f"epoch {epoch_index + 1} draw count must be 168, got {len(selected)}"
            )
        expected_per_profile = len(selected) // len(profile_order)
        if any(profile_counts[name] != expected_per_profile for name in profile_order):
            raise PipelineError(
                f"epoch {epoch_index + 1} profile draws are not balanced: "
                f"{dict(profile_counts)}"
            )
        if any(class_counts[name] != 24 for name in expected_classes):
            raise PipelineError(
                f"epoch {epoch_index + 1} class draws are not balanced: "
                f"{dict(class_counts)}"
            )
        epoch_samples.append(selected)
        total_profile_counts.update(profile_counts)
        total_class_counts.update(class_counts)
        total_class_profile_counts.update(class_profile_counts)
        per_epoch.append(
            {
                "epoch": epoch_index + 1,
                "draw_count": len(selected),
                "unique_parent_count": len(parent_ids),
                "optimizer_update_count": updates_per_epoch,
                "class_counts": {
                    name: class_counts[name] for name in expected_classes
                },
                "profile_counts": {
                    name: profile_counts[name] for name in profile_order
                },
                "class_profile_counts": {
                    class_name: {
                        profile: class_profile_counts[(class_name, profile)]
                        for profile in profile_order
                    }
                    for class_name in expected_classes
                },
            }
        )

    # Validate the rotation contract independently of the requested run length.
    rotation_failures: list[str] = []
    for class_index, class_name in enumerate(expected_classes):
        for position, parent in enumerate(ordered_parents_by_class[class_name]):
            cycle_profiles = {
                profile_order[
                    (position + class_index + cycle_epoch) % len(profile_order)
                ]
                for cycle_epoch in range(len(profile_order))
            }
            if cycle_profiles != set(profile_order):
                rotation_failures.append(parent.sample_id)
    if rotation_failures:
        raise PipelineError(
            "seven-epoch family rotation failed for parents: "
            + ", ".join(rotation_failures[:10])
        )

    audit = {
        "schema_version": "1.0",
        "status": "PASS",
        "mode": "family_balanced_parent_variant",
        "sampling_seed": sampling_seed,
        "family_key": "base sample_id == auxiliary parent_sample_id == family_split_id",
        "family_count": len(base_by_id),
        "family_counts_per_class": {
            name: base_class_counts[name] for name in expected_classes
        },
        "augmentation_family_count": len(augmentation_family_ids),
        "base_parent_count": len(base_by_id),
        "auxiliary_variant_count": len(auxiliary_samples),
        "candidate_pool_sample_count": len(base_by_id) + len(auxiliary_samples),
        "candidates_per_family": len(profile_order),
        "profile_order": profile_order,
        "samples_per_epoch": len(base_by_id),
        "batch_size": batch_size,
        "optimizer_updates_per_epoch": updates_per_epoch,
        "epoch_count": epoch_count,
        "optimizer_update_budget": optimizer_update_budget,
        "planned_optimizer_update_count": planned_optimizer_updates,
        "comparison_update_reference": {
            "c0_base_samples_per_epoch": len(base_by_id),
            "c0_optimizer_updates_per_epoch": updates_per_epoch,
            "c0_full_schedule_optimizer_updates": planned_optimizer_updates,
            "c2_append_samples_per_epoch": len(base_by_id) + len(auxiliary_samples),
            "c2_append_optimizer_updates_per_epoch": math.ceil(
                (len(base_by_id) + len(auxiliary_samples)) / batch_size
            ),
            "c2_append_full_schedule_optimizer_updates": epoch_count
            * math.ceil(
                (len(base_by_id) + len(auxiliary_samples)) / batch_size
            ),
            "note": (
                "C3 fixes the C0-sized update budget; C2 append has a larger "
                "budget and must remain identified as a separate ablation."
            ),
        },
        "planned_draw_count": epoch_count * len(base_by_id),
        "planned_profile_counts": {
            name: total_profile_counts[name] for name in profile_order
        },
        "planned_class_counts": {
            name: total_class_counts[name] for name in expected_classes
        },
        "planned_class_profile_counts": {
            class_name: {
                profile: total_class_profile_counts[(class_name, profile)]
                for profile in profile_order
            }
            for class_name in expected_classes
        },
        "per_epoch": per_epoch,
        "leakage_gate": {
            "status": "PASS",
            "scope": (
                "exact auxiliary parent/sample/image-hash exclusion from the "
                "immutable v2 validation and test partitions"
            ),
            "base_parent_overlap_with_validation_test_count": len(
                set(base_by_id) & evaluation_ids
            ),
            "auxiliary_parent_overlap_with_validation_test_count": len(
                set(variants_by_parent) & evaluation_ids
            ),
            "auxiliary_sample_id_overlap_with_validation_test_count": len(
                auxiliary_ids & evaluation_ids
            ),
            "auxiliary_image_sha256_overlap_with_validation_test_count": len(
                auxiliary_hashes & evaluation_hashes
            ),
            "validation_and_test_are_base_v2_only": True,
            "does_not_claim_independent_specimen_or_base_group_split": True,
        },
        "rotation_gate": {
            "status": "PASS",
            "cycle_length_epochs": len(profile_order),
            "every_parent_draws_every_profile_once_per_cycle": True,
        },
        "sampling_plan_fingerprint_sha256": hashlib.sha256(
            "".join(plan_lines).encode("utf-8")
        ).hexdigest(),
        "warnings": [
            "The six condition variants are derivatives of each base parent, not independent specimens.",
            "Validation and test remain the immutable synthetic-v2 base split; these metrics remain same-base synthetic sanity checks.",
        ],
    }
    return epoch_samples, audit


def write_split_artifacts(
    output_directory: Path,
    records: Sequence[SplitRecord],
    manifest_audit: dict[str, Any],
    split_audit: dict[str, Any],
    auxiliary_manifest_audit: dict[str, Any] | None = None,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    assignment_path = output_directory / "split_assignments.csv"
    with assignment_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "sample_id",
                "image_path",
                "primary_class",
                "severity",
                "split",
                "model_split",
                "class_severity_rank",
                "split_key_sha256",
                "validation_key_sha256",
                "sample_seed",
                "image_sha256",
                "base_group_id",
                "source_specimen_group",
            ]
        )
        for record in sorted(
            records,
            key=lambda item: (
                item.split,
                item.sample.label,
                item.sample.severity,
                item.class_severity_rank,
                item.sample.sample_id,
            ),
        ):
            writer.writerow(
                [
                    record.sample.sample_id,
                    record.sample.image_path,
                    record.sample.label,
                    record.sample.severity,
                    record.split,
                    record.model_split,
                    record.class_severity_rank,
                    record.split_key_sha256,
                    record.validation_key_sha256,
                    record.sample.sample_seed,
                    record.sample.image_sha256,
                    record.sample.base_group_id,
                    record.sample.source_specimen_group,
                ]
            )
    write_json(output_directory / "manifest_audit.json", manifest_audit)
    write_json(output_directory / "split_audit.json", split_audit)
    if auxiliary_manifest_audit is not None:
        write_json(
            output_directory / "auxiliary_manifest_audit.json",
            auxiliary_manifest_audit,
        )


def split_samples(
    records: Sequence[SplitRecord], split_name: str
) -> list[Sample]:
    return [record.sample for record in records if record.model_split == split_name]


def confusion_matrix(
    true_indices: Sequence[int], predicted_indices: Sequence[int], class_count: int
) -> list[list[int]]:
    if len(true_indices) != len(predicted_indices):
        raise PipelineError("prediction and target lengths differ")
    matrix = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for true_index, predicted_index in zip(true_indices, predicted_indices):
        if not 0 <= true_index < class_count:
            raise PipelineError(f"target index out of range: {true_index}")
        if not 0 <= predicted_index < class_count:
            raise PipelineError(f"prediction index out of range: {predicted_index}")
        matrix[true_index][predicted_index] += 1
    return matrix


def calculate_metrics(
    matrix: Sequence[Sequence[int]], classes: Sequence[str]
) -> dict[str, Any]:
    if len(matrix) != len(classes) or any(len(row) != len(classes) for row in matrix):
        raise PipelineError("confusion matrix dimensions do not match classes")
    total = sum(sum(row) for row in matrix)
    per_class: list[dict[str, Any]] = []
    for index, class_name in enumerate(classes):
        true_positive = int(matrix[index][index])
        support = int(sum(matrix[index]))
        predicted_count = int(sum(row[index] for row in matrix))
        false_positive = predicted_count - true_positive
        false_negative = support - true_positive
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator if precision_denominator else 0.0
        )
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "class": class_name,
                "class_index": index,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
            }
        )

    macro = {
        metric: sum(row[metric] for row in per_class) / len(per_class)
        for metric in ("precision", "recall", "f1")
    }
    weighted = {
        metric: (
            sum(row[metric] * row["support"] for row in per_class) / total
            if total
            else 0.0
        )
        for metric in ("precision", "recall", "f1")
    }
    macro["support"] = total
    weighted["support"] = total
    correct = sum(int(matrix[index][index]) for index in range(len(classes)))
    return {
        "accuracy": correct / total if total else 0.0,
        "sample_count": total,
        "per_class": per_class,
        "macro_avg": macro,
        "weighted_avg": weighted,
    }


def write_confusion_png(
    path: Path, matrix: Sequence[Sequence[int]], classes: Sequence[str]
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError as error:
        raise PipelineError(
            "Pillow is required to write confusion_matrix.png; install "
            "training/configs/requirements-classification.txt"
        ) from error

    cell = 96
    left = 180
    top = 120
    bottom = 70
    width = left + cell * len(classes) + 20
    height = top + cell * len(classes) + bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
        small_font = ImageFont.truetype("arial.ttf", 12)
        title_font = ImageFont.truetype("arialbd.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
        small_font = font
        title_font = font

    draw.text((20, 18), "Confusion matrix", fill="black", font=title_font)
    draw.text(
        (20, 48),
        "rows = true class, columns = predicted class",
        fill=(70, 70, 70),
        font=font,
    )
    maximum = max((max(row) for row in matrix), default=1) or 1

    for index, class_name in enumerate(classes):
        x = left + index * cell + 4
        draw.text((x, top - 48), class_name, fill="black", font=small_font)
        y = top + index * cell + cell // 2 - 8
        draw.text((8, y), class_name, fill="black", font=font)

    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            ratio = value / maximum
            color = (
                int(245 - 185 * ratio),
                int(248 - 125 * ratio),
                int(255 - 35 * ratio),
            )
            x0 = left + column_index * cell
            y0 = top + row_index * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle((x0, y0, x1, y1), fill=color, outline=(180, 180, 180))
            value_text = str(value)
            try:
                box = draw.textbbox((0, 0), value_text, font=font)
                text_width = box[2] - box[0]
                text_height = box[3] - box[1]
            except AttributeError:
                text_width, text_height = draw.textsize(value_text, font=font)
            draw.text(
                (
                    x0 + (cell - text_width) / 2,
                    y0 + (cell - text_height) / 2,
                ),
                value_text,
                fill="white" if ratio > 0.58 else "black",
                font=font,
            )
    draw.text(
        (20, height - 42),
        "Scope: synthetic same-base sanity test (not real-domain generalization)",
        fill=(145, 35, 35),
        font=small_font,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def write_evaluation_artifacts(
    output_directory: Path,
    classes: Sequence[str],
    predictions: Sequence[dict[str, Any]],
    matrix: Sequence[Sequence[int]],
    metrics: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    prediction_path = output_directory / "predictions.csv"
    prediction_columns = [
        "sample_id",
        "image_path",
        "true_class",
        "severity",
        "predicted_class",
        "correct",
        "confidence",
        "split",
    ]
    with prediction_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=prediction_columns, lineterminator="\n"
        )
        writer.writeheader()
        for row in predictions:
            writer.writerow({column: row[column] for column in prediction_columns})

    with (output_directory / "metrics_per_class.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fieldnames = [
            "class",
            "class_index",
            "precision",
            "recall",
            "f1",
            "support",
            "true_positive",
            "false_positive",
            "false_negative",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in metrics["per_class"]:
            formatted = dict(row)
            for key in ("precision", "recall", "f1"):
                formatted[key] = f"{float(formatted[key]):.10f}"
            writer.writerow(formatted)

    with (output_directory / "confusion_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["true_class\\predicted_class", *classes])
        for class_name, row in zip(classes, matrix):
            writer.writerow([class_name, *row])

    summary = {
        "schema_version": "1.0",
        "evaluation_scope": metadata["evaluation_scope"],
        "accuracy": metrics["accuracy"],
        "sample_count": metrics["sample_count"],
        "macro_avg": metrics["macro_avg"],
        "weighted_avg": metrics["weighted_avg"],
        "per_class": metrics["per_class"],
        "confusion_matrix": [list(row) for row in matrix],
        "classes": list(classes),
        "metadata": metadata,
    }
    write_json(output_directory / "metrics_summary.json", summary)
    write_confusion_png(output_directory / "confusion_matrix.png", matrix, classes)


def seed_everything(seed: int, torch_module: Any) -> None:
    random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    torch_module.use_deterministic_algorithms(True)
    if hasattr(torch_module.backends, "cudnn"):
        torch_module.backends.cudnn.deterministic = True
        torch_module.backends.cudnn.benchmark = False


def load_ml_dependencies(require_torchvision: bool = False) -> tuple[Any, Any, Any]:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise PipelineError(
            "PyTorch is required for training/evaluation; install "
            "training/configs/requirements-classification.txt"
        ) from error
    try:
        from PIL import Image, ImageOps
    except ModuleNotFoundError as error:
        raise PipelineError(
            "Pillow is required for training/evaluation; install "
            "training/configs/requirements-classification.txt"
        ) from error
    if require_torchvision:
        try:
            import torchvision  # noqa: F401
        except ModuleNotFoundError as error:
            raise PipelineError(
                "torchvision is required for ResNet-18; install "
                "training/configs/requirements-classification.txt"
            ) from error
    return torch, Image, ImageOps


def choose_device(requested: str, torch_module: Any) -> Any:
    if requested == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch_module.cuda.is_available():
        raise PipelineError("CUDA requested but torch.cuda.is_available() is false")
    if requested not in {"cpu", "cuda"}:
        raise PipelineError(f"unsupported device: {requested}")
    return torch_module.device(requested)


def create_dataset_class(torch_module: Any, image_module: Any, image_ops_module: Any) -> Any:
    class ClassificationDataset(torch_module.utils.data.Dataset):
        def __init__(
            self,
            samples: Sequence[Sample],
            class_to_index: dict[str, int],
            input_size: int,
            mean: Sequence[float],
            std: Sequence[float],
            fixed_component_roi_xyxy: Sequence[int],
            training: bool = False,
            augmentation: dict[str, Any] | None = None,
            augmentation_seed: int = 0,
        ) -> None:
            self.samples = list(samples)
            self.class_to_index = class_to_index
            self.input_size = input_size
            self.mean = torch_module.tensor(mean, dtype=torch_module.float32).view(3, 1, 1)
            self.std = torch_module.tensor(std, dtype=torch_module.float32).view(3, 1, 1)
            self.fixed_component_roi_xyxy = tuple(fixed_component_roi_xyxy)
            self.training = training
            self.augmentation = augmentation or {"enabled": False}
            self.augmentation_seed = augmentation_seed
            self.epoch = 0
            if (self.std <= 0).any():
                raise PipelineError("normalization std values must be positive")

        def set_epoch(self, epoch: int) -> None:
            self.epoch = int(epoch)

        def set_samples(self, samples: Sequence[Sample]) -> None:
            if not samples:
                raise PipelineError("classification dataset sample list cannot be empty")
            self.samples = list(samples)

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int) -> tuple[Any, int, str]:
            sample = self.samples[index]
            with image_module.open(sample.image_absolute_path) as opened:
                image = opened.convert("RGB")
                if image.size != (sample.width, sample.height):
                    raise PipelineError(
                        f"decoded image dimensions changed for {sample.sample_id}: "
                        f"manifest={(sample.width, sample.height)} decoded={image.size}"
                    )
                # This ROI is fixed globally and is never derived from the label,
                # defect mask, defect bbox, or sample-specific metadata.
                image = image.crop(self.fixed_component_roi_xyxy)
                if self.training and bool(self.augmentation.get("enabled", False)):
                    augmentation_key = hashlib.sha256(
                        (
                            f"{self.augmentation_seed}\0{self.epoch}\0"
                            f"{sample.sample_id}"
                        ).encode("utf-8")
                    ).digest()
                    rng = random.Random(int.from_bytes(augmentation_key[:8], "big"))
                    if rng.random() < float(
                        self.augmentation.get("horizontal_flip_probability", 0.0)
                    ):
                        image = image_ops_module.mirror(image)
                    rotation_limit = float(
                        self.augmentation.get("rotation_degrees", 0.0)
                    )
                    translation_fraction = float(
                        self.augmentation.get("translation_fraction", 0.0)
                    )
                    angle = rng.uniform(-rotation_limit, rotation_limit)
                    translate_x = round(
                        rng.uniform(-translation_fraction, translation_fraction)
                        * image.width
                    )
                    translate_y = round(
                        rng.uniform(-translation_fraction, translation_fraction)
                        * image.height
                    )
                    image = image.rotate(
                        angle,
                        resample=image_module.Resampling.BILINEAR,
                        translate=(translate_x, translate_y),
                        fillcolor=(244, 244, 244),
                    )
                image = image.resize(
                    (self.input_size, self.input_size),
                    resample=image_module.Resampling.BILINEAR,
                )
                buffer = bytearray(image.tobytes())
            tensor = torch_module.frombuffer(buffer, dtype=torch_module.uint8)
            tensor = tensor.reshape(self.input_size, self.input_size, 3)
            tensor = tensor.permute(2, 0, 1).to(dtype=torch_module.float32).div_(255.0)
            tensor = (tensor - self.mean) / self.std
            return tensor, self.class_to_index[sample.label], sample.sample_id

    return ClassificationDataset


def build_model(
    config: dict[str, Any],
    class_count: int,
    torch_module: Any,
    weights_mode_override: str | None = None,
    local_weights_override: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    model_config = config["model"]
    architecture = str(model_config.get("architecture", "small_cnn"))
    weights_config = _require_mapping(model_config.get("weights", {}), "model.weights")
    weights_mode = weights_mode_override or str(weights_config.get("mode", "none"))
    configured_path = weights_config.get("path")
    local_weights = local_weights_override
    if local_weights is None and isinstance(configured_path, str) and configured_path:
        local_weights = Path(configured_path).expanduser().resolve()
    nn = torch_module.nn

    if architecture == "small_cnn":
        if weights_mode != "none" or local_weights is not None:
            raise PipelineError("small_cnn supports weights.mode='none' only")

        class ConvBlock(nn.Module):
            def __init__(self, in_channels: int, out_channels: int) -> None:
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )

            def forward(self, inputs: Any) -> Any:
                return self.layers(inputs)

        class SmallDefectCNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = nn.Sequential(
                    ConvBlock(3, 32),
                    ConvBlock(32, 64),
                    ConvBlock(64, 128),
                    ConvBlock(128, 192),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(p=0.20),
                    nn.Linear(192, class_count),
                )

            def forward(self, inputs: Any) -> Any:
                return self.classifier(self.features(inputs))

        model = SmallDefectCNN()
        return model, {
            "architecture": architecture,
            "weights_mode": "none",
            "weights_path": None,
            "pretrained": False,
        }

    if architecture != "resnet18":
        raise PipelineError(f"unsupported model architecture: {architecture}")

    load_ml_dependencies(require_torchvision=True)
    import torchvision

    model = torchvision.models.resnet18(weights=None)
    loaded_path: Path | None = None
    if weights_mode == "none":
        if local_weights is not None:
            raise PipelineError("local weights path supplied while weights.mode is 'none'")
    elif weights_mode == "local_path":
        if local_weights is None:
            raise PipelineError("weights.mode='local_path' requires model.weights.path")
        loaded_path = local_weights
    elif weights_mode == "torchvision_cache":
        from urllib.parse import urlparse

        filename = Path(
            urlparse(torchvision.models.ResNet18_Weights.DEFAULT.url).path
        ).name
        expected_filename = weights_config.get("expected_cache_filename")
        if expected_filename and expected_filename != filename:
            raise PipelineError(
                "configured torchvision cache filename differs from torchvision "
                f"ResNet18_Weights.DEFAULT: {expected_filename} != {filename}"
            )
        loaded_path = Path(torch_module.hub.get_dir()) / "checkpoints" / filename
    else:
        raise PipelineError(f"unsupported weights mode: {weights_mode}")

    if loaded_path is not None:
        if not loaded_path.is_file():
            raise PipelineError(
                "local torchvision weights not found; this pipeline never downloads weights: "
                f"{loaded_path}"
            )
        try:
            state = torch_module.load(
                loaded_path, map_location="cpu", weights_only=True
            )
        except TypeError:
            state = torch_module.load(loaded_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise PipelineError(f"unexpected local weights format: {loaded_path}")
        state = {
            key.removeprefix("module."): value for key, value in state.items()
        }
        model.load_state_dict(state, strict=True)

    model.fc = nn.Linear(model.fc.in_features, class_count)
    return model, {
        "architecture": architecture,
        "weights_mode": weights_mode,
        # Persist only the portable filename. Absolute cache paths can expose a
        # workstation username and are not needed because the SHA-256 identifies
        # the exact pretrained file.
        "weights_path": loaded_path.name if loaded_path else None,
        "weights_sha256": sha256_file(loaded_path) if loaded_path else None,
        "pretrained": loaded_path is not None,
        "network_download_permitted": False,
    }


def evaluate_model(
    model: Any,
    loader: Any,
    samples_by_id: dict[str, Sample],
    classes: Sequence[str],
    device: Any,
    torch_module: Any,
) -> tuple[list[dict[str, Any]], list[list[int]], dict[str, Any]]:
    model.eval()
    true_indices: list[int] = []
    predicted_indices: list[int] = []
    predictions: list[dict[str, Any]] = []
    with torch_module.no_grad():
        for inputs, targets, sample_ids in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            if logits.ndim != 2 or logits.shape[1] != len(classes):
                raise PipelineError(
                    f"model output shape must be [N,{len(classes)}], "
                    f"got {tuple(logits.shape)}"
                )
            if not bool(torch_module.isfinite(logits).all().item()):
                raise PipelineError("model emitted non-finite logits")
            probabilities = torch_module.softmax(logits, dim=1)
            if not bool(torch_module.isfinite(probabilities).all().item()):
                raise PipelineError("model emitted non-finite probabilities")
            probability_sums = probabilities.sum(dim=1)
            if not bool(
                torch_module.allclose(
                    probability_sums,
                    torch_module.ones_like(probability_sums),
                    rtol=1e-5,
                    atol=1e-6,
                )
            ):
                raise PipelineError("model probability rows do not sum to one")
            confidence, predicted = probabilities.max(dim=1)
            target_values = targets.cpu().tolist()
            predicted_values = predicted.cpu().tolist()
            confidence_values = confidence.cpu().tolist()
            for sample_id, target, prediction, score in zip(
                sample_ids,
                target_values,
                predicted_values,
                confidence_values,
            ):
                sample = samples_by_id[sample_id]
                predictions.append(
                    {
                        "sample_id": sample_id,
                        "image_path": sample.image_path,
                        "true_class": classes[target],
                        "severity": sample.severity,
                        "predicted_class": classes[prediction],
                        "correct": "YES" if target == prediction else "NO",
                        "confidence": f"{score:.10f}",
                        "split": "test",
                    }
                )
            true_indices.extend(target_values)
            predicted_indices.extend(predicted_values)
    matrix = confusion_matrix(true_indices, predicted_indices, len(classes))
    metrics = calculate_metrics(matrix, classes)
    return predictions, matrix, metrics


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else math.nan
