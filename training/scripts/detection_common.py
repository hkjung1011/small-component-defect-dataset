"""Integrity gates and data utilities for the v4/v5 component detector.

The integrity preflight intentionally uses only the Python standard library.
PyTorch, torchvision, and Pillow are imported lazily for optional environment
reporting or training; no path downloads packages or model weights.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


STATUS_CATEGORIES: tuple[tuple[int, str], ...] = (
    (1, "normal_proxy"),
    (2, "scratch"),
    (3, "surface_spot"),
    (4, "discoloration"),
    (5, "contamination"),
    (6, "lead_breakage"),
    (7, "body_chip"),
    (8, "body_crack"),
)
SUPPORTED_TASKS = ("component_localization", "component_status")


class PipelineError(RuntimeError):
    """Raised when a hard data, configuration, or runtime gate fails."""


@dataclass(frozen=True)
class ComponentAnnotation:
    scene_instance_id: int
    bbox_xyxy: tuple[float, float, float, float]
    status_category_id: int
    status_name: str
    source_parent_sample_id: str


@dataclass(frozen=True)
class SceneRecord:
    index: int
    release_key: str
    release_image_id: int
    scene_id: str
    family_id: str
    variant_name: str
    image_path: str
    image_absolute_path: Path
    image_sha256: str
    width: int
    height: int
    source_parent_ids: tuple[str, ...]
    source_scene_id: str | None
    source_variant_index: int | None
    source_image_path: str | None
    source_image_sha256: str | None
    annotations: tuple[ComponentAnnotation, ...]


@dataclass(frozen=True)
class PreflightResult:
    config: dict[str, Any]
    config_path: Path
    repository_root: Path
    weight_path: Path
    scenes: tuple[SceneRecord, ...]
    family_to_indices: dict[str, tuple[int, ...]]
    fingerprints: dict[str, Any]
    summary: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise PipelineError(
            f"refusing non-finite or non-JSON value for {path.name}: {error}"
        ) from error
    path.write_text(serialized + "\n", encoding="utf-8")


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PipelineError(f"config field must be an object: {field}")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PipelineError(f"config field must be a positive integer: {field}")
    if value <= 0:
        raise PipelineError(f"config field must be a positive integer: {field}")
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PipelineError(f"config field must be a finite number: {field}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise PipelineError(f"config field must be a finite number: {field}")
    return parsed


def _probability(value: Any, field: str) -> float:
    parsed = _finite_number(value, field)
    if not 0.0 <= parsed <= 1.0:
        raise PipelineError(f"config probability must be in [0, 1]: {field}")
    return parsed


def resolve_repository_path(repository_root: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise PipelineError("repository path must be a non-empty string")
    candidate_value = Path(relative_path)
    if candidate_value.is_absolute():
        raise PipelineError(f"repository path must be relative: {relative_path}")
    candidate = (repository_root / candidate_value).resolve()
    try:
        candidate.relative_to(repository_root)
    except ValueError as error:
        raise PipelineError(f"path escapes repository root: {relative_path}") from error
    return candidate


def _load_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise PipelineError(f"{description} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PipelineError(f"cannot read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise PipelineError(f"{description} must contain a JSON object: {path}")
    return value


def load_config(config_path: Path) -> tuple[dict[str, Any], Path, Path]:
    config_path = config_path.resolve()
    config = _load_json(config_path, "detector config")
    if len(config_path.parents) < 3:
        raise PipelineError(f"cannot infer repository root from {config_path}")
    repository_root = config_path.parents[2]

    if config.get("schema_version") != "component-detector-train-v1":
        raise PipelineError("unsupported detector config schema_version")
    if config.get("default_task") != "component_localization":
        raise PipelineError("default_task must be component_localization")
    if config.get("annotation_namespace") != "component_status_only":
        raise PipelineError(
            "annotation_namespace must be component_status_only; defect annotations "
            "must not enter this pipeline"
        )
    supported = _mapping(config.get("supported_tasks"), "supported_tasks")
    if tuple(supported) != SUPPORTED_TASKS:
        raise PipelineError(
            "supported_tasks must contain component_localization then component_status"
        )
    expected_status = [name for _, name in STATUS_CATEGORIES]
    if supported["component_localization"].get("foreground_classes") != ["component"]:
        raise PipelineError("component_localization must have one component foreground class")
    if supported["component_status"].get("foreground_classes") != expected_status:
        raise PipelineError("component_status class order does not match the canonical namespace")

    releases = config.get("releases")
    if not isinstance(releases, list) or [item.get("key") for item in releases] != ["v4", "v5"]:
        raise PipelineError("releases must be ordered as v4 then v5")
    for release in releases:
        if not isinstance(release, dict):
            raise PipelineError("every release entry must be an object")
        coco_path = str(release.get("component_coco", ""))
        if Path(coco_path).name != "component_status_train.json" or "defect" in coco_path.lower():
            raise PipelineError(
                "only component_status_train.json is permitted; defect namespace mixing is forbidden"
            )
        for digest_field in (
            "expected_manifest_sha256",
            "expected_component_coco_sha256",
            "expected_release_metadata_sha256",
            "expected_image_content_fingerprint_sha256",
        ):
            digest = release.get(digest_field)
            if not isinstance(digest, str) or len(digest) != 64:
                raise PipelineError(f"release {release.get('key')} has invalid {digest_field}")
        if release.get("key") == "v4":
            split_digest = release.get("expected_source_split_assignments_sha256")
            if not isinstance(split_digest, str) or len(split_digest) != 64:
                raise PipelineError("v4 release must pin source split assignments SHA-256")

    source_catalog = _mapping(config.get("source_parent_catalog"), "source_parent_catalog")
    if source_catalog.get("id_field") != "sample_id":
        raise PipelineError("source_parent_catalog.id_field must be sample_id")
    if _positive_int(
        source_catalog.get("expected_row_count"), "source_parent_catalog.expected_row_count"
    ) != 700:
        raise PipelineError("source parent catalog must contain exactly 700 rows")
    source_digest = source_catalog.get("expected_sha256")
    if not isinstance(source_digest, str) or len(source_digest) != 64:
        raise PipelineError("source_parent_catalog.expected_sha256 must be a SHA-256 digest")

    split_assignments = _mapping(
        config.get("source_split_assignments"), "source_split_assignments"
    )
    split_digest = split_assignments.get("expected_sha256")
    if not isinstance(split_digest, str) or len(split_digest) != 64:
        raise PipelineError("source_split_assignments.expected_sha256 must be a SHA-256 digest")
    if _positive_int(
        split_assignments.get("expected_row_count"),
        "source_split_assignments.expected_row_count",
    ) != 700:
        raise PipelineError("source split assignments must contain exactly 700 rows")
    required_split_columns = [
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
    if split_assignments.get("required_columns") != required_split_columns:
        raise PipelineError("source_split_assignments required_columns schema changed")
    if split_assignments.get("expected_model_split_counts") != {
        "gradient_train": 168,
        "validation": 28,
        "test": 504,
    }:
        raise PipelineError("source split assignment model_split counts changed")
    if split_assignments.get("required_referenced_parent_model_split") != "gradient_train":
        raise PipelineError("referenced parents must be restricted to gradient_train")
    if releases[0].get("expected_source_split_assignments_sha256") != split_digest:
        raise PipelineError("v4 release and authoritative source split digests disagree")
    if _positive_int(
        split_assignments.get("expected_referenced_parent_count"),
        "source_split_assignments.expected_referenced_parent_count",
    ) != 168:
        raise PipelineError("expected referenced source-parent count must be 168")
    for field in ("expected_referenced_validation_count", "expected_referenced_test_count"):
        value = split_assignments.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            raise PipelineError(f"source_split_assignments.{field} must be 0")

    family = _mapping(config.get("family_policy"), "family_policy")
    required_family_values = {
        "family_field": "composition_family_id",
        "expected_family_count": 384,
        "expected_v4_variants_per_family": 1,
        "expected_v5_variants_per_family": 2,
        "instances_per_scene": 5,
        "sampler": "one_rotating_variant_per_family_per_epoch",
    }
    if any(family.get(key) != value for key, value in required_family_values.items()):
        raise PipelineError("family_policy does not match the fixed v4/v5 family contract")

    split = _mapping(config.get("split_policy"), "split_policy")
    if split.get("train_only") is not True:
        raise PipelineError("split_policy.train_only must be true")
    if split.get("validation_permitted") is not False or split.get("test_permitted") is not False:
        raise PipelineError("validation and test creation are forbidden for this connected source graph")

    model = _mapping(config.get("model"), "model")
    if model.get("architecture") != "fasterrcnn_mobilenet_v3_large_fpn":
        raise PipelineError("only fasterrcnn_mobilenet_v3_large_fpn is supported")
    min_size = _positive_int(model.get("min_size"), "model.min_size")
    max_size = _positive_int(model.get("max_size"), "model.max_size")
    if min_size > max_size:
        raise PipelineError("model.min_size must not exceed model.max_size")
    if _positive_int(
        model.get("trainable_backbone_layers"), "model.trainable_backbone_layers"
    ) > 6:
        raise PipelineError("model.trainable_backbone_layers must be in [1, 6]")
    weights = _mapping(model.get("weights"), "model.weights")
    if weights.get("mode") != "torchvision_cache_local_only":
        raise PipelineError("pretrained weights mode must be torchvision_cache_local_only")
    if weights.get("network_download_permitted") is not False:
        raise PipelineError("network_download_permitted must be false")
    if weights.get("expected_cache_filename") != "fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth":
        raise PipelineError("unexpected official detector weight cache filename")
    if weights.get("expected_sha256") != "fb6a3cc702b1df54c18a44b26708cd083614211062d0c36d2ca7bf9270df3533":
        raise PipelineError("unexpected official detector weight SHA-256")

    training = _mapping(config.get("training"), "training")
    _positive_int(training.get("seed"), "training.seed")
    _positive_int(training.get("augmentation_seed"), "training.augmentation_seed")
    _positive_int(training.get("epochs"), "training.epochs")
    if _positive_int(training.get("batch_size"), "training.batch_size") != 1:
        raise PipelineError("the checked default detector batch_size must be 1")
    workers = training.get("num_workers")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise PipelineError("training.num_workers must be a non-negative integer")
    if training.get("amp") is not True:
        raise PipelineError("training.amp must be true; it is applied only on CUDA")
    if training.get("optimizer") != "sgd":
        raise PipelineError("training.optimizer must be sgd")
    if _finite_number(training.get("learning_rate"), "training.learning_rate") <= 0.0:
        raise PipelineError("training.learning_rate must be positive")
    momentum = _finite_number(training.get("momentum"), "training.momentum")
    if not 0.0 <= momentum < 1.0:
        raise PipelineError("training.momentum must be in [0, 1)")
    if _finite_number(training.get("weight_decay"), "training.weight_decay") < 0.0:
        raise PipelineError("training.weight_decay must be non-negative")
    if _finite_number(training.get("gradient_clip_norm"), "training.gradient_clip_norm") <= 0.0:
        raise PipelineError("training.gradient_clip_norm must be positive")
    retry_limit = _positive_int(
        training.get("nonfinite_gradient_max_retries_per_batch"),
        "training.nonfinite_gradient_max_retries_per_batch",
    )
    if retry_limit > 16:
        raise PipelineError("nonfinite_gradient_max_retries_per_batch must not exceed 16")
    if training.get("schedule") != "fixed_epochs_no_validation":
        raise PipelineError("training schedule must be fixed_epochs_no_validation")
    if _positive_int(training.get("smoke_max_steps"), "training.smoke_max_steps") > 8:
        raise PipelineError("smoke_max_steps must not exceed 8")
    augmentation = _mapping(training.get("augmentation"), "training.augmentation")
    if augmentation.get("library") != "torchvision.transforms.v2" or augmentation.get("bbox_aware") is not True:
        raise PipelineError("augmentation must use bbox-aware torchvision.transforms.v2")
    for prohibited in ("random_erasing", "cutmix", "mixup", "mosaic"):
        if augmentation.get(prohibited) is not False:
            raise PipelineError(f"prohibited detector augmentation must be false: {prohibited}")
    _probability(
        augmentation.get("horizontal_flip_probability"),
        "training.augmentation.horizontal_flip_probability",
    )
    _probability(
        augmentation.get("affine_probability"),
        "training.augmentation.affine_probability",
    )
    _probability(
        augmentation.get("color_jitter_probability"),
        "training.augmentation.color_jitter_probability",
    )
    _probability(
        augmentation.get("gaussian_blur_probability"),
        "training.augmentation.gaussian_blur_probability",
    )
    rotation = _finite_number(
        augmentation.get("rotation_degrees"), "training.augmentation.rotation_degrees"
    )
    if not 0.0 <= rotation <= 5.0:
        raise PipelineError("weak affine rotation must be in [0, 5] degrees")
    translation = _finite_number(
        augmentation.get("translation_fraction"),
        "training.augmentation.translation_fraction",
    )
    if not 0.0 <= translation <= 0.04:
        raise PipelineError("weak affine translation must be in [0, 0.04]")
    scale_range = augmentation.get("scale_range")
    if not isinstance(scale_range, list) or len(scale_range) != 2:
        raise PipelineError("training.augmentation.scale_range must contain two values")
    scale_low = _finite_number(scale_range[0], "training.augmentation.scale_range[0]")
    scale_high = _finite_number(scale_range[1], "training.augmentation.scale_range[1]")
    if not 0.0 < scale_low <= 1.0 <= scale_high or scale_high > 1.10:
        raise PipelineError("weak affine scale_range must satisfy 0 < low <= 1 <= high <= 1.10")
    for name, maximum in (
        ("brightness", 0.20),
        ("contrast", 0.20),
        ("saturation", 0.20),
        ("hue", 0.05),
    ):
        amount = _finite_number(
            augmentation.get(name), f"training.augmentation.{name}"
        )
        if not 0.0 <= amount <= maximum:
            raise PipelineError(f"weak color jitter {name} must be in [0, {maximum}]")
    blur_kernel = _positive_int(
        augmentation.get("gaussian_blur_kernel"),
        "training.augmentation.gaussian_blur_kernel",
    )
    if blur_kernel % 2 == 0:
        raise PipelineError("gaussian_blur_kernel must be odd")
    blur_sigma = augmentation.get("gaussian_blur_sigma")
    if not isinstance(blur_sigma, list) or len(blur_sigma) != 2:
        raise PipelineError("gaussian_blur_sigma must contain two values")
    sigma_low = _finite_number(blur_sigma[0], "training.augmentation.gaussian_blur_sigma[0]")
    sigma_high = _finite_number(blur_sigma[1], "training.augmentation.gaussian_blur_sigma[1]")
    if not 0.0 < sigma_low <= sigma_high <= 2.0:
        raise PipelineError("gaussian_blur_sigma must satisfy 0 < low <= high <= 2")

    diagnostics = _mapping(config.get("diagnostics"), "diagnostics")
    if diagnostics.get("scope") != "TRAIN_DIAGNOSTIC_ONLY":
        raise PipelineError("diagnostic scope must be TRAIN_DIAGNOSTIC_ONLY")
    if diagnostics.get("performance_claim_permitted") is not False:
        raise PipelineError("performance claims must be disabled")
    _probability(diagnostics.get("sample_score_threshold"), "diagnostics.sample_score_threshold")
    _positive_int(diagnostics.get("sample_max_predictions"), "diagnostics.sample_max_predictions")
    return config, config_path, repository_root


def _torch_cache_root() -> Path:
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        return Path(torch_home).expanduser().resolve()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return (Path(xdg_cache).expanduser() / "torch").resolve()
    return (Path.home() / ".cache" / "torch").resolve()


def resolve_and_verify_cached_weight(config: dict[str, Any]) -> tuple[Path, str]:
    weights = config["model"]["weights"]
    filename = weights["expected_cache_filename"]
    weight_path = _torch_cache_root() / "hub" / "checkpoints" / filename
    if not weight_path.is_file():
        raise PipelineError(
            "official Faster R-CNN weight is absent from the local torchvision cache: "
            f"{filename}. Network download is disabled; provision the verified cache file first."
        )
    actual_sha = sha256_file(weight_path)
    expected_sha = weights["expected_sha256"]
    if actual_sha != expected_sha:
        raise PipelineError(
            f"cached official weight SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
        )
    return weight_path, actual_sha


def _read_csv_with_header(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise PipelineError(f"CSV file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise PipelineError(f"CSV file has no header: {path}")
        header = list(reader.fieldnames)
        rows = list(reader)
    return header, rows


def _read_manifest(path: Path) -> list[dict[str, str]]:
    return _read_csv_with_header(path)[1]


def _jpeg_dimensions(path: Path) -> tuple[int, int]:
    """Read JPEG SOF dimensions without depending on Pillow during preflight."""

    start_of_frame_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    try:
        with path.open("rb") as stream:
            if stream.read(2) != b"\xff\xd8":
                raise PipelineError(f"expected JPEG SOI marker: {path}")
            while True:
                byte = stream.read(1)
                while byte and byte != b"\xff":
                    byte = stream.read(1)
                if not byte:
                    break
                marker_byte = stream.read(1)
                while marker_byte == b"\xff":
                    marker_byte = stream.read(1)
                if not marker_byte:
                    break
                marker = marker_byte[0]
                if marker in {0x01, *range(0xD0, 0xDA)}:
                    continue
                length_bytes = stream.read(2)
                if len(length_bytes) != 2:
                    break
                segment_length = int.from_bytes(length_bytes, "big")
                if segment_length < 2:
                    raise PipelineError(f"invalid JPEG segment length: {path}")
                if marker in start_of_frame_markers:
                    payload = stream.read(5)
                    if len(payload) != 5:
                        break
                    height = int.from_bytes(payload[1:3], "big")
                    width = int.from_bytes(payload[3:5], "big")
                    if width <= 0 or height <= 0:
                        raise PipelineError(f"invalid JPEG dimensions: {path}")
                    return width, height
                stream.seek(segment_length - 2, 1)
    except OSError as error:
        raise PipelineError(f"cannot inspect JPEG dimensions {path}: {error}") from error
    raise PipelineError(f"JPEG SOF dimension marker not found: {path}")


def _parse_int(value: Any, description: str) -> int:
    if isinstance(value, bool):
        raise PipelineError(f"invalid integer for {description}: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and (stripped.isdigit() or (stripped[0] == "-" and stripped[1:].isdigit())):
            return int(stripped)
    raise PipelineError(f"invalid integer for {description}: {value!r}")


def _validate_categories(coco: dict[str, Any], release_key: str) -> None:
    categories = coco.get("categories")
    if not isinstance(categories, list):
        raise PipelineError(f"{release_key} COCO categories must be a list")
    actual: list[tuple[int, str]] = []
    for category in categories:
        if not isinstance(category, dict):
            raise PipelineError(f"{release_key} COCO category is not an object")
        actual.append((_parse_int(category.get("id"), "category id"), str(category.get("name"))))
    if tuple(actual) != STATUS_CATEGORIES:
        raise PipelineError(
            f"{release_key} COCO is not the canonical eight-class component-status namespace"
        )


def _validate_release(
    *,
    release_config: dict[str, Any],
    repository_root: Path,
    next_index: int,
) -> tuple[list[SceneRecord], dict[str, str], dict[str, tuple[str, ...]], dict[str, Any]]:
    key = str(release_config["key"])
    root_path = resolve_repository_path(repository_root, str(release_config["root"]))
    manifest_path = resolve_repository_path(repository_root, str(release_config["manifest"]))
    coco_path = resolve_repository_path(repository_root, str(release_config["component_coco"]))
    release_path = resolve_repository_path(repository_root, str(release_config["release_metadata"]))
    metadata = _load_json(release_path, f"{key} release metadata")
    coco = _load_json(coco_path, f"{key} component-status COCO")
    rows = _read_manifest(manifest_path)
    expected_scenes = _positive_int(release_config.get("expected_scene_count"), f"{key}.expected_scene_count")
    expected_training_use = str(release_config.get("expected_training_use"))
    if len(rows) != expected_scenes:
        raise PipelineError(f"{key} manifest scene count must be {expected_scenes}, got {len(rows)}")
    if metadata.get("scene_count") != expected_scenes:
        raise PipelineError(f"{key} release metadata scene_count mismatch")
    if metadata.get("instances_per_scene") != 5:
        raise PipelineError(f"{key} release metadata instances_per_scene must be 5")
    if metadata.get("training_use") != expected_training_use:
        raise PipelineError(f"{key} release metadata training_use mismatch")
    if metadata.get("evaluation_eligible") != "NO":
        raise PipelineError(f"{key} release metadata evaluation_eligible must be NO")
    if metadata.get("classification_eligible") != "NO":
        raise PipelineError(f"{key} release metadata classification_eligible must be NO")
    if key == "v4" and metadata.get("source_split_assignments_sha256") != release_config.get(
        "expected_source_split_assignments_sha256"
    ):
        raise PipelineError("v4 release metadata source split assignment SHA-256 changed")

    manifest_sha = sha256_file(manifest_path)
    coco_sha = sha256_file(coco_path)
    release_metadata_sha = sha256_file(release_path)
    if manifest_sha != release_config["expected_manifest_sha256"]:
        raise PipelineError(f"{key} manifest differs from the config-pinned known-good digest")
    if coco_sha != release_config["expected_component_coco_sha256"]:
        raise PipelineError(f"{key} component COCO differs from the config-pinned known-good digest")
    if release_metadata_sha != release_config["expected_release_metadata_sha256"]:
        raise PipelineError(
            f"{key} release metadata differs from the config-pinned known-good digest"
        )
    if metadata.get("manifest_sha256") != manifest_sha:
        raise PipelineError(f"{key} manifest hash is not pinned by release metadata")
    if metadata.get("component_coco_sha256") != coco_sha:
        raise PipelineError(f"{key} component COCO hash is not pinned by release metadata")

    required_manifest_fields = {
        "scene_id",
        "image_id",
        "image_path",
        "split",
        "training_use",
        "evaluation_eligible",
        "classification_eligible",
        "instance_count",
        "component_status_labels",
        "source_parent_ids",
        "composition_family_id",
        "width",
        "height",
        "image_sha256",
    }
    if key == "v5":
        required_manifest_fields.update(
            {"source_scene_id", "source_variant_index", "source_image_path", "source_image_sha256"}
        )
    if not rows or not required_manifest_fields.issubset(rows[0]):
        missing = sorted(required_manifest_fields - (set(rows[0]) if rows else set()))
        raise PipelineError(f"{key} manifest missing required columns: {missing}")

    _validate_categories(coco, key)
    coco_images = coco.get("images")
    coco_annotations = coco.get("annotations")
    if not isinstance(coco_images, list) or len(coco_images) != expected_scenes:
        raise PipelineError(f"{key} COCO image count must be {expected_scenes}")
    if not isinstance(coco_annotations, list) or len(coco_annotations) != expected_scenes * 5:
        raise PipelineError(f"{key} COCO annotation count must be {expected_scenes * 5}")

    images_by_id: dict[int, dict[str, Any]] = {}
    for image in coco_images:
        if not isinstance(image, dict):
            raise PipelineError(f"{key} COCO image entry is not an object")
        image_id = _parse_int(image.get("id"), f"{key} COCO image id")
        if image_id in images_by_id:
            raise PipelineError(f"{key} duplicate COCO image id: {image_id}")
        images_by_id[image_id] = image
    if set(images_by_id) != set(range(1, expected_scenes + 1)):
        raise PipelineError(f"{key} COCO image IDs must restart locally at 1..{expected_scenes}")

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    annotation_ids: set[int] = set()
    for annotation in coco_annotations:
        if not isinstance(annotation, dict):
            raise PipelineError(f"{key} COCO annotation entry is not an object")
        annotation_id = _parse_int(annotation.get("id"), f"{key} annotation id")
        if annotation_id in annotation_ids:
            raise PipelineError(f"{key} duplicate COCO annotation id: {annotation_id}")
        annotation_ids.add(annotation_id)
        image_id = _parse_int(annotation.get("image_id"), f"{key} annotation image_id")
        if image_id not in images_by_id:
            raise PipelineError(f"{key} annotation references unknown image id: {image_id}")
        annotations_by_image[image_id].append(annotation)

    records: list[SceneRecord] = []
    scene_to_sha: dict[str, str] = {}
    scene_to_parents: dict[str, tuple[str, ...]] = {}
    seen_scene_ids: set[str] = set()
    seen_image_paths: set[str] = set()
    image_hash_lines: list[str] = []

    for row_number, row in enumerate(rows, start=2):
        prefix = f"{key} manifest row {row_number}"
        scene_id = row["scene_id"]
        if not scene_id or scene_id in seen_scene_ids:
            raise PipelineError(f"{prefix} has empty or duplicate scene_id: {scene_id!r}")
        seen_scene_ids.add(scene_id)
        image_id = _parse_int(row["image_id"], f"{prefix} image_id")
        image = images_by_id.get(image_id)
        if image is None:
            raise PipelineError(f"{prefix} image_id is absent from release-scoped COCO: {image_id}")
        if image.get("scene_id") != scene_id:
            raise PipelineError(f"{prefix} scene_id does not match COCO image")
        if row["split"] != "train" or row["training_use"] != expected_training_use:
            raise PipelineError(f"{prefix} must remain in the declared TRAIN_ONLY release")
        if row["evaluation_eligible"] != "NO" or row["classification_eligible"] != "NO":
            raise PipelineError(f"{prefix} must have evaluation/classification eligibility NO")
        if _parse_int(row["instance_count"], f"{prefix} instance_count") != 5:
            raise PipelineError(f"{prefix} must contain exactly five component instances")
        width = _parse_int(row["width"], f"{prefix} width")
        height = _parse_int(row["height"], f"{prefix} height")
        if width != _parse_int(image.get("width"), f"{prefix} COCO width") or height != _parse_int(
            image.get("height"), f"{prefix} COCO height"
        ):
            raise PipelineError(f"{prefix} dimensions disagree with COCO")

        image_relative = row["image_path"].replace("\\", "/")
        if image_relative in seen_image_paths:
            raise PipelineError(f"{prefix} has duplicate image_path")
        seen_image_paths.add(image_relative)
        image_absolute = resolve_repository_path(repository_root, image_relative)
        expected_coco_absolute = (root_path / str(image.get("file_name"))).resolve()
        try:
            expected_coco_absolute.relative_to(root_path)
        except ValueError as error:
            raise PipelineError(f"{prefix} COCO file_name escapes its release root") from error
        if image_absolute != expected_coco_absolute:
            raise PipelineError(f"{prefix} image_path disagrees with release-relative COCO file_name")
        if not image_absolute.is_file():
            raise PipelineError(f"{prefix} image does not exist: {image_relative}")
        expected_image_sha = row["image_sha256"].lower()
        if len(expected_image_sha) != 64:
            raise PipelineError(f"{prefix} has malformed image_sha256")
        actual_image_sha = sha256_file(image_absolute)
        if actual_image_sha != expected_image_sha:
            raise PipelineError(f"{prefix} image SHA-256 mismatch")
        actual_width, actual_height = _jpeg_dimensions(image_absolute)
        if (actual_width, actual_height) != (width, height):
            raise PipelineError(
                f"{prefix} decoded JPEG dimensions {(actual_width, actual_height)} "
                f"do not match metadata {(width, height)}"
            )
        image_hash_lines.append(f"{key}|{scene_id}|{actual_image_sha}")

        family_id = row["composition_family_id"]
        if not family_id:
            raise PipelineError(f"{prefix} has empty composition_family_id")
        if key == "v4":
            if family_id != scene_id:
                raise PipelineError(f"{prefix} v4 family must equal its scene_id")
            variant_name = "v4"
            source_scene_id: str | None = None
            source_variant_index: int | None = None
            source_image_path: str | None = None
            source_image_sha256: str | None = None
        else:
            if row["source_scene_id"] != family_id:
                raise PipelineError(f"{prefix} v5 source_scene_id must equal composition_family_id")
            variant_index = _parse_int(row["source_variant_index"], f"{prefix} source_variant_index")
            if variant_index not in (0, 1):
                raise PipelineError(f"{prefix} v5 source_variant_index must be 0 or 1")
            variant_name = f"v5_{variant_index}"
            source_scene_id = row["source_scene_id"]
            source_variant_index = variant_index
            source_image_path = row["source_image_path"].replace("\\", "/")
            source_image_sha256 = row["source_image_sha256"].lower()
            if len(source_image_sha256) != 64:
                raise PipelineError(f"{prefix} has malformed source_image_sha256")
            source_absolute = resolve_repository_path(repository_root, source_image_path)
            if not source_absolute.is_file():
                raise PipelineError(f"{prefix} source_image_path does not exist")
            if (
                image.get("composition_family_id") != family_id
                or image.get("source_scene_id") != source_scene_id
                or _parse_int(image.get("source_variant_index"), f"{prefix} COCO source_variant_index")
                != source_variant_index
            ):
                raise PipelineError(f"{prefix} v5 lineage disagrees with COCO image metadata")

        labels = tuple(value for value in row["component_status_labels"].split("|") if value)
        parents = tuple(value for value in row["source_parent_ids"].split("|") if value)
        if len(labels) != 5 or len(parents) != 5:
            raise PipelineError(f"{prefix} must have five status labels and five source parents")
        if any(label not in dict(STATUS_CATEGORIES).values() for label in labels):
            raise PipelineError(f"{prefix} contains a label outside component_status namespace")

        parsed_annotations_by_instance: dict[int, ComponentAnnotation] = {}
        status_names: list[str] = []
        image_annotations = annotations_by_image.get(image_id, [])
        if len(image_annotations) != 5:
            raise PipelineError(f"{prefix} COCO image must have exactly five annotations")
        for annotation in sorted(image_annotations, key=lambda item: int(item["id"])):
            category_id = _parse_int(annotation.get("category_id"), f"{prefix} category_id")
            category_map = dict(STATUS_CATEGORIES)
            if category_id not in category_map:
                raise PipelineError(f"{prefix} annotation category is outside component_status namespace")
            if _parse_int(annotation.get("iscrowd", 0), f"{prefix} iscrowd") != 0:
                raise PipelineError(f"{prefix} crowd annotations are unsupported")
            bbox = annotation.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise PipelineError(f"{prefix} annotation bbox must be COCO xywh")
            try:
                x, y, box_width, box_height = (float(value) for value in bbox)
            except (TypeError, ValueError) as error:
                raise PipelineError(f"{prefix} annotation bbox contains non-numeric values") from error
            if not all(math.isfinite(value) for value in (x, y, box_width, box_height)):
                raise PipelineError(f"{prefix} annotation bbox contains non-finite values")
            if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
                raise PipelineError(f"{prefix} annotation bbox is invalid")
            if x + box_width > width + 1e-6 or y + box_height > height + 1e-6:
                raise PipelineError(f"{prefix} annotation bbox exceeds image bounds")
            try:
                area = float(annotation.get("area", 0.0))
            except (TypeError, ValueError) as error:
                raise PipelineError(f"{prefix} annotation area is non-numeric") from error
            if not math.isfinite(area) or area <= 0.0:
                raise PipelineError(f"{prefix} annotation area must be positive")
            if area > box_width * box_height + 1e-6:
                raise PipelineError(f"{prefix} annotation area exceeds its bbox area")
            attributes = annotation.get("attributes")
            if not isinstance(attributes, dict):
                raise PipelineError(f"{prefix} annotation attributes must be an object")
            scene_instance_id = _parse_int(
                attributes.get("scene_instance_id"), f"{prefix} scene_instance_id"
            )
            if scene_instance_id not in range(1, 6):
                raise PipelineError(f"{prefix} scene_instance_id must be in 1..5")
            if scene_instance_id in parsed_annotations_by_instance:
                raise PipelineError(f"{prefix} has duplicate scene_instance_id")
            source_parent_sample_id = str(attributes.get("source_parent_sample_id", ""))
            if source_parent_sample_id != parents[scene_instance_id - 1]:
                raise PipelineError(
                    f"{prefix} COCO source parent disagrees with manifest slot {scene_instance_id}"
                )
            status_name = category_map[category_id]
            if status_name != labels[scene_instance_id - 1]:
                raise PipelineError(
                    f"{prefix} COCO status disagrees with manifest slot {scene_instance_id}"
                )
            status_names.append(status_name)
            parsed_annotations_by_instance[scene_instance_id] = ComponentAnnotation(
                scene_instance_id=scene_instance_id,
                bbox_xyxy=(x, y, x + box_width, y + box_height),
                status_category_id=category_id,
                status_name=status_name,
                source_parent_sample_id=source_parent_sample_id,
            )
        if Counter(status_names) != Counter(labels):
            raise PipelineError(f"{prefix} COCO categories disagree with manifest component_status_labels")
        if set(parsed_annotations_by_instance) != set(range(1, 6)):
            raise PipelineError(f"{prefix} COCO scene_instance_id set must be exactly 1..5")
        parsed_annotations = tuple(
            parsed_annotations_by_instance[instance_id] for instance_id in range(1, 6)
        )

        record = SceneRecord(
            index=next_index + len(records),
            release_key=key,
            release_image_id=image_id,
            scene_id=scene_id,
            family_id=family_id,
            variant_name=variant_name,
            image_path=image_relative,
            image_absolute_path=image_absolute,
            image_sha256=actual_image_sha,
            width=width,
            height=height,
            source_parent_ids=parents,
            source_scene_id=source_scene_id,
            source_variant_index=source_variant_index,
            source_image_path=source_image_path,
            source_image_sha256=source_image_sha256,
            annotations=parsed_annotations,
        )
        records.append(record)
        scene_to_sha[scene_id] = actual_image_sha
        scene_to_parents[scene_id] = parents

    image_content_fingerprint = sha256_lines(sorted(image_hash_lines))
    if image_content_fingerprint != release_config["expected_image_content_fingerprint_sha256"]:
        raise PipelineError(
            f"{key} image payload differs from the config-pinned known-good fingerprint"
        )
    hashes = {
        "manifest_sha256": manifest_sha,
        "component_coco_sha256": coco_sha,
        "release_metadata_sha256": release_metadata_sha,
        "validated_image_content_fingerprint_sha256": image_content_fingerprint,
    }
    return records, scene_to_sha, scene_to_parents, hashes


def _connected_component_count(adjacency: dict[str, set[str]]) -> int:
    unseen = set(adjacency)
    components = 0
    while unseen:
        components += 1
        start = next(iter(unseen))
        queue: deque[str] = deque([start])
        unseen.remove(start)
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
    return components


def validate_pipeline(config_path: Path) -> PreflightResult:
    config, resolved_config_path, repository_root = load_config(config_path)
    weight_path, weight_sha = resolve_and_verify_cached_weight(config)

    source_catalog_config = config["source_parent_catalog"]
    source_catalog_path = resolve_repository_path(
        repository_root, str(source_catalog_config["manifest"])
    )
    source_catalog_sha = sha256_file(source_catalog_path)
    if source_catalog_sha != source_catalog_config["expected_sha256"]:
        raise PipelineError("source parent catalog differs from its config-pinned known-good digest")
    source_catalog_rows = _read_manifest(source_catalog_path)
    if len(source_catalog_rows) != int(source_catalog_config["expected_row_count"]):
        raise PipelineError("source parent catalog row count mismatch")
    source_id_field = str(source_catalog_config["id_field"])
    if not source_catalog_rows or source_id_field not in source_catalog_rows[0]:
        raise PipelineError("source parent catalog is missing its authoritative ID field")
    source_parent_ids = [row[source_id_field] for row in source_catalog_rows]
    if any(not value for value in source_parent_ids) or len(set(source_parent_ids)) != len(
        source_parent_ids
    ):
        raise PipelineError("source parent catalog IDs must be non-empty and unique")
    authoritative_source_ids = set(source_parent_ids)
    source_catalog_by_id = {row[source_id_field]: row for row in source_catalog_rows}

    split_config = config["source_split_assignments"]
    split_assignments_path = resolve_repository_path(
        repository_root, str(split_config["path"])
    )
    split_assignments_sha = sha256_file(split_assignments_path)
    if split_assignments_sha != split_config["expected_sha256"]:
        raise PipelineError(
            "source split assignments differ from the config-pinned authoritative digest"
        )
    split_header, split_rows = _read_csv_with_header(split_assignments_path)
    if split_header != split_config["required_columns"]:
        raise PipelineError("source split assignment CSV header/schema changed")
    if len(split_rows) != int(split_config["expected_row_count"]):
        raise PipelineError("source split assignment row count mismatch")
    split_ids = [row["sample_id"] for row in split_rows]
    if any(not value for value in split_ids) or len(set(split_ids)) != len(split_ids):
        raise PipelineError("source split assignment sample IDs must be non-empty and unique")
    if set(split_ids) != authoritative_source_ids:
        raise PipelineError("source split assignments do not cover the authoritative v2 catalog exactly")
    split_by_id: dict[str, dict[str, str]] = {}
    model_split_counts: Counter[str] = Counter()
    catalog_identity_fields = (
        "image_path",
        "primary_class",
        "severity",
        "sample_seed",
        "image_sha256",
        "base_group_id",
        "source_specimen_group",
    )
    for row in split_rows:
        sample_id = row["sample_id"]
        catalog_row = source_catalog_by_id[sample_id]
        if any(row[field] != catalog_row[field] for field in catalog_identity_fields):
            raise PipelineError(
                f"source split assignment identity differs from v2 catalog: {sample_id}"
            )
        split_name = row["split"]
        model_split = row["model_split"]
        if model_split not in {"gradient_train", "validation", "test"}:
            raise PipelineError(f"unknown source model_split for {sample_id}: {model_split}")
        if (model_split in {"gradient_train", "validation"} and split_name != "train") or (
            model_split == "test" and split_name != "test"
        ):
            raise PipelineError(f"source split/model_split contract mismatch: {sample_id}")
        _parse_int(row["class_severity_rank"], f"{sample_id} class_severity_rank")
        _parse_int(row["sample_seed"], f"{sample_id} sample_seed")
        for digest_field in ("split_key_sha256", "image_sha256"):
            if len(row[digest_field]) != 64:
                raise PipelineError(f"malformed {digest_field} in source split: {sample_id}")
        validation_key = row["validation_key_sha256"]
        if (model_split == "test" and validation_key) or (
            model_split != "test" and len(validation_key) != 64
        ):
            raise PipelineError(f"validation key contract mismatch in source split: {sample_id}")
        model_split_counts[model_split] += 1
        split_by_id[sample_id] = row
    if dict(model_split_counts) != split_config["expected_model_split_counts"]:
        raise PipelineError("source split assignment model_split counts mismatch")

    all_records: list[SceneRecord] = []
    release_scene_sha: dict[str, dict[str, str]] = {}
    release_parents: dict[str, dict[str, tuple[str, ...]]] = {}
    release_hashes: dict[str, dict[str, str]] = {}
    release_id_sets: dict[str, set[int]] = {}
    for release_config in config["releases"]:
        key = str(release_config["key"])
        records, scene_sha, parents, hashes = _validate_release(
            release_config=release_config,
            repository_root=repository_root,
            next_index=len(all_records),
        )
        all_records.extend(records)
        release_scene_sha[key] = scene_sha
        release_parents[key] = parents
        release_hashes[key] = hashes
        release_id_sets[key] = {record.release_image_id for record in records}

    if len(all_records) != 1152:
        raise PipelineError(f"combined scene count must be 1152, got {len(all_records)}")
    for record in all_records:
        if len(set(record.source_parent_ids)) != 5:
            raise PipelineError(f"scene {record.scene_id} must reference five distinct source parents")
        unknown_parents = set(record.source_parent_ids) - authoritative_source_ids
        if unknown_parents:
            raise PipelineError(
                f"scene {record.scene_id} references parents absent from authoritative v2 catalog"
            )
    referenced_parent_ids = {
        parent_id for record in all_records for parent_id in record.source_parent_ids
    }
    v4_referenced_parent_ids = {
        parent_id
        for record in all_records
        if record.release_key == "v4"
        for parent_id in record.source_parent_ids
    }
    v5_referenced_parent_ids = {
        parent_id
        for record in all_records
        if record.release_key == "v5"
        for parent_id in record.source_parent_ids
    }
    expected_referenced_count = int(split_config["expected_referenced_parent_count"])
    if (
        len(referenced_parent_ids) != expected_referenced_count
        or v4_referenced_parent_ids != referenced_parent_ids
        or v5_referenced_parent_ids != referenced_parent_ids
    ):
        raise PipelineError(
            "v4/v5 must reference the same exact set of 168 authoritative source parents"
        )
    referenced_model_split_counts = Counter(
        split_by_id[parent_id]["model_split"] for parent_id in referenced_parent_ids
    )
    required_parent_split = split_config["required_referenced_parent_model_split"]
    if referenced_model_split_counts != {required_parent_split: expected_referenced_count}:
        raise PipelineError(
            "every referenced v4/v5 source parent must belong exclusively to gradient_train"
        )
    if (
        referenced_model_split_counts.get("validation", 0)
        != int(split_config["expected_referenced_validation_count"])
        or referenced_model_split_counts.get("test", 0)
        != int(split_config["expected_referenced_test_count"])
    ):
        raise PipelineError("referenced source parents include validation or test samples")
    referenced_parent_fingerprint = sha256_lines(sorted(referenced_parent_ids))
    scoped_keys = {(record.release_key, record.release_image_id) for record in all_records}
    if len(scoped_keys) != len(all_records):
        raise PipelineError("release-scoped (release_key, image_id) keys are not unique")
    overlapping_ids = release_id_sets["v4"] & release_id_sets["v5"]
    if overlapping_ids != set(range(1, 385)):
        raise PipelineError("v4/v5 local image-ID restart contract changed unexpectedly")
    scene_ids = [record.scene_id for record in all_records]
    if len(set(scene_ids)) != len(scene_ids):
        raise PipelineError("scene_id must remain globally unique even though COCO IDs restart")

    family_members: dict[str, list[SceneRecord]] = defaultdict(list)
    for record in all_records:
        family_members[record.family_id].append(record)
    if len(family_members) != 384:
        raise PipelineError(f"composition family count must be 384, got {len(family_members)}")
    for family_id, members in family_members.items():
        variant_counts = Counter(member.variant_name for member in members)
        if variant_counts != {"v4": 1, "v5_0": 1, "v5_1": 1}:
            raise PipelineError(f"family {family_id} must contain v4 + v5_0 + v5_1 exactly once")
        v4_record = next(member for member in members if member.release_key == "v4")
        for v5_record in (member for member in members if member.release_key == "v5"):
            v5_row_parents = release_parents["v5"][v5_record.scene_id]
            if v5_row_parents != v4_record.source_parent_ids:
                raise PipelineError(f"family {family_id} v5 source parents differ from v4")
            if (
                v5_record.source_scene_id != v4_record.scene_id
                or v5_record.source_image_path != v4_record.image_path
                or v5_record.source_image_sha256 != v4_record.image_sha256
            ):
                raise PipelineError(f"family {family_id} v5 source image lineage differs from v4")
            if v5_record.annotations != v4_record.annotations:
                raise PipelineError(
                    f"family {family_id} v5 component bbox/status targets do not exactly replay v4"
                )

    # Prove that the scene/source-parent graph is one connected component.  This
    # is the reason no internal validation or test partition is offered.
    adjacency: dict[str, set[str]] = defaultdict(set)
    for record in all_records:
        scene_node = f"scene:{record.scene_id}"
        if record.release_key == "v4":
            for parent_id in record.source_parent_ids:
                parent_node = f"source:{parent_id}"
                adjacency[scene_node].add(parent_node)
                adjacency[parent_node].add(scene_node)
        else:
            source_node = f"scene:{record.family_id}"
            adjacency[scene_node].add(source_node)
            adjacency[source_node].add(scene_node)
    component_count = _connected_component_count(adjacency)
    if component_count != 1:
        raise PipelineError(
            f"source-parent graph must be one connected component, got {component_count}"
        )

    family_to_indices = {
        family_id: tuple(
            member.index
            for member in sorted(
                members,
                key=lambda item: {"v4": 0, "v5_0": 1, "v5_1": 2}[item.variant_name],
            )
        )
        for family_id, members in sorted(family_members.items())
    }
    sampler_contract = FamilyVariantSampler(
        family_to_indices, seed=int(config["training"]["augmentation_seed"])
    )
    selections_by_epoch: list[set[int]] = []
    for epoch in range(3):
        sampler_contract.set_epoch(epoch)
        selection = list(sampler_contract)
        if len(selection) != 384 or len(set(selection)) != 384:
            raise PipelineError("FamilyVariantSampler must select 384 unique families per epoch")
        selections_by_epoch.append(set(selection))
    for family_id, indices in family_to_indices.items():
        chosen = {
            next(index for index in indices if index in epoch_selection)
            for epoch_selection in selections_by_epoch
        }
        if chosen != set(indices):
            raise PipelineError(
                f"FamilyVariantSampler does not cycle all three variants for family {family_id}"
            )
    data_fingerprint = sha256_lines(
        [
            f"{key}|{name}|{value}"
            for key in sorted(release_hashes)
            for name, value in sorted(release_hashes[key].items())
        ]
        + [
            f"source_parent_catalog|{source_catalog_sha}",
            f"source_split_assignments|{split_assignments_sha}",
            f"referenced_gradient_train_parents|{referenced_parent_fingerprint}",
        ]
        + [
            f"scene|{record.release_key}|{record.scene_id}|{record.family_id}|{record.image_sha256}"
            for record in sorted(all_records, key=lambda item: (item.release_key, item.release_image_id))
        ]
    )
    fingerprints: dict[str, Any] = {
        "config_sha256": sha256_file(resolved_config_path),
        "official_weight_filename": weight_path.name,
        "official_weight_sha256": weight_sha,
        "data_fingerprint_sha256": data_fingerprint,
        "source_parent_catalog_sha256": source_catalog_sha,
        "source_split_assignments_sha256": split_assignments_sha,
        "referenced_gradient_train_parent_ids_sha256": referenced_parent_fingerprint,
        "release_files": release_hashes,
    }
    summary = {
        "status": "PASS",
        "scope": "TRAIN_ONLY",
        "metric_scope": "TRAIN_DIAGNOSTIC_ONLY",
        "scene_count": len(all_records),
        "component_instance_count": sum(len(record.annotations) for record in all_records),
        "release_scene_counts": dict(Counter(record.release_key for record in all_records)),
        "composition_family_count": len(family_to_indices),
        "variants_per_family": 3,
        "samples_per_epoch": len(family_to_indices),
        "source_parent_graph_connected_components": component_count,
        "authoritative_source_model_split_counts": dict(sorted(model_split_counts.items())),
        "referenced_unique_source_parent_count": len(referenced_parent_ids),
        "referenced_parent_model_split_counts": dict(
            sorted(referenced_model_split_counts.items())
        ),
        "referenced_parent_validation_count": referenced_model_split_counts.get(
            "validation", 0
        ),
        "referenced_parent_test_count": referenced_model_split_counts.get("test", 0),
        "referenced_parents_all_gradient_train": True,
        "v4_release_source_split_assignment_pin_verified": True,
        "sampler_three_epoch_cycle_verified": True,
        "release_scoped_image_id_overlap_count": len(overlapping_ids),
        "validation_created": False,
        "test_created": False,
        "performance_claim_permitted": False,
    }
    return PreflightResult(
        config=config,
        config_path=resolved_config_path,
        repository_root=repository_root,
        weight_path=weight_path,
        scenes=tuple(all_records),
        family_to_indices=family_to_indices,
        fingerprints=fingerprints,
        summary=summary,
    )


class FamilyVariantSampler:
    """Choose one of v4/v5_0/v5_1 per family, cycling every epoch."""

    def __init__(self, family_to_indices: dict[str, tuple[int, ...]], seed: int) -> None:
        self.family_to_indices = dict(family_to_indices)
        self.seed = int(seed)
        self.epoch = 0
        if not self.family_to_indices:
            raise PipelineError("FamilyVariantSampler requires at least one family")
        for family_id, indices in self.family_to_indices.items():
            if len(indices) != 3:
                raise PipelineError(f"family {family_id} does not have three rotating variants")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def selected_variant_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for family_id, indices in self.family_to_indices.items():
            offset = int.from_bytes(
                hashlib.sha256(f"{self.seed}|{family_id}".encode("utf-8")).digest()[:8],
                "big",
            ) % len(indices)
            selected_position = (offset + self.epoch) % len(indices)
            counts[("v4", "v5_0", "v5_1")[selected_position]] += 1
        return dict(sorted(counts.items()))

    def __iter__(self) -> Iterator[int]:
        selected: list[int] = []
        for family_id, indices in self.family_to_indices.items():
            offset = int.from_bytes(
                hashlib.sha256(f"{self.seed}|{family_id}".encode("utf-8")).digest()[:8],
                "big",
            ) % len(indices)
            selected.append(indices[(offset + self.epoch) % len(indices)])
        shuffler = random.Random(self.seed + 1_000_003 * self.epoch)
        shuffler.shuffle(selected)
        return iter(selected)

    def __len__(self) -> int:
        return len(self.family_to_indices)


class ComponentDetectionDataset:
    """Torch DataLoader-compatible dataset with release-scoped scene records."""

    def __init__(
        self,
        scenes: Sequence[SceneRecord],
        task: str,
        transforms: Any | None,
    ) -> None:
        if task not in SUPPORTED_TASKS:
            raise PipelineError(f"unsupported detector task: {task}")
        self.scenes = tuple(scenes)
        self.task = task
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        try:
            import torch
            from PIL import Image
            from torchvision import tv_tensors
            from torchvision.transforms.functional import pil_to_tensor
        except ImportError as error:
            raise PipelineError(
                "training requires the Codex bundled Python runtime with torch, torchvision, and Pillow"
            ) from error

        scene = self.scenes[index]
        with Image.open(scene.image_absolute_path) as opened:
            rgb = opened.convert("RGB")
            image = tv_tensors.Image(pil_to_tensor(rgb))
        boxes = tv_tensors.BoundingBoxes(
            torch.tensor([annotation.bbox_xyxy for annotation in scene.annotations], dtype=torch.float32),
            format="XYXY",
            canvas_size=(scene.height, scene.width),
        )
        if self.task == "component_localization":
            labels = torch.ones((len(scene.annotations),), dtype=torch.int64)
        else:
            labels = torch.tensor(
                [annotation.status_category_id for annotation in scene.annotations], dtype=torch.int64
            )
        target: dict[str, Any] = {"boxes": boxes, "labels": labels}
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        target["image_id"] = torch.tensor([scene.index], dtype=torch.int64)
        return image, target


def build_train_transforms(config: dict[str, Any]) -> Any:
    try:
        import torch
        from torchvision.transforms import InterpolationMode, v2
    except ImportError as error:
        raise PipelineError(
            "training requires the Codex bundled Python runtime with torch and torchvision"
        ) from error
    augmentation = config["training"]["augmentation"]
    scale = tuple(float(value) for value in augmentation["scale_range"])
    sigma = tuple(float(value) for value in augmentation["gaussian_blur_sigma"])
    return v2.Compose(
        [
            v2.RandomHorizontalFlip(p=float(augmentation["horizontal_flip_probability"])),
            v2.RandomApply(
                [
                    v2.RandomAffine(
                        degrees=float(augmentation["rotation_degrees"]),
                        translate=(
                            float(augmentation["translation_fraction"]),
                            float(augmentation["translation_fraction"]),
                        ),
                        scale=scale,
                        interpolation=InterpolationMode.BILINEAR,
                        fill=(0, 0, 0),
                    )
                ],
                p=float(augmentation["affine_probability"]),
            ),
            v2.RandomApply(
                [
                    v2.ColorJitter(
                        brightness=float(augmentation["brightness"]),
                        contrast=float(augmentation["contrast"]),
                        saturation=float(augmentation["saturation"]),
                        hue=float(augmentation["hue"]),
                    )
                ],
                p=float(augmentation["color_jitter_probability"]),
            ),
            v2.RandomApply(
                [
                    v2.GaussianBlur(
                        kernel_size=int(augmentation["gaussian_blur_kernel"]), sigma=sigma
                    )
                ],
                p=float(augmentation["gaussian_blur_probability"]),
            ),
            v2.SanitizeBoundingBoxes(min_size=2.0, min_area=4.0),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def build_plain_transform() -> Any:
    try:
        import torch
        from torchvision.transforms import v2
    except ImportError as error:
        raise PipelineError(
            "training requires the Codex bundled Python runtime with torch and torchvision"
        ) from error
    return v2.Compose([v2.ToDtype(torch.float32, scale=True)])


def detection_collate_fn(batch: Sequence[tuple[Any, dict[str, Any]]]) -> tuple[list[Any], list[dict[str, Any]]]:
    images, targets = zip(*batch)
    return list(images), list(targets)


def runtime_environment(*, device: str | None = None) -> dict[str, Any]:
    environment: dict[str, Any] = {
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import PIL
        import torch
        import torchvision

        environment.update(
            {
                "torch_version": torch.__version__,
                "torchvision_version": torchvision.__version__,
                "pillow_version": PIL.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
        if torch.cuda.is_available():
            environment["cuda_device_count"] = torch.cuda.device_count()
            environment["cuda_device_name_0"] = torch.cuda.get_device_name(0)
    except ImportError:
        environment["training_runtime_available"] = False
    if device is not None:
        environment["selected_device"] = device
    return environment


def code_fingerprints(repository_root: Path) -> dict[str, str]:
    paths = (
        "training/scripts/detection_common.py",
        "training/scripts/train_eval_detector.py",
    )
    values: dict[str, str] = {}
    for relative in paths:
        path = resolve_repository_path(repository_root, relative)
        if not path.is_file():
            raise PipelineError(f"pipeline code file not found: {relative}")
        values[relative] = sha256_file(path)
    values["combined_sha256"] = sha256_lines(
        f"{relative}|{values[relative]}" for relative in paths
    )
    return values
