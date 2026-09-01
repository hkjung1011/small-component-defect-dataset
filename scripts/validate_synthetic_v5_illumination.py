"""Independently validate the synthetic-v5 illumination auxiliary release.

The validator treats every published annotation and metric as untrusted.  It
pins the v5 config/generator/runtime and every v4 source file, verifies the
transitive v2 gradient-train lineage, checks exact schemas and balance, proves
that no synthetic value is represented as measured lux, compares all inherited
geometry byte-for-byte, and fully replays all 768 published variants (two per
v4 composition).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import io
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont, features


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "synthetic_v5_illumination.json"
DEFAULT_RELEASE = ROOT / "synthetic" / "v5_illumination"
EXPECTED_CONFIG_SHA256 = "4ad0c3beb74fb766d6702b71cf1475fd6f9d1d62d9132d9f8320f1ca689057eb"
EXPECTED_GENERATOR_SHA256 = "60603aad54df4e82a6bf12b9a0160f5af51baaae1d39086116301c5f190717b7"
EXPECTED_REQUIREMENTS_SHA256 = "445952014c05a210088d847236be8fab262f66d28109ff8abb6d168d157c2b21"
EXPECTED_QC_STATUS = "AUTO_PASS_MULTI_LIGHT_LUX_PROXY_REPLAY"
EXPECTED_MARKER = ".synthetic_v5_illumination_marker"
EXPECTED_MARKER_TEXT = "synthetic-v5-illumination\n"
FLOAT_TOLERANCE = 1.1e-6

CLASSES = [
    "normal_proxy",
    "scratch",
    "surface_spot",
    "discoloration",
    "contamination",
    "lead_breakage",
    "body_chip",
    "body_crack",
]

CONFIG_FIELDS = {
    "release",
    "sample_id_prefix",
    "generator_version",
    "qc_gate_version",
    "runtime_contract",
    "task_type",
    "global_seed",
    "scene_count",
    "variants_per_source_scene",
    "instances_per_scene",
    "split",
    "training_use",
    "evaluation_eligible",
    "classification_eligible",
    "photometry_domain",
    "absolute_lux_eligible",
    "measured_illuminance_lux",
    "photometric_calibration_status",
    "coordinate_frame",
    "classes",
    "source",
    "target_illuminance_bins",
    "multi_light_rigs",
    "shadow_regimes",
    "rendering",
    "balance_contract",
    "qc",
}

MANIFEST_FIELDS = [
    "scene_id",
    "image_id",
    "image_path",
    "component_instance_mask_path",
    "defect_semantic_mask_path",
    "shadow_attenuation_mask_path",
    "component_yolo_path",
    "defect_yolo_path",
    "source_scene_id",
    "source_variant_index",
    "source_image_path",
    "pixel_base_method",
    "source_lighting_profile",
    "source_image_sha256",
    "source_component_mask_sha256",
    "source_defect_mask_sha256",
    "split",
    "task_type",
    "training_use",
    "evaluation_eligible",
    "classification_eligible",
    "photometry_domain",
    "absolute_lux_eligible",
    "measured_illuminance_lux",
    "photometric_calibration_status",
    "synthetic_illuminance_proxy_bin",
    "capture_plan_target_lux",
    "synthetic_relative_light_power",
    "relative_light_power_unit",
    "calibrated_to_lux",
    "multi_light_rig_id",
    "shadow_regime_id",
    "condition_cell_id",
    "condition_cell_index",
    "light_source_count",
    "coordinate_frame_name",
    "scene_seed",
    "attempt",
    "instance_count",
    "component_status_labels",
    "source_parent_ids",
    "composition_family_id",
    "width",
    "height",
    "image_sha256",
    "component_mask_sha256",
    "defect_mask_sha256",
    "shadow_mask_sha256",
    "component_yolo_sha256",
    "defect_yolo_sha256",
    "background_mean_luma",
    "background_p99_luma",
    "component_mean_luma",
    "component_dark_fraction",
    "component_saturated_fraction",
    "pre_sensor_positive_spill_max",
    "post_jpeg_paired_clean_spill_p99",
    "post_jpeg_paired_clean_spill_max",
    "post_jpeg_paired_clean_spill_fraction",
    "post_jpeg_paired_clean_spill_energy",
    "minimum_defect_mean_abs_delta",
    "minimum_defect_changed_fraction",
    "shadow_nonzero_fraction",
    "shadow_max_attenuation",
    "config_sha256",
    "generator_version",
    "qc_gate_version",
    "qc_status",
    "human_verified",
    "sensor_params_json",
    "light_sources_json",
    "cast_shadow_params_json",
]

LIGHTING_SCENE_FIELDS = {
    "lighting_scene_id",
    "scene_id",
    "source_scene_id",
    "source_variant_index",
    "pixel_base_method",
    "condition_cell_index",
    "condition_cell_id",
    "photometry_domain",
    "absolute_lux_eligible",
    "measured_illuminance_lux",
    "photometric_calibration_status",
    "synthetic_illuminance_proxy_bin",
    "capture_plan_target_lux",
    "synthetic_lux_proxy",
    "multi_light_rig_id",
    "shadow_regime_id",
    "source_count",
    "coordinate_frame",
    "sensor_params",
    "cast_shadow_rows",
    "paired_clean_defect_visibility",
    "metrics",
}

LIGHT_SOURCE_FIELDS = {
    "lighting_scene_id",
    "light_id",
    "multi_light_rig_id",
    "coordinate_frame_name",
    "image_plane_azimuth_deg",
    "elevation_proxy_deg",
    "direction_vector_image_xy",
    "relative_intensity",
    "relative_intensity_unit",
    "cct_proxy_kelvin",
    "cct_calibrated",
    "anchor_xy_fraction",
}

LIGHTING_EFFECT_FIELDS = {
    "instance_index",
    "source_rows",
    "base_relative_light_power",
    "directional_gain_strength",
    "hotspot_strength",
    "hotspot_sigma_fraction",
    "occluder_shadow_applied",
    "occluder_shadow_angle_deg",
    "occluder_shadow_strength",
    "occluder_shadow_width_fraction",
    "occluder_shadow_center",
    "gain_min",
    "gain_mean",
    "gain_max",
    "color_gain_rgb",
}

LIGHTING_EFFECT_SOURCE_FIELDS = {
    "light_id",
    "effective_weight",
    "image_plane_azimuth_deg",
    "local_azimuth_proxy_deg",
    "elevation_proxy_deg",
    "cct_proxy_kelvin",
}

DEFECT_VISIBILITY_FIELDS = {
    "area",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "major",
    "minor",
    "diag",
    "mean_abs_delta",
    "delta_e76_p50",
    "changed_fraction",
    "instance_index",
    "class_name",
    "semantic_id",
}

CAST_SHADOW_FIELDS = {
    "instance_index",
    "light_id",
    "shadow_kind",
    "shadow_direction_deg",
    "offset_x_px",
    "offset_y_px",
    "length_px",
    "blur_radius_px",
    "opacity",
}

RELEASE_FIELDS = {
    "release",
    "generator_version",
    "qc_gate_version",
    "task_type",
    "split",
    "training_use",
    "evaluation_eligible",
    "classification_eligible",
    "photometry_domain",
    "absolute_lux_eligible",
    "measured_illuminance_lux",
    "photometric_calibration_status",
    "config_path",
    "config_sha256",
    "generator_script",
    "generator_script_sha256",
    "runtime_contract",
    "source_release",
    "source_pins",
    "scene_count",
    "source_scene_count",
    "variants_per_source_scene",
    "composition_family_policy",
    "pixel_base_method",
    "instances_per_scene",
    "component_instance_count",
    "defect_instance_count",
    "class_counts",
    "synthetic_illuminance_proxy_bin_scene_counts",
    "multi_light_rig_scene_counts",
    "shadow_regime_scene_counts",
    "condition_cell_count",
    "condition_cell_scene_count_range",
    "class_condition_cell_count_range",
    "class_illuminance_proxy_bin_count_range",
    "class_rig_count_range",
    "class_rig_illuminance_proxy_count_range",
    "capture_plan_target_lux_by_proxy_bin",
    "manifest_sha256",
    "instances_sha256",
    "lighting_scenes_sha256",
    "light_sources_sha256",
    "component_coco_sha256",
    "defect_coco_sha256",
    "summary_sha256",
    "condition_matrix_sha256",
    "contact_sheet_sha256",
    "contact_sheet_overlay_sha256",
    "paired_condition_comparison_sha256",
    "limitations",
    "tracked_payload_bytes",
}


class ValidationSetupError(RuntimeError):
    """Raised when a trusted validation input cannot be established."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                raise ValidationSetupError(f"blank JSONL line: {path}:{line_number}")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValidationSetupError(
                    f"JSONL row is not an object: {path}:{line_number}"
                )
            rows.append(value)
    return rows


def repository_path(value: str, field: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValidationSetupError(
            f"{field} must be a non-empty repository-relative path: {value!r}"
        )
    candidate = (ROOT / value).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValidationSetupError(f"{field} escapes repository: {value}") from error
    return candidate


def release_path(release_root: Path, relative: str, field: str) -> Path:
    candidate = (release_root / relative).resolve()
    try:
        candidate.relative_to(release_root.resolve())
    except ValueError as error:
        raise ValidationSetupError(f"{field} escapes release: {relative}") from error
    return candidate


def require_pinned_file(
    block: dict[str, Any], path_key: str, sha_key: str, label: str
) -> Path:
    raw_path = block.get(path_key)
    expected = block.get(sha_key)
    if not isinstance(raw_path, str):
        raise ValidationSetupError(f"{label}.{path_key} is not a path string")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in expected)
    ):
        raise ValidationSetupError(f"{label}.{sha_key} is not SHA-256")
    path = repository_path(raw_path, f"{label}.{path_key}")
    if not path.is_file():
        raise ValidationSetupError(f"missing pinned file: {path}")
    actual = sha256_file(path)
    if actual != expected.lower():
        raise ValidationSetupError(
            f"pin mismatch {label}.{path_key}: expected={expected.lower()} actual={actual}"
        )
    return path


def integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} is boolean, not integer")
    try:
        result = int(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} is not integer: {value!r}") from error
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field} is not integral: {value!r}")
    return result


def finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} is not numeric: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} is not finite: {value!r}")
    return result


def same_float(
    actual: Any,
    expected: float,
    field: str,
    prefix: str,
    errors: list[str],
    tolerance: float = FLOAT_TOLERANCE,
) -> None:
    try:
        value = finite_float(actual, field)
    except ValueError as error:
        errors.append(f"{prefix}: {error}")
        return
    if not math.isclose(value, float(expected), rel_tol=0.0, abs_tol=tolerance):
        errors.append(
            f"{prefix}: {field} mismatch actual={value!r} expected={expected!r}"
        )


def parse_json_field(value: str, field: str, prefix: str, errors: list[str]) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        errors.append(f"{prefix}: invalid {field}: {error}")
        return None


def current_runtime() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "libjpeg": str(features.version("jpg")),
        "zlib": str(features.version("zlib")),
    }


def validate_runtime(config: dict[str, Any], errors: list[str]) -> None:
    contract = config.get("runtime_contract")
    if not isinstance(contract, dict):
        errors.append("config.runtime_contract is not an object")
        return
    expected = {
        key: str(contract.get(key))
        for key in ("python", "numpy", "pillow", "libjpeg", "zlib")
    }
    actual = current_runtime()
    if actual != expected:
        errors.append(f"runtime mismatch actual={actual} expected={expected}")
    if contract.get("requirements_path") != "requirements-synthetic.txt":
        errors.append("runtime requirements path mismatch")
    if contract.get("requirements_sha256") != EXPECTED_REQUIREMENTS_SHA256:
        errors.append("runtime requirements SHA contract mismatch")
    try:
        requirements = repository_path(
            str(contract.get("requirements_path")), "runtime requirements"
        )
        if not requirements.is_file() or sha256_file(requirements) != EXPECTED_REQUIREMENTS_SHA256:
            errors.append("runtime requirements file/hash mismatch")
    except ValidationSetupError as error:
        errors.append(str(error))


def validate_proxy_semantics(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else key
            lowered = key.lower()
            if "measured" in lowered and "lux" in lowered and item not in (None, ""):
                errors.append(f"{child}: measured lux must be null/empty")
            if key == "measured_illuminance_lux" and item not in (None, ""):
                errors.append(f"{child}: measured illuminance must be null/empty")
            if key == "calibrated_to_lux" and item not in (False, "NO"):
                errors.append(f"{child}: synthetic proxy must not be calibrated to lux")
            if key == "absolute_lux_eligible" and item != "NO":
                errors.append(f"{child}: absolute lux eligibility must be NO")
            validate_proxy_semantics(item, child, errors)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            validate_proxy_semantics(item, f"{path}[{index}]", errors)


def validate_config(config: dict[str, Any], errors: list[str]) -> None:
    if set(config) != CONFIG_FIELDS:
        errors.append(
            f"config field inventory mismatch missing={sorted(CONFIG_FIELDS-set(config))} "
            f"extra={sorted(set(config)-CONFIG_FIELDS)}"
        )
    exact = {
        "release": "synthetic-v5-illumination",
        "sample_id_prefix": "syn-v5-light",
        "generator_version": "5.0.0",
        "qc_gate_version": "paired-neutral-multi-light-proxy-v2",
        "task_type": "multi_instance_detection_segmentation_lighting_auxiliary",
        "scene_count": 768,
        "variants_per_source_scene": 2,
        "instances_per_scene": 5,
        "split": "train",
        "training_use": "TRAIN_ONLY_MULTI_LIGHT_AUXILIARY",
        "evaluation_eligible": "NO",
        "classification_eligible": "NO",
        "photometry_domain": "SYNTHETIC_PROXY",
        "absolute_lux_eligible": "NO",
        "measured_illuminance_lux": None,
        "photometric_calibration_status": "NOT_CALIBRATED_SYNTHETIC_PROXY",
    }
    for field, expected in exact.items():
        if config.get(field) != expected:
            errors.append(
                f"config.{field} mismatch expected={expected!r} actual={config.get(field)!r}"
            )
    if config.get("classes") != CLASSES:
        errors.append("config class order mismatch")
    coordinate = config.get("coordinate_frame")
    expected_coordinate = {
        "name": "image_xy_clockwise_degrees",
        "origin": "image_top_left",
        "positive_x": "image_right",
        "positive_y": "image_down",
        "azimuth_zero_degrees": "positive_x",
        "azimuth_positive_direction": "clockwise",
        "elevation_semantics": "dimensionless_2d_rendering_proxy_not_physical_3d_angle",
    }
    if canonical_json(coordinate) != canonical_json(expected_coordinate):
        errors.append("config coordinate-frame contract mismatch")
    bins = config.get("target_illuminance_bins")
    if not isinstance(bins, list) or len(bins) != 6:
        errors.append("config must have exactly six target illuminance bins")
        bins = []
    expected_bin_ids = ["P0", "P1", "P2", "P3", "P4", "P5"]
    if [item.get("id") for item in bins if isinstance(item, dict)] != expected_bin_ids:
        errors.append("target illuminance bin order/inventory mismatch")
    for index, item in enumerate(bins):
        prefix = f"config.target_illuminance_bins[{index}]"
        expected_fields = {
            "id",
            "capture_plan_target_lux",
            "relative_light_power",
            "relative_light_power_unit",
            "calibrated_to_lux",
            "synthetic_component_exposure_ev",
            "sensor_noise_sigma",
            "sensor_blur_radius_px",
            "jpeg_quality",
        }
        if not isinstance(item, dict) or set(item) != expected_fields:
            errors.append(f"{prefix}: field inventory mismatch")
            continue
        if item.get("calibrated_to_lux") is not False:
            errors.append(f"{prefix}: calibrated_to_lux must be false")
        if item.get("relative_light_power_unit") != "1":
            errors.append(f"{prefix}: relative light power unit must be dimensionless")
        try:
            if finite_float(item["relative_light_power"], "relative light power") <= 0:
                errors.append(f"{prefix}: relative light power must be positive")
            if integer(item["capture_plan_target_lux"], "capture-plan target") <= 0:
                errors.append(f"{prefix}: capture target bin must be positive")
            for range_field in ("sensor_noise_sigma", "sensor_blur_radius_px", "jpeg_quality"):
                bounds = item[range_field]
                if not isinstance(bounds, list) or len(bounds) != 2:
                    errors.append(f"{prefix}.{range_field}: expected two bounds")
                elif finite_float(bounds[0], range_field) > finite_float(bounds[1], range_field):
                    errors.append(f"{prefix}.{range_field}: reversed bounds")
        except (ValueError, KeyError) as error:
            errors.append(f"{prefix}: {error}")
    rigs = config.get("multi_light_rigs")
    if not isinstance(rigs, list) or len(rigs) != 4:
        errors.append("config must have exactly four multi-light rigs")
        rigs = []
    rig_ids: set[str] = set()
    for index, rig in enumerate(rigs):
        prefix = f"config.multi_light_rigs[{index}]"
        if not isinstance(rig, dict) or set(rig) != {"id", "sources"}:
            errors.append(f"{prefix}: rig schema mismatch")
            continue
        rig_id = rig.get("id")
        if not isinstance(rig_id, str) or not rig_id or rig_id in rig_ids:
            errors.append(f"{prefix}: invalid/duplicate rig id")
        rig_ids.add(str(rig_id))
        sources = rig.get("sources")
        if not isinstance(sources, list) or len(sources) not in (2, 3):
            errors.append(f"{prefix}: source count must be two or three")
            continue
        light_ids: set[str] = set()
        for source_index, source in enumerate(sources):
            source_prefix = f"{prefix}.sources[{source_index}]"
            expected_fields = {
                "light_id",
                "image_plane_azimuth_deg",
                "elevation_proxy_deg",
                "relative_intensity",
                "cct_proxy_kelvin",
                "anchor_xy_fraction",
            }
            if not isinstance(source, dict) or set(source) != expected_fields:
                errors.append(f"{source_prefix}: source schema mismatch")
                continue
            light_id = source.get("light_id")
            if not isinstance(light_id, str) or not light_id or light_id in light_ids:
                errors.append(f"{source_prefix}: invalid/duplicate light id")
            light_ids.add(str(light_id))
            try:
                azimuth = finite_float(source["image_plane_azimuth_deg"], "azimuth")
                elevation = finite_float(source["elevation_proxy_deg"], "elevation")
                intensity = finite_float(source["relative_intensity"], "intensity")
                cct = integer(source["cct_proxy_kelvin"], "CCT proxy")
                anchor = source["anchor_xy_fraction"]
                if not 0.0 <= azimuth < 360.0:
                    errors.append(f"{source_prefix}: azimuth outside [0,360)")
                if not 0.0 < elevation <= 90.0:
                    errors.append(f"{source_prefix}: elevation proxy outside (0,90]")
                if intensity <= 0.0 or cct <= 0:
                    errors.append(f"{source_prefix}: non-positive intensity/CCT proxy")
                if not isinstance(anchor, list) or len(anchor) != 2:
                    errors.append(f"{source_prefix}: invalid anchor")
                else:
                    finite_float(anchor[0], "anchor x")
                    finite_float(anchor[1], "anchor y")
            except (ValueError, KeyError) as error:
                errors.append(f"{source_prefix}: {error}")
    shadows = config.get("shadow_regimes")
    if not isinstance(shadows, list) or len(shadows) != 2:
        errors.append("config must have exactly two shadow regimes")
    elif len({item.get("id") for item in shadows if isinstance(item, dict)}) != 2:
        errors.append("config shadow-regime IDs are not unique")
    rendering = config.get("rendering")
    if not isinstance(rendering, dict):
        errors.append("config.rendering is not an object")
    else:
        if rendering.get("model") != "deterministic_2d_image_plane_multi_source_proxy":
            errors.append("config rendering model mismatch")
        if rendering.get("component_only_positive_light") is not True:
            errors.append("config must constrain positive light to components")
        if rendering.get("belt_positive_light_enabled") is not False:
            errors.append("config must disable positive-light gain on the belt")
    expected_balance = {
        "scene_per_illuminance_bin": 128,
        "scene_per_rig": 192,
        "scene_per_rig_illuminance_cell": 32,
        "scene_per_shadow_regime": 384,
        "scene_per_full_condition_cell": 16,
        "source_profile_scene_per_full_condition_cell": 4,
        "source_scene_reuse": 2,
        "instance_per_class": 480,
        "instance_per_class_illuminance_bin_target": 80,
        "instance_per_class_illuminance_bin_range": [79, 81],
        "instance_per_class_rig_target": 120,
        "instance_per_class_rig_range": [119, 121],
        "instance_per_class_shadow_regime": 240,
        "instance_per_class_rig_illuminance_cell_target": 20,
        "instance_per_class_rig_illuminance_cell_range": [19, 21],
        "instance_per_class_full_condition_cell_target": 10,
        "instance_per_class_full_condition_cell_range": [9, 11],
    }
    if config.get("balance_contract") != expected_balance:
        errors.append("config exact balance contract mismatch")
    validate_proxy_semantics(config, "config", errors)


def load_source_context(
    config: dict[str, Any], errors: list[str]
) -> dict[str, Any]:
    source = config.get("source")
    if not isinstance(source, dict):
        raise ValidationSetupError("config.source is not an object")
    pin_keys = {
        "config": ("config_path", "config_sha256"),
        "generator": ("generator_path", "generator_sha256"),
        "manifest": ("manifest_path", "manifest_sha256"),
        "instances": ("instances_path", "instances_sha256"),
        "component_coco": ("component_coco_path", "component_coco_sha256"),
        "defect_coco": ("defect_coco_path", "defect_coco_sha256"),
        "release": ("release_metadata_path", "release_metadata_sha256"),
    }
    paths = {
        name: require_pinned_file(source, path_key, sha_key, f"source.{name}")
        for name, (path_key, sha_key) in pin_keys.items()
    }
    expected_source_static = {
        "release": "synthetic-v4-conveyor",
        "release_root": "synthetic/v4_conveyor",
        "required_split": "train",
        "required_parent_model_split": "gradient_train",
        "expected_scene_count": 384,
        "expected_instance_count": 1920,
        "expected_source_lighting_profiles": [
            "neutral_component_spot",
            "warm_component_spot",
            "cool_component_spot",
            "side_component_spot",
        ],
    }
    for field, expected in expected_source_static.items():
        if source.get(field) != expected:
            errors.append(f"source.{field} contract mismatch")
    v4_config = load_json(paths["config"])
    v4_release = load_json(paths["release"])
    manifest_fields, v4_scenes = read_csv(paths["manifest"])
    v4_instances = read_jsonl(paths["instances"])
    v4_component_coco = load_json(paths["component_coco"])
    v4_defect_coco = load_json(paths["defect_coco"])
    if len(v4_scenes) != 384 or len(v4_instances) != 1920:
        errors.append(
            f"source inventory mismatch scenes={len(v4_scenes)} instances={len(v4_instances)}"
        )
    if len({row.get("scene_id") for row in v4_scenes}) != len(v4_scenes):
        errors.append("source manifest has duplicate scene IDs")
    if v4_release.get("release") != source.get("release"):
        errors.append("source release metadata name mismatch")
    if v4_release.get("manifest_sha256") != source.get("manifest_sha256"):
        errors.append("source release metadata/manifest SHA mismatch")
    if v4_release.get("instances_sha256") != source.get("instances_sha256"):
        errors.append("source release metadata/instances SHA mismatch")

    # Establish the transitive v4 -> v2 fixed split allowlist independently.
    v4_source = v4_config.get("source")
    if not isinstance(v4_source, dict):
        raise ValidationSetupError("pinned v4 config source is not an object")
    v2_manifest_path = require_pinned_file(
        v4_source, "manifest_path", "manifest_sha256", "v4.source.manifest"
    )
    require_pinned_file(v4_source, "config_path", "config_sha256", "v4.source.config")
    split_path = require_pinned_file(
        v4_source,
        "split_assignments_path",
        "split_assignments_sha256",
        "v4.source.split",
    )
    _, v2_rows_list = read_csv(v2_manifest_path)
    _, split_rows_list = read_csv(split_path)
    v2_rows = {row["sample_id"]: row for row in v2_rows_list}
    split_rows = {row["sample_id"]: row for row in split_rows_list}
    if len(v2_rows) != len(v2_rows_list) or len(split_rows) != len(split_rows_list):
        errors.append("transitive v2 manifest/split contains duplicate sample IDs")
    if set(v2_rows) != set(split_rows):
        errors.append("transitive v2 manifest/split inventories differ")
    gradient_ids = {
        sample_id
        for sample_id, row in split_rows.items()
        if row.get("model_split") == v4_source.get("required_parent_model_split")
    }
    forbidden_ids = set(split_rows) - gradient_ids
    if len(gradient_ids) != 168:
        errors.append(f"transitive gradient-train parent count mismatch: {len(gradient_ids)}")
    gradient_class_counts = Counter(v2_rows[item]["primary_class"] for item in gradient_ids)
    if len(gradient_class_counts) != 7 or set(gradient_class_counts.values()) != {24}:
        errors.append(
            f"transitive gradient parent class balance mismatch: {gradient_class_counts}"
        )

    parent_usage: Counter[str] = Counter()
    for index, instance in enumerate(v4_instances):
        prefix = f"source instance {index}"
        parent_id = instance.get("source_parent_sample_id")
        if parent_id not in gradient_ids:
            if parent_id in forbidden_ids:
                errors.append(f"{prefix}: validation/test parent leaked: {parent_id}")
            else:
                errors.append(f"{prefix}: unknown transitive parent: {parent_id}")
            continue
        parent_usage[parent_id] += 1
        source_row = v2_rows[parent_id]
        expected = {
            "source_parent_model_split": "gradient_train",
            "source_parent_class": source_row["primary_class"],
            "source_parent_severity": source_row["severity"],
            "source_parent_image_path": source_row["image_path"],
            "source_parent_mask_path": source_row["mask_path"],
            "source_parent_image_sha256": source_row["image_sha256"],
            "source_parent_mask_sha256": source_row["mask_sha256"],
            "family_split_id": parent_id,
            "evaluation_eligible": "NO",
            "classification_eligible": "NO",
        }
        for field, expected_value in expected.items():
            if instance.get(field) != expected_value:
                errors.append(f"{prefix}: transitive field mismatch {field}")
    if set(parent_usage) != gradient_ids:
        errors.append("source transitive parent inventory differs from gradient allowlist")
    if set(parent_usage) & forbidden_ids:
        errors.append("source contains forbidden validation/test parent")

    return {
        "paths": paths,
        "v4_config": v4_config,
        "v4_release": v4_release,
        "manifest_fields": manifest_fields,
        "scenes": v4_scenes,
        "instances": v4_instances,
        "component_coco": v4_component_coco,
        "defect_coco": v4_defect_coco,
        "v2_rows": v2_rows,
        "split_rows": split_rows,
        "gradient_ids": gradient_ids,
        "forbidden_ids": forbidden_ids,
        "parent_usage": parent_usage,
    }


def expected_keys(counter: Counter[Any], keys: Iterable[Any], value: int, label: str, errors: list[str]) -> None:
    expected_set = set(keys)
    if set(counter) != expected_set:
        errors.append(
            f"{label} key inventory mismatch missing={len(expected_set-set(counter))} "
            f"extra={len(set(counter)-expected_set)}"
        )
    wrong = [(key, counter[key]) for key in expected_set if counter[key] != value]
    if wrong:
        errors.append(f"{label} count mismatch examples={wrong[:8]} expected={value}")


def expected_keys_range(
    counter: Counter[Any],
    keys: Iterable[Any],
    bounds: Iterable[int],
    label: str,
    errors: list[str],
) -> None:
    expected_set = set(keys)
    if set(counter) != expected_set:
        errors.append(
            f"{label} key inventory mismatch missing={len(expected_set-set(counter))} "
            f"extra={len(set(counter)-expected_set)}"
        )
    low, high = [int(value) for value in bounds]
    wrong = [
        (key, counter[key])
        for key in expected_set
        if not low <= counter[key] <= high
    ]
    if wrong:
        errors.append(
            f"{label} count-range mismatch examples={wrong[:8]} allowed=[{low},{high}]"
        )


def validate_condition_balance(
    config: dict[str, Any],
    scenes: list[dict[str, str]],
    instances: list[dict[str, Any]],
    errors: list[str],
) -> dict[str, Counter[Any]]:
    lux_ids = [item["id"] for item in config["target_illuminance_bins"]]
    rig_ids = [item["id"] for item in config["multi_light_rigs"]]
    shadow_ids = [item["id"] for item in config["shadow_regimes"]]
    source_profiles = config["source"]["expected_source_lighting_profiles"]
    cells = [f"{rig}__{lux}__{shadow}" for rig in rig_ids for lux in lux_ids for shadow in shadow_ids]
    scene_lux = Counter(row.get("synthetic_illuminance_proxy_bin") for row in scenes)
    scene_rig = Counter(row.get("multi_light_rig_id") for row in scenes)
    scene_shadow = Counter(row.get("shadow_regime_id") for row in scenes)
    scene_cell = Counter(row.get("condition_cell_id") for row in scenes)
    scene_rig_lux = Counter(
        (row.get("multi_light_rig_id"), row.get("synthetic_illuminance_proxy_bin"))
        for row in scenes
    )
    source_profile_cell = Counter(
        (row.get("source_lighting_profile"), row.get("condition_cell_id"))
        for row in scenes
    )
    class_count = Counter(row.get("component_status_class") for row in instances)
    class_lux = Counter(
        (row.get("component_status_class"), row.get("synthetic_illuminance_proxy_bin"))
        for row in instances
    )
    class_rig = Counter(
        (row.get("component_status_class"), row.get("multi_light_rig_id"))
        for row in instances
    )
    class_shadow = Counter(
        (row.get("component_status_class"), row.get("shadow_regime_id"))
        for row in instances
    )
    class_rig_lux = Counter(
        (
            row.get("component_status_class"),
            row.get("multi_light_rig_id"),
            row.get("synthetic_illuminance_proxy_bin"),
        )
        for row in instances
    )
    class_cell = Counter(
        (
            row.get("component_status_class"),
            f"{row.get('multi_light_rig_id')}__{row.get('synthetic_illuminance_proxy_bin')}__{row.get('shadow_regime_id')}",
        )
        for row in instances
    )
    source_reuse = Counter(row.get("source_scene_id") for row in scenes)
    contract = config["balance_contract"]
    expected_keys(scene_lux, lux_ids, contract["scene_per_illuminance_bin"], "scene/lux", errors)
    expected_keys(scene_rig, rig_ids, contract["scene_per_rig"], "scene/rig", errors)
    expected_keys(
        scene_shadow,
        shadow_ids,
        contract["scene_per_shadow_regime"],
        "scene/shadow",
        errors,
    )
    expected_keys(
        scene_cell,
        cells,
        contract["scene_per_full_condition_cell"],
        "scene/full-condition",
        errors,
    )
    expected_keys(
        scene_rig_lux,
        ((rig, lux) for rig in rig_ids for lux in lux_ids),
        contract["scene_per_rig_illuminance_cell"],
        "scene/rig/lux",
        errors,
    )
    expected_keys(
        source_profile_cell,
        ((profile, cell) for profile in source_profiles for cell in cells),
        contract["source_profile_scene_per_full_condition_cell"],
        "source-profile/full-condition",
        errors,
    )
    expected_keys(class_count, CLASSES, contract["instance_per_class"], "class", errors)
    expected_keys_range(
        class_lux,
        ((name, lux) for name in CLASSES for lux in lux_ids),
        contract["instance_per_class_illuminance_bin_range"],
        "class/lux",
        errors,
    )
    expected_keys_range(
        class_rig,
        ((name, rig) for name in CLASSES for rig in rig_ids),
        contract["instance_per_class_rig_range"],
        "class/rig",
        errors,
    )
    expected_keys(
        class_shadow,
        ((name, shadow) for name in CLASSES for shadow in shadow_ids),
        contract["instance_per_class_shadow_regime"],
        "class/shadow",
        errors,
    )
    expected_keys_range(
        class_rig_lux,
        ((name, rig, lux) for name in CLASSES for rig in rig_ids for lux in lux_ids),
        contract["instance_per_class_rig_illuminance_cell_range"],
        "class/rig/lux",
        errors,
    )
    expected_keys_range(
        class_cell,
        ((name, cell) for name in CLASSES for cell in cells),
        contract["instance_per_class_full_condition_cell_range"],
        "class/full-condition",
        errors,
    )
    expected_keys(
        source_reuse,
        (
            row.get("source_scene_id")
            for row in scenes
            if row.get("source_variant_index") == "0"
        ),
        contract["source_scene_reuse"],
        "source-scene/reuse",
        errors,
    )
    return {
        "scene_lux": scene_lux,
        "scene_rig": scene_rig,
        "scene_shadow": scene_shadow,
        "scene_cell": scene_cell,
        "class_count": class_count,
        "class_cell": class_cell,
        "class_lux": class_lux,
        "class_rig": class_rig,
        "class_rig_lux": class_rig_lux,
        "class_shadow": class_shadow,
        "source_profile_cell": source_profile_cell,
        "source_reuse": source_reuse,
    }


def validate_sensor_params(
    params: Any, lux: dict[str, Any], prefix: str, errors: list[str]
) -> None:
    expected_fields = {"noise_sigma", "noise_seed", "blur_radius_px", "jpeg_quality"}
    if not isinstance(params, dict) or set(params) != expected_fields:
        errors.append(f"{prefix}: sensor parameter schema mismatch")
        return
    try:
        sigma = finite_float(params["noise_sigma"], "noise sigma")
        blur = finite_float(params["blur_radius_px"], "blur radius")
        quality = integer(params["jpeg_quality"], "JPEG quality")
        seed = integer(params["noise_seed"], "noise seed")
        if not float(lux["sensor_noise_sigma"][0]) <= sigma <= float(
            lux["sensor_noise_sigma"][1]
        ):
            errors.append(f"{prefix}: sensor noise outside bin range")
        if not float(lux["sensor_blur_radius_px"][0]) <= blur <= float(
            lux["sensor_blur_radius_px"][1]
        ):
            errors.append(f"{prefix}: sensor blur outside bin range")
        if not int(lux["jpeg_quality"][0]) <= quality <= int(lux["jpeg_quality"][1]):
            errors.append(f"{prefix}: JPEG quality outside bin range")
        if not 0 <= seed < 2**32:
            errors.append(f"{prefix}: noise seed outside uint32 range")
    except (ValueError, KeyError) as error:
        errors.append(f"{prefix}: invalid sensor parameters: {error}")


def validate_light_source_rows(
    rows: list[dict[str, Any]],
    scene_id: str,
    rig: dict[str, Any],
    coordinate: dict[str, Any],
    errors: list[str],
) -> None:
    prefix = f"{scene_id}/light-sources"
    if len(rows) != len(rig["sources"]):
        errors.append(f"{prefix}: source count mismatch")
        return
    by_id = {row.get("light_id"): row for row in rows}
    if len(by_id) != len(rows):
        errors.append(f"{prefix}: duplicate light IDs")
    for source in rig["sources"]:
        light_id = source["light_id"]
        row = by_id.get(light_id)
        item_prefix = f"{prefix}/{light_id}"
        if row is None:
            errors.append(f"{item_prefix}: missing row")
            continue
        if set(row) != LIGHT_SOURCE_FIELDS:
            errors.append(f"{item_prefix}: exact schema mismatch")
        expected_exact = {
            "lighting_scene_id": scene_id,
            "light_id": light_id,
            "multi_light_rig_id": rig["id"],
            "coordinate_frame_name": coordinate["name"],
            "image_plane_azimuth_deg": float(source["image_plane_azimuth_deg"]),
            "elevation_proxy_deg": float(source["elevation_proxy_deg"]),
            "relative_intensity": float(source["relative_intensity"]),
            "relative_intensity_unit": "1",
            "cct_proxy_kelvin": int(source["cct_proxy_kelvin"]),
            "cct_calibrated": False,
            "anchor_xy_fraction": source["anchor_xy_fraction"],
        }
        for field, expected in expected_exact.items():
            if canonical_json(row.get(field)) != canonical_json(expected):
                errors.append(f"{item_prefix}: metadata mismatch {field}")
        try:
            azimuth = finite_float(row.get("image_plane_azimuth_deg"), "azimuth")
            vector = row.get("direction_vector_image_xy")
            if not isinstance(vector, list) or len(vector) != 2:
                raise ValueError("direction vector must have two entries")
            x = finite_float(vector[0], "direction x")
            y = finite_float(vector[1], "direction y")
            radians = math.radians(azimuth)
            expected_vector = [round(math.cos(radians), 10), round(math.sin(radians), 10)]
            if not math.isclose(x, expected_vector[0], abs_tol=1e-10) or not math.isclose(
                y, expected_vector[1], abs_tol=1e-10
            ):
                errors.append(f"{item_prefix}: angle/vector mismatch")
            if not math.isclose(math.hypot(x, y), 1.0, abs_tol=2e-10):
                errors.append(f"{item_prefix}: direction vector is not unit length")
        except ValueError as error:
            errors.append(f"{item_prefix}: {error}")


def validate_effect(
    effect: Any,
    source_instance: dict[str, Any],
    rig: dict[str, Any],
    lux: dict[str, Any],
    prefix: str,
    errors: list[str],
) -> None:
    if not isinstance(effect, dict) or set(effect) != LIGHTING_EFFECT_FIELDS:
        errors.append(f"{prefix}: lighting effect schema mismatch")
        return
    if effect.get("instance_index") != source_instance.get("instance_index"):
        errors.append(f"{prefix}: lighting effect instance index mismatch")
    same_float(
        effect.get("base_relative_light_power"),
        float(lux["relative_light_power"]),
        "base relative light power",
        prefix,
        errors,
        tolerance=1e-12,
    )
    source_rows = effect.get("source_rows")
    if not isinstance(source_rows, list) or len(source_rows) != len(rig["sources"]):
        errors.append(f"{prefix}: effect source-row count mismatch")
        return
    by_id = {row.get("light_id"): row for row in source_rows if isinstance(row, dict)}
    weights: list[float] = []
    rotation = float(source_instance["rotation_degrees"])
    for source in rig["sources"]:
        row = by_id.get(source["light_id"])
        light_prefix = f"{prefix}/{source['light_id']}"
        if not isinstance(row, dict) or set(row) != LIGHTING_EFFECT_SOURCE_FIELDS:
            errors.append(f"{light_prefix}: effect-source schema mismatch")
            continue
        expected = {
            "image_plane_azimuth_deg": float(source["image_plane_azimuth_deg"]),
            "local_azimuth_proxy_deg": round(
                (float(source["image_plane_azimuth_deg"]) - rotation) % 360.0, 8
            ),
            "elevation_proxy_deg": float(source["elevation_proxy_deg"]),
            "cct_proxy_kelvin": int(source["cct_proxy_kelvin"]),
        }
        for field, expected_value in expected.items():
            same_float(row.get(field), expected_value, field, light_prefix, errors, 1e-8)
        try:
            weight = finite_float(row.get("effective_weight"), "effective weight")
            if weight <= 0.0:
                errors.append(f"{light_prefix}: non-positive effective weight")
            weights.append(weight)
        except ValueError as error:
            errors.append(f"{light_prefix}: {error}")
    # Generator normalizes in float32 and serializes each source weight to 8
    # decimals.  Across three sources the observed worst-case representation
    # error is 1.2e-7; 2e-7 is the deterministic format-level upper gate.
    if weights and not math.isclose(sum(weights), 1.0, abs_tol=2e-7):
        errors.append(f"{prefix}: effective source weights do not sum to one")
    try:
        gain_min = finite_float(effect.get("gain_min"), "gain min")
        gain_mean = finite_float(effect.get("gain_mean"), "gain mean")
        gain_max = finite_float(effect.get("gain_max"), "gain max")
        if not 0.0 < gain_min <= gain_mean <= gain_max:
            errors.append(f"{prefix}: invalid gain ordering")
        color = effect.get("color_gain_rgb")
        if not isinstance(color, list) or len(color) != 3:
            errors.append(f"{prefix}: invalid color gain")
        elif any(finite_float(value, "color gain") <= 0.0 for value in color):
            errors.append(f"{prefix}: non-positive color gain")
        applied = effect.get("occluder_shadow_applied")
        if not isinstance(applied, bool):
            errors.append(f"{prefix}: occluder flag is not boolean")
        strength = finite_float(effect.get("occluder_shadow_strength"), "occluder strength")
        width = finite_float(effect.get("occluder_shadow_width_fraction"), "occluder width")
        if not applied and (strength != 0.0 or width != 0.0):
            errors.append(f"{prefix}: unapplied occluder has non-zero parameters")
    except (ValueError, TypeError) as error:
        errors.append(f"{prefix}: invalid lighting effect: {error}")


def validate_cast_shadow_rows(
    rows: Any,
    rig: dict[str, Any],
    regime: dict[str, Any],
    config: dict[str, Any],
    scene_id: str,
    errors: list[str],
) -> None:
    prefix = f"{scene_id}/cast-shadows"
    expected_count = 5 * (1 + len(rig["sources"]))
    if not isinstance(rows, list) or len(rows) != expected_count:
        errors.append(f"{prefix}: row count mismatch")
        return
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != CAST_SHADOW_FIELDS:
            errors.append(f"{prefix}: exact row schema mismatch")
            continue
        try:
            key = (integer(row["instance_index"], "instance index"), str(row["light_id"]))
        except (KeyError, ValueError) as error:
            errors.append(f"{prefix}: {error}")
            continue
        if key in by_key:
            errors.append(f"{prefix}: duplicate row {key}")
        by_key[key] = row
    source_by_id = {source["light_id"]: source for source in rig["sources"]}
    rendering = config["rendering"]
    for instance_index in range(1, 6):
        contact = by_key.get((instance_index, "CONTACT"))
        contact_prefix = f"{prefix}/{instance_index}/CONTACT"
        if contact is None:
            errors.append(f"{contact_prefix}: missing contact-shadow row")
        else:
            if contact.get("shadow_kind") != "contact":
                errors.append(f"{contact_prefix}: shadow_kind must be contact")
            if contact.get("shadow_direction_deg") is not None:
                errors.append(f"{contact_prefix}: contact direction must be null")
            try:
                dx = integer(contact["offset_x_px"], "contact offset x")
                dy = integer(contact["offset_y_px"], "contact offset y")
                length = finite_float(contact["length_px"], "contact length")
                blur = finite_float(contact["blur_radius_px"], "contact blur")
                opacity = finite_float(contact["opacity"], "contact opacity")
                offset_scale = float(regime["contact_offset_scale"])
                x_bounds = sorted(
                    float(value) * offset_scale
                    for value in rendering["contact_shadow_offset_x_px"]
                )
                y_bounds = sorted(
                    float(value) * offset_scale
                    for value in rendering["contact_shadow_offset_y_px"]
                )
                # Generator rounds scaled offsets to integer pixels.
                if not math.floor(x_bounds[0] - 0.5) <= dx <= math.ceil(x_bounds[1] + 0.5):
                    errors.append(f"{contact_prefix}: x offset outside regime range")
                if not math.floor(y_bounds[0] - 0.5) <= dy <= math.ceil(y_bounds[1] + 0.5):
                    errors.append(f"{contact_prefix}: y offset outside regime range")
                if length != 0.0:
                    errors.append(f"{contact_prefix}: contact length must be zero")
                base_blur = rendering["contact_shadow_blur_radius_px"]
                blur_bounds = [
                    float(base_blur[0]) * float(regime["contact_blur_scale"]),
                    float(base_blur[1]) * float(regime["contact_blur_scale"]),
                ]
                if not blur_bounds[0] <= blur <= blur_bounds[1]:
                    errors.append(f"{contact_prefix}: blur outside regime range")
                base_opacity = rendering["contact_shadow_opacity"]
                opacity_bounds = [
                    float(base_opacity[0]) * float(regime["contact_opacity_scale"]),
                    float(base_opacity[1]) * float(regime["contact_opacity_scale"]),
                ]
                if not opacity_bounds[0] <= opacity <= opacity_bounds[1]:
                    errors.append(f"{contact_prefix}: opacity outside regime range")
            except (KeyError, ValueError) as error:
                errors.append(f"{contact_prefix}: {error}")
        for light_id, source in source_by_id.items():
            row = by_key.get((instance_index, light_id))
            item_prefix = f"{prefix}/{instance_index}/{light_id}"
            if row is None:
                errors.append(f"{item_prefix}: missing row")
                continue
            if row.get("shadow_kind") != "directional_cast":
                errors.append(f"{item_prefix}: shadow_kind must be directional_cast")
            azimuth = float(source["image_plane_azimuth_deg"])
            expected_direction = round((azimuth + 180.0) % 360.0, 8)
            same_float(
                row.get("shadow_direction_deg"),
                expected_direction,
                "shadow direction",
                item_prefix,
                errors,
                1e-8,
            )
            try:
                dx = integer(row["offset_x_px"], "offset x")
                dy = integer(row["offset_y_px"], "offset y")
                length = finite_float(row["length_px"], "length")
                blur = finite_float(row["blur_radius_px"], "blur")
                opacity = finite_float(row["opacity"], "opacity")
                expected_x = -math.cos(math.radians(azimuth))
                expected_y = -math.sin(math.radians(azimuth))
                if dx * expected_x + dy * expected_y <= 0.0:
                    errors.append(f"{item_prefix}: shadow offset opposes recorded direction")
                if abs(math.hypot(dx, dy) - length) > math.sqrt(2.0):
                    errors.append(f"{item_prefix}: shadow offset/length mismatch")
                blur_bounds = rendering["cast_shadow_blur_radius_px"]
                expected_blur_bounds = [
                    float(blur_bounds[0]) * float(regime["directional_blur_scale"]),
                    float(blur_bounds[1]) * float(regime["directional_blur_scale"]),
                ]
                if not expected_blur_bounds[0] <= blur <= expected_blur_bounds[1]:
                    errors.append(f"{item_prefix}: blur outside regime range")
                opacity_bounds = rendering["cast_shadow_opacity"]
                scale = float(regime["directional_opacity_scale"])
                intensity_factor = 0.70 + 0.30 * float(source["relative_intensity"])
                expected_opacity = [
                    float(opacity_bounds[0]) * scale * intensity_factor,
                    float(opacity_bounds[1]) * scale * intensity_factor,
                ]
                if not expected_opacity[0] <= opacity <= expected_opacity[1]:
                    errors.append(f"{item_prefix}: opacity outside regime range")
            except (KeyError, ValueError) as error:
                errors.append(f"{item_prefix}: {error}")


def validate_scene_metrics(
    scene: dict[str, str],
    lighting: dict[str, Any],
    shadow_mask: np.ndarray,
    component_mask: np.ndarray,
    config: dict[str, Any],
    prefix: str,
    errors: list[str],
) -> None:
    qc = config["qc"]
    metric_fields = {
        "background_mean_luma": "background_mean_luma",
        "background_p99_luma": "background_p99_luma",
        "component_mean_luma": "component_mean_luma",
        "component_dark_fraction": "component_dark_fraction",
        "component_saturated_fraction": "component_saturated_fraction",
        "pre_sensor_positive_spill_max": "pre_sensor_positive_spill_max",
        "post_jpeg_paired_clean_spill_p99": "post_jpeg_paired_clean_spill_p99",
        "post_jpeg_paired_clean_spill_max": "post_jpeg_paired_clean_spill_max",
        "post_jpeg_paired_clean_spill_fraction": "post_jpeg_paired_clean_spill_fraction",
        "post_jpeg_paired_clean_spill_energy": "post_jpeg_paired_clean_spill_energy",
        "shadow_nonzero_fraction": "shadow_nonzero_fraction",
        "shadow_max_attenuation": "shadow_max_attenuation",
    }
    metrics = lighting.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(metric_fields):
        errors.append(f"{prefix}: lighting metric schema mismatch")
        return
    for manifest_field, metric_field in metric_fields.items():
        same_float(scene.get(manifest_field), metrics[metric_field], manifest_field, prefix, errors, 2.1e-6)
    try:
        values = {field: finite_float(scene.get(field), field) for field in metric_fields}
        gates = [
            (
                values["background_mean_luma"] <= float(qc["maximum_background_mean_luma"]),
                "background mean",
            ),
            (
                values["background_p99_luma"] <= float(qc["maximum_background_p99_luma"]),
                "background p99",
            ),
            (
                values["component_mean_luma"] >= float(qc["minimum_component_mean_luma"]),
                "component mean low",
            ),
            (
                values["component_mean_luma"] <= float(qc["maximum_component_mean_luma"]),
                "component mean high",
            ),
            (
                values["component_dark_fraction"] <= float(qc["maximum_component_dark_fraction"]),
                "component dark fraction",
            ),
            (
                values["component_saturated_fraction"]
                <= float(qc["maximum_component_saturated_fraction"]),
                "component saturated fraction",
            ),
            (
                values["pre_sensor_positive_spill_max"]
                <= float(qc["pre_sensor_maximum_positive_spill"]),
                "pre-sensor positive spill",
            ),
            (
                values["post_jpeg_paired_clean_spill_p99"]
                <= float(qc["post_jpeg_paired_clean_spill_p99"]),
                "post-JPEG spill p99",
            ),
            (
                values["post_jpeg_paired_clean_spill_max"]
                <= float(qc["post_jpeg_paired_clean_spill_max"]),
                "post-JPEG spill max",
            ),
            (
                values["post_jpeg_paired_clean_spill_fraction"]
                <= float(qc["post_jpeg_paired_clean_spill_fraction"]),
                "post-JPEG spill fraction",
            ),
            (
                values["shadow_nonzero_fraction"]
                >= float(qc["minimum_shadow_nonzero_fraction"]),
                "shadow support",
            ),
            (
                values["shadow_max_attenuation"]
                <= float(qc["maximum_shadow_attenuation"]) + 1e-8,
                "shadow attenuation",
            ),
        ]
        for passed, label in gates:
            if not passed:
                errors.append(f"{prefix}: {label} gate failed")
        for fraction_field in (
            "component_dark_fraction",
            "component_saturated_fraction",
            "post_jpeg_paired_clean_spill_fraction",
            "shadow_nonzero_fraction",
        ):
            if not 0.0 <= values[fraction_field] <= 1.0:
                errors.append(f"{prefix}: {fraction_field} outside [0,1]")
        if values["post_jpeg_paired_clean_spill_energy"] < 0.0:
            errors.append(f"{prefix}: negative post-JPEG spill energy")
    except (ValueError, KeyError) as error:
        errors.append(f"{prefix}: invalid scene metrics: {error}")

    actual_nonzero = float(np.mean(shadow_mask > 0))
    same_float(
        scene.get("shadow_nonzero_fraction"),
        actual_nonzero,
        "shadow nonzero fraction from mask",
        prefix,
        errors,
        1.1e-8,
    )
    maximum_u8 = int(shadow_mask.max())
    if maximum_u8 > int(math.ceil(float(qc["maximum_shadow_attenuation"]) * 255.0)):
        errors.append(f"{prefix}: shadow mask exceeds attenuation range")
    if np.any(shadow_mask[component_mask > 0] != 0):
        errors.append(f"{prefix}: shadow attenuation overlaps component support")
    cast_rows = lighting.get("cast_shadow_rows")
    if isinstance(cast_rows, list) and cast_rows:
        maximum_extent = max(
            math.hypot(float(row["offset_x_px"]), float(row["offset_y_px"]))
            + 6.0 * float(row["blur_radius_px"])
            + 4.0
            for row in cast_rows
        )
        radius = int(math.ceil(maximum_extent))
        support = np.zeros_like(component_mask, dtype=bool)
        for instance_id in range(1, 6):
            ys, xs = np.where(component_mask == instance_id)
            if len(xs) == 0:
                continue
            x0 = max(0, int(xs.min()) - radius)
            x1 = min(component_mask.shape[1], int(xs.max()) + radius + 1)
            y0 = max(0, int(ys.min()) - radius)
            y1 = min(component_mask.shape[0], int(ys.max()) + radius + 1)
            support[y0:y1, x0:x1] = True
        if np.any((shadow_mask > 0) & ~support):
            errors.append(f"{prefix}: shadow mask has nonlocal support")


def validate_defect_visibility(
    scene: dict[str, str],
    lighting_rows: Any,
    published_instances: list[dict[str, Any]],
    config: dict[str, Any],
    prefix: str,
    errors: list[str],
) -> dict[int, dict[str, Any]]:
    if not isinstance(lighting_rows, list):
        errors.append(f"{prefix}: paired-clean defect visibility is not a list")
        return {}
    visibility_by_instance: dict[int, dict[str, Any]] = {}
    for row in lighting_rows:
        if not isinstance(row, dict) or set(row) != DEFECT_VISIBILITY_FIELDS:
            errors.append(f"{prefix}: paired-clean visibility exact schema mismatch")
            continue
        try:
            instance_index = integer(row["instance_index"], "visibility instance index")
        except (KeyError, ValueError) as error:
            errors.append(f"{prefix}: invalid visibility instance index: {error}")
            continue
        if instance_index in visibility_by_instance:
            errors.append(f"{prefix}: duplicate visibility instance {instance_index}")
        visibility_by_instance[instance_index] = row

    expected_defects = {
        int(row["instance_index"]): row
        for row in published_instances
        if row.get("component_status_class") != "normal_proxy"
    }
    if set(visibility_by_instance) != set(expected_defects):
        errors.append(
            f"{prefix}: paired-clean visibility instance inventory mismatch "
            f"actual={sorted(visibility_by_instance)} expected={sorted(expected_defects)}"
        )
    for instance_index, instance in expected_defects.items():
        row = visibility_by_instance.get(instance_index)
        if row is None:
            continue
        item_prefix = f"{prefix}/visibility-{instance_index}"
        expected_exact = {
            "instance_index": instance_index,
            "class_name": instance["component_status_class"],
            "semantic_id": int(instance["defect_semantic_id"]),
        }
        for field, expected in expected_exact.items():
            if row.get(field) != expected:
                errors.append(f"{item_prefix}: mismatch {field}")
        try:
            area = integer(row["area"], "visibility area")
            bbox_w = integer(row["bbox_w"], "visibility bbox width")
            bbox_h = integer(row["bbox_h"], "visibility bbox height")
            major = integer(row["major"], "visibility major")
            minor = integer(row["minor"], "visibility minor")
            mean_delta = finite_float(row["mean_abs_delta"], "mean abs delta")
            changed_fraction = finite_float(row["changed_fraction"], "changed fraction")
            delta_e = finite_float(row["delta_e76_p50"], "delta E76 p50")
            if area <= 0 or bbox_w <= 0 or bbox_h <= 0:
                errors.append(f"{item_prefix}: empty resized defect support")
            if major != max(bbox_w, bbox_h) or minor != min(bbox_w, bbox_h):
                errors.append(f"{item_prefix}: major/minor geometry mismatch")
            same_float(
                row.get("diag"),
                round(math.hypot(bbox_w, bbox_h), 6),
                "visibility diagonal",
                item_prefix,
                errors,
                1.1e-6,
            )
            if mean_delta < float(
                config["qc"]["minimum_defect_mean_abs_delta"][instance["component_status_class"]]
            ):
                errors.append(f"{item_prefix}: mean-absolute-delta gate failed")
            if changed_fraction < float(
                config["qc"]["minimum_defect_changed_fraction"][instance["component_status_class"]]
            ):
                errors.append(f"{item_prefix}: changed-fraction gate failed")
            if not 0.0 <= changed_fraction <= 1.0 or delta_e < 0.0:
                errors.append(f"{item_prefix}: invalid visibility metric range")
        except (KeyError, ValueError) as error:
            errors.append(f"{item_prefix}: invalid paired-clean visibility: {error}")

    if visibility_by_instance:
        minimum_delta = min(
            finite_float(row.get("mean_abs_delta"), "mean abs delta")
            for row in visibility_by_instance.values()
        )
        minimum_fraction = min(
            finite_float(row.get("changed_fraction"), "changed fraction")
            for row in visibility_by_instance.values()
        )
        same_float(
            scene.get("minimum_defect_mean_abs_delta"),
            minimum_delta,
            "minimum defect mean abs delta",
            prefix,
            errors,
            1.1e-6,
        )
        same_float(
            scene.get("minimum_defect_changed_fraction"),
            minimum_fraction,
            "minimum defect changed fraction",
            prefix,
            errors,
            1.1e-8,
        )
    return visibility_by_instance


def dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square dilation, byte-equivalent in shape semantics to the pinned v4 helper."""

    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    binary = np.asarray(mask, dtype=np.uint8)
    window = radius * 2 + 1
    horizontal_padded = np.pad(binary, ((0, 0), (radius, radius)))
    horizontal_sum = np.pad(
        np.cumsum(horizontal_padded, axis=1, dtype=np.int32), ((0, 0), (1, 0))
    )
    horizontal = (horizontal_sum[:, window:] - horizontal_sum[:, :-window]) > 0
    vertical_padded = np.pad(
        horizontal.astype(np.uint8), ((radius, radius), (0, 0))
    )
    vertical_sum = np.pad(
        np.cumsum(vertical_padded, axis=0, dtype=np.int32), ((1, 0), (0, 0))
    )
    return (vertical_sum[window:, :] - vertical_sum[:-window, :]) > 0


def validate_instance_local_contrast(
    image_array: np.ndarray,
    component_mask: np.ndarray,
    defect_mask: np.ndarray,
    instances: list[dict[str, Any]],
    v4_config: dict[str, Any],
    prefix: str,
    statistics: dict[str, list[float]],
    errors: list[str],
) -> None:
    """Reapply the pinned v4 per-instance local belt contrast contract to v5."""

    qc = v4_config["qc"]
    missing_classes = set(
        v4_config["component_alpha"]["missing_material_classes"]
    )
    nominal_masks: dict[int, np.ndarray] = {}
    nominal_union = np.zeros(component_mask.shape, dtype=bool)
    for instance in instances:
        instance_index = int(instance["instance_index"])
        nominal = component_mask == instance_index
        if instance["component_status_class"] in missing_classes:
            nominal = np.logical_or(
                nominal, defect_mask == int(instance["defect_semantic_id"])
            )
        nominal_masks[instance_index] = nominal
        nominal_union |= nominal
    pixels = np.asarray(image_array, dtype=np.float32)
    luma = 0.2126 * pixels[..., 0] + 0.7152 * pixels[..., 1] + 0.0722 * pixels[..., 2]
    inner = int(qc["local_belt_annulus_inner_radius_px"])
    outer = int(qc["local_belt_annulus_outer_radius_px"])
    annulus_exclusion = dilate_bool(nominal_union, inner)
    for instance in instances:
        instance_index = int(instance["instance_index"])
        item_prefix = f"{prefix}/instance-{instance_index}/local-contrast"
        visible = component_mask == instance_index
        annulus = dilate_bool(nominal_masks[instance_index], outer) & ~annulus_exclusion
        area = int(annulus.sum())
        if area < 100:
            errors.append(f"{item_prefix}: local belt annulus area below v4 minimum 100")
            continue
        if not np.any(visible):
            errors.append(f"{item_prefix}: empty visible instance mask")
            continue
        component_mean = float(luma[visible].mean())
        local_mean = float(luma[annulus].mean())
        delta = component_mean - local_mean
        ratio = component_mean / max(local_mean, 1e-6)
        statistics["component_mean_luma"].append(component_mean)
        statistics["local_belt_mean_luma"].append(local_mean)
        statistics["local_luma_delta"].append(delta)
        statistics["local_luma_ratio"].append(ratio)
        if delta < float(qc["minimum_instance_component_background_luma_delta"]):
            errors.append(
                f"{item_prefix}: delta={delta:.6f} below pinned v4 gate "
                f"{qc['minimum_instance_component_background_luma_delta']}"
            )
        if ratio < float(qc["minimum_instance_component_background_luma_ratio"]):
            errors.append(
                f"{item_prefix}: ratio={ratio:.6f} below pinned v4 gate "
                f"{qc['minimum_instance_component_background_luma_ratio']}"
            )


def jpeg_payload(image: Image.Image, quality: int) -> bytes:
    output = io.BytesIO()
    image.save(
        output,
        format="JPEG",
        quality=int(quality),
        optimize=True,
        progressive=False,
        subsampling=2,
    )
    return output.getvalue()


def png_l_payload(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="L").save(
        output, format="PNG", optimize=True
    )
    return output.getvalue()


def draw_overlay(image: Image.Image, instances: list[dict[str, Any]]) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    colors = [
        (52, 211, 153),
        (248, 113, 113),
        (251, 191, 36),
        (96, 165, 250),
        (244, 114, 182),
        (167, 139, 250),
        (251, 146, 60),
        (34, 211, 238),
    ]
    for instance in instances:
        x, y, width, height = [int(value) for value in instance["visible_bbox_xywh"]]
        color = colors[int(instance["component_yolo_class_id"])]
        draw.rectangle((x, y, x + width - 1, y + height - 1), outline=color, width=4)
    return output


def contact_sheet_payload(
    config: dict[str, Any],
    scenes: list[dict[str, str]],
    instances_by_scene: dict[str, list[dict[str, Any]]],
    overlay: bool,
) -> bytes:
    rigs = [item["id"] for item in config["multi_light_rigs"]]
    lux_ids = [item["id"] for item in config["target_illuminance_bins"]]
    shadow_ids = [item["id"] for item in config["shadow_regimes"]]
    tile_width, tile_height, label_height = 160, 90, 22
    sheet = Image.new(
        "RGB",
        (
            len(rigs) * len(shadow_ids) * tile_width,
            len(lux_ids) * (tile_height + label_height),
        ),
        (15, 15, 15),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    by_cell = {row["condition_cell_id"]: row for row in scenes}
    for row_index, lux_id in enumerate(lux_ids):
        for rig_index, rig_id in enumerate(rigs):
            for shadow_index, shadow_id in enumerate(shadow_ids):
                cell_id = f"{rig_id}__{lux_id}__{shadow_id}"
                row = by_cell[cell_id]
                with Image.open(repository_path(row["image_path"], "contact image")) as opened:
                    image = opened.convert("RGB")
                    image.load()
                if overlay:
                    image = draw_overlay(image, instances_by_scene[row["scene_id"]])
                image = image.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
                left = (rig_index * len(shadow_ids) + shadow_index) * tile_width
                top = row_index * (tile_height + label_height)
                sheet.paste(image, (left, top))
                label = f"{lux_id} {rig_id[:9]} {shadow_id[:4]}"
                draw.text(
                    (left + 3, top + tile_height + 4),
                    label,
                    fill=(235, 235, 235),
                    font=font,
                )
    return jpeg_payload(sheet, 92)


def paired_comparison_payload(scenes: list[dict[str, str]]) -> bytes:
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in scenes:
        by_source[row["source_scene_id"]].append(row)
    source_ids = sorted(by_source)
    sample_count = min(12, len(source_ids))
    selected_indices = [
        int(round(index * (len(source_ids) - 1) / max(1, sample_count - 1)))
        for index in range(sample_count)
    ]
    selected = [source_ids[index] for index in selected_indices]
    tile_width, tile_height, label_height = 160, 90, 34
    pair_width = tile_width * 2
    columns = 2
    row_count = int(math.ceil(sample_count / columns))
    sheet = Image.new(
        "RGB",
        (columns * pair_width, row_count * (tile_height + label_height)),
        (15, 15, 15),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for pair_index, source_scene_id in enumerate(selected):
        variants = sorted(
            by_source[source_scene_id], key=lambda row: int(row["source_variant_index"])
        )
        if len(variants) != 2:
            raise ValidationSetupError(
                f"paired comparison inventory mismatch: {source_scene_id}"
            )
        left = (pair_index % columns) * pair_width
        top = (pair_index // columns) * (tile_height + label_height)
        for variant_index, row in enumerate(variants):
            with Image.open(repository_path(row["image_path"], "paired image")) as opened:
                image = opened.convert("RGB")
                image.load()
            image = image.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
            sheet.paste(image, (left + variant_index * tile_width, top))
        first, second = variants
        first_label = (
            f"A {first['synthetic_illuminance_proxy_bin']} "
            f"{first['multi_light_rig_id'][:10]} {first['shadow_regime_id'][:4]}"
        )
        second_label = (
            f"B {second['synthetic_illuminance_proxy_bin']} "
            f"{second['multi_light_rig_id'][:10]} {second['shadow_regime_id'][:4]}"
        )
        draw.text(
            (left + 3, top + tile_height + 2),
            first_label,
            fill=(235, 235, 235),
            font=font,
        )
        draw.text(
            (left + tile_width + 3, top + tile_height + 2),
            second_label,
            fill=(235, 235, 235),
            font=font,
        )
    return jpeg_payload(sheet, 92)


def build_expected_coco(
    source_document: dict[str, Any],
    defect_document: bool,
    config: dict[str, Any],
    sorted_source_scenes: list[dict[str, str]],
    scenes_by_id: dict[str, dict[str, str]],
    assignment: dict[tuple[str, int], dict[str, Any]],
    source_instances_by_scene: dict[str, list[dict[str, Any]]],
    instances_by_scene: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    info = {
        "description": "Synthetic paired multi-light illuminance-proxy train-only auxiliary release",
        "version": config["generator_version"],
        "year": 2026,
        "photometry_domain": config["photometry_domain"],
        "absolute_lux_eligible": config["absolute_lux_eligible"],
    }
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    source_annotations = {
        int(row["id"]): row for row in source_document["annotations"]
    }
    annotation_id = 1
    variants = int(config["variants_per_source_scene"])
    for source_index, source_scene in enumerate(sorted_source_scenes):
        source_scene_id = source_scene["scene_id"]
        source_instances = sorted(
            source_instances_by_scene[source_scene_id],
            key=lambda row: int(row["instance_index"]),
        )
        for variant_index in range(variants):
            output_index = source_index * variants + variant_index
            image_id = output_index + 1
            scene_id = f"{config['sample_id_prefix']}-{output_index:04d}"
            scene = scenes_by_id[scene_id]
            condition = assignment[(source_scene_id, variant_index)]
            images.append(
                {
                    "id": image_id,
                    "file_name": Path(scene["image_path"])
                    .relative_to(Path("synthetic/v5_illumination"))
                    .as_posix(),
                    "width": int(source_scene["width"]),
                    "height": int(source_scene["height"]),
                    "scene_id": scene_id,
                    "source_scene_id": source_scene_id,
                    "source_variant_index": variant_index,
                    "composition_family_id": source_scene_id,
                    "synthetic_illuminance_proxy_bin": condition["lux"]["id"],
                    "capture_plan_target_lux": int(
                        condition["lux"]["capture_plan_target_lux"]
                    ),
                    "synthetic_relative_light_power": float(
                        condition["lux"]["relative_light_power"]
                    ),
                    "multi_light_rig_id": condition["rig"]["id"],
                    "shadow_regime_id": condition["shadow"]["id"],
                    "absolute_lux_eligible": config["absolute_lux_eligible"],
                    "measured_illuminance_lux": None,
                    "photometry_domain": config["photometry_domain"],
                }
            )
            published_by_index = {
                int(row["instance_index"]): row
                for row in instances_by_scene[scene_id]
            }
            for source_instance in source_instances:
                instance_index = int(source_instance["instance_index"])
                source_key = (
                    "defect_annotation_id"
                    if defect_document
                    else "component_annotation_id"
                )
                source_annotation_id = source_instance[source_key]
                if source_annotation_id is None:
                    continue
                annotation = deepcopy(source_annotations[int(source_annotation_id)])
                annotation["id"] = annotation_id
                annotation["image_id"] = image_id
                attributes = annotation["attributes"]
                if defect_document:
                    attributes["source_defect_annotation_id"] = int(
                        source_annotation_id
                    )
                else:
                    attributes["source_component_annotation_id"] = int(
                        source_annotation_id
                    )
                attributes["source_scene_id"] = source_scene_id
                attributes["source_variant_index"] = variant_index
                attributes["composition_family_id"] = source_scene_id
                attributes["lighting_scene_id"] = scene_id
                if defect_document:
                    attributes["component_annotation_id"] = output_index * 5 + instance_index
                    attributes["semantic_mask"] = (
                        f"masks/defect_semantic/train/{scene_id}.png"
                    )
                    attributes["paired_clean_defect_visibility"] = published_by_index[
                        instance_index
                    ]["paired_clean_defect_visibility"]
                else:
                    attributes["visible_instance_mask"] = (
                        f"masks/component_visible_instances/train/{scene_id}.png"
                    )
                annotations.append(annotation)
                annotation_id += 1
    return {
        "info": info,
        "licenses": deepcopy(source_document.get("licenses", [])),
        "images": images,
        "annotations": annotations,
        "categories": deepcopy(source_document["categories"]),
    }


def validate_release_metadata(
    metadata: dict[str, Any],
    config: dict[str, Any],
    config_path: Path,
    generator_path: Path,
    source: dict[str, Any],
    paths: dict[str, Path],
    counters: dict[str, Counter[Any]],
    release_files: dict[str, Path],
    payload: int,
    errors: list[str],
) -> None:
    if set(metadata) != RELEASE_FIELDS:
        errors.append(
            f"release metadata schema mismatch missing={sorted(RELEASE_FIELDS-set(metadata))} "
            f"extra={sorted(set(metadata)-RELEASE_FIELDS)}"
        )
    expected_static = {
        "release": config["release"],
        "generator_version": config["generator_version"],
        "qc_gate_version": config["qc_gate_version"],
        "task_type": config["task_type"],
        "split": "train",
        "training_use": config["training_use"],
        "evaluation_eligible": "NO",
        "classification_eligible": "NO",
        "photometry_domain": "SYNTHETIC_PROXY",
        "absolute_lux_eligible": "NO",
        "measured_illuminance_lux": None,
        "photometric_calibration_status": config["photometric_calibration_status"],
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "generator_script": generator_path.relative_to(ROOT).as_posix(),
        "generator_script_sha256": EXPECTED_GENERATOR_SHA256,
        "runtime_contract": current_runtime(),
        "source_release": config["source"]["release"],
        "scene_count": 768,
        "source_scene_count": 384,
        "variants_per_source_scene": 2,
        "composition_family_policy": "SOURCE_V4_SCENE_SHARED_BY_BOTH_VARIANTS",
        "pixel_base_method": "v4_geometry_replay_neutral_added_light_shadow_sensor_pre_jpeg",
        "instances_per_scene": 5,
        "component_instance_count": 3840,
        "defect_instance_count": 3360,
        "condition_cell_count": 48,
        "condition_cell_scene_count_range": [
            min(counters["scene_cell"].values()),
            max(counters["scene_cell"].values()),
        ],
        "class_condition_cell_count_range": [
            min(counters["class_cell"].values()),
            max(counters["class_cell"].values()),
        ],
        "class_illuminance_proxy_bin_count_range": [
            min(counters["class_lux"].values()),
            max(counters["class_lux"].values()),
        ],
        "class_rig_count_range": [
            min(counters["class_rig"].values()),
            max(counters["class_rig"].values()),
        ],
        "class_rig_illuminance_proxy_count_range": [
            min(counters["class_rig_lux"].values()),
            max(counters["class_rig_lux"].values()),
        ],
        "capture_plan_target_lux_by_proxy_bin": {
            item["id"]: int(item["capture_plan_target_lux"])
            for item in config["target_illuminance_bins"]
        },
    }
    for field, expected in expected_static.items():
        if canonical_json(metadata.get(field)) != canonical_json(expected):
            errors.append(f"release metadata mismatch {field}")
    expected_source_pins = {key: sha256_file(path) for key, path in paths.items()}
    if metadata.get("source_pins") != expected_source_pins:
        errors.append("release source pin map mismatch")
    expected_counts = {
        "class_counts": dict(sorted(counters["class_count"].items())),
        "synthetic_illuminance_proxy_bin_scene_counts": dict(
            sorted(counters["scene_lux"].items())
        ),
        "multi_light_rig_scene_counts": dict(sorted(counters["scene_rig"].items())),
        "shadow_regime_scene_counts": dict(sorted(counters["scene_shadow"].items())),
    }
    for field, expected in expected_counts.items():
        if metadata.get(field) != expected:
            errors.append(f"release count mismatch {field}")
    hash_fields = {
        "manifest_sha256": "manifest",
        "instances_sha256": "instances",
        "lighting_scenes_sha256": "lighting_scenes",
        "light_sources_sha256": "light_sources",
        "component_coco_sha256": "component_coco",
        "defect_coco_sha256": "defect_coco",
        "summary_sha256": "summary",
        "condition_matrix_sha256": "condition_matrix",
        "contact_sheet_sha256": "contact_sheet",
        "contact_sheet_overlay_sha256": "contact_sheet_overlay",
        "paired_condition_comparison_sha256": "paired_condition_comparison",
    }
    for field, key in hash_fields.items():
        if metadata.get(field) != sha256_file(release_files[key]):
            errors.append(f"release hash mismatch {field}")
    if metadata.get("tracked_payload_bytes") != payload:
        errors.append(
            f"release payload mismatch metadata={metadata.get('tracked_payload_bytes')} actual={payload}"
        )
    if payload > float(config["qc"]["maximum_payload_mib"]) * 1024 * 1024:
        errors.append("release payload exceeds configured maximum")
    expected_limitations = [
        "capture_plan_target_lux is a future real-capture plan target, not measured synthetic lux.",
        "All photometric changes are deterministic 2D proxies and are not radiometrically calibrated.",
        "Each v4 composition is replayed twice under distinct proxy lighting conditions; both variants and their v4 source share one composition_family_id and must stay in one split.",
        "All scenes inherit the same train-only v4/v2 physical base family and must not enter validation or test.",
        "normal_proxy is paired-clean synthetic data and is not confirmed real OK data.",
        "This auxiliary release changes lighting appearance but adds no new real specimen diversity.",
    ]
    if metadata.get("limitations") != expected_limitations:
        errors.append("release limitation contract mismatch")
    validate_proxy_semantics(metadata, "release", errors)


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    release_root = args.release.resolve()
    errors: list[str] = []
    try:
        config_path.relative_to(ROOT.resolve())
        release_root.relative_to((ROOT / "synthetic").resolve())
        if not config_path.is_file():
            raise ValidationSetupError(f"missing config: {config_path}")
        if sha256_file(config_path) != EXPECTED_CONFIG_SHA256:
            raise ValidationSetupError(
                f"v5 config SHA mismatch: {sha256_file(config_path)}"
            )
        generator_path = ROOT / "scripts" / "generate_synthetic_v5_illumination.py"
        if not generator_path.is_file() or sha256_file(generator_path) != EXPECTED_GENERATOR_SHA256:
            raise ValidationSetupError("v5 generator SHA mismatch")
        config = load_json(config_path)
        if not isinstance(config, dict):
            raise ValidationSetupError("config root is not an object")
        validate_config(config, errors)
        validate_runtime(config, errors)
        source = load_source_context(config, errors)
        if not release_root.is_dir():
            raise ValidationSetupError(f"release directory is missing: {release_root}")
        marker = release_root / EXPECTED_MARKER
        if not marker.is_file() or marker.read_text(encoding="ascii") != EXPECTED_MARKER_TEXT:
            raise ValidationSetupError("release marker is missing or invalid")

        release_files = {
            "manifest": release_root / "annotations" / "manifest.csv",
            "instances": release_root / "annotations" / "instances.jsonl",
            "lighting_scenes": release_root / "annotations" / "lighting_scenes.jsonl",
            "light_sources": release_root / "annotations" / "light_sources.jsonl",
            "component_coco": release_root
            / "annotations"
            / "coco"
            / "component_status_train.json",
            "defect_coco": release_root / "annotations" / "coco" / "defects_train.json",
            "summary": release_root / "annotations" / "summary.csv",
            "condition_matrix": release_root / "annotations" / "condition_matrix.csv",
            "release": release_root / "annotations" / "release.json",
            "contact_sheet": release_root / "contact_sheet.jpg",
            "contact_sheet_overlay": release_root / "contact_sheet_overlay.jpg",
            "paired_condition_comparison": release_root
            / "paired_condition_comparison.jpg",
        }
        missing_release_files = [str(path) for path in release_files.values() if not path.is_file()]
        if missing_release_files:
            raise ValidationSetupError(
                "release is incomplete; missing: " + ", ".join(missing_release_files)
            )
        manifest_fields, scenes = read_csv(release_files["manifest"])
        instances = read_jsonl(release_files["instances"])
        lighting_scenes = read_jsonl(release_files["lighting_scenes"])
        light_sources = read_jsonl(release_files["light_sources"])
        component_coco = load_json(release_files["component_coco"])
        defect_coco = load_json(release_files["defect_coco"])
        summary_fields, summary_rows = read_csv(release_files["summary"])
        condition_matrix_fields, condition_matrix_rows = read_csv(
            release_files["condition_matrix"]
        )
        metadata = load_json(release_files["release"])
        if manifest_fields != MANIFEST_FIELDS:
            errors.append("manifest exact column order/schema mismatch")
        if len(scenes) != 768 or len(instances) != 3840:
            errors.append(
                f"release count mismatch scenes={len(scenes)} instances={len(instances)}"
            )
        if len(lighting_scenes) != 768 or len(light_sources) != 1920:
            errors.append(
                f"lighting metadata count mismatch scenes={len(lighting_scenes)} "
                f"sources={len(light_sources)}"
            )
        if any(set(row) != LIGHTING_SCENE_FIELDS for row in lighting_scenes):
            errors.append("lighting_scenes exact schema mismatch")
        if any(set(row) != LIGHT_SOURCE_FIELDS for row in light_sources):
            errors.append("light_sources exact schema mismatch")
        if not isinstance(component_coco, dict) or len(component_coco.get("images", [])) != 768 or len(component_coco.get("annotations", [])) != 3840:
            errors.append("component COCO cardinality mismatch")
        if not isinstance(defect_coco, dict) or len(defect_coco.get("images", [])) != 768 or len(defect_coco.get("annotations", [])) != 3360:
            errors.append("defect COCO cardinality mismatch")
        if not isinstance(metadata, dict):
            raise ValidationSetupError("release metadata root is not an object")
        validate_proxy_semantics(lighting_scenes, "lighting_scenes", errors)
        validate_proxy_semantics(instances, "instances", errors)

        scripts_path = str((ROOT / "scripts").resolve())
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        generator = importlib.import_module("generate_synthetic_v5_illumination")
        source_scenes = source["scenes"]
        source_instances = source["instances"]
        assignment = generator.build_condition_assignment(config, source_scenes)

        source_scene_by_id = {row["scene_id"]: row for row in source_scenes}
        source_instances_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in source_instances:
            source_instances_by_scene[row["scene_id"]].append(row)
        scenes_by_id = {row.get("scene_id"): row for row in scenes}
        if len(scenes_by_id) != len(scenes):
            errors.append("manifest has duplicate scene IDs")
        v5_scenes_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in scenes:
            v5_scenes_by_source[str(row.get("source_scene_id"))].append(row)
        if set(v5_scenes_by_source) != set(source_scene_by_id):
            errors.append("v5 source scene inventory differs from v4")
        for source_scene_id, variants in v5_scenes_by_source.items():
            if len(variants) != 2 or {row.get("source_variant_index") for row in variants} != {"0", "1"}:
                errors.append(f"source pair inventory mismatch: {source_scene_id}")
                continue
            first, second = sorted(variants, key=lambda row: int(row["source_variant_index"]))
            if first.get("composition_family_id") != source_scene_id or second.get("composition_family_id") != source_scene_id:
                errors.append(f"source pair composition family mismatch: {source_scene_id}")
            for field in (
                "multi_light_rig_id",
                "synthetic_illuminance_proxy_bin",
                "shadow_regime_id",
                "shadow_mask_sha256",
                "image_sha256",
            ):
                if first.get(field) == second.get(field):
                    errors.append(f"source pair must differ in {field}: {source_scene_id}")
        instances_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in instances:
            instances_by_scene[str(row.get("scene_id"))].append(row)
        lighting_by_scene = {row.get("scene_id"): row for row in lighting_scenes}
        if len(lighting_by_scene) != len(lighting_scenes):
            errors.append("lighting_scenes has duplicate scene IDs")
        lights_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in light_sources:
            lights_by_scene[str(row.get("lighting_scene_id"))].append(row)

        rig_by_id = {row["id"]: row for row in config["multi_light_rigs"]}
        lux_by_id = {row["id"]: row for row in config["target_illuminance_bins"]}
        shadow_by_id = {row["id"]: row for row in config["shadow_regimes"]}
        expected_inventory: set[Path] = {marker.resolve()}
        expected_inventory.update(path.resolve() for path in release_files.values())
        v5_parent_usage: Counter[str] = Counter()
        source_parent_usage: Counter[str] = Counter(
            row["source_parent_sample_id"] for row in source_instances
        )
        contrast_statistics: dict[str, list[float]] = defaultdict(list)

        sorted_source_scenes = sorted(source_scenes, key=lambda row: int(row["image_id"]))
        expected_defect_ids: dict[tuple[str, int], int] = {}
        next_defect_id = 1
        for output_index in range(int(config["scene_count"])):
            source_index = output_index // int(config["variants_per_source_scene"])
            source_scene = sorted_source_scenes[source_index]
            scene_id = f"{config['sample_id_prefix']}-{output_index:04d}"
            for source_instance in sorted(
                source_instances_by_scene[source_scene["scene_id"]],
                key=lambda row: int(row["instance_index"]),
            ):
                if source_instance["defect_annotation_id"] is not None:
                    expected_defect_ids[(scene_id, int(source_instance["instance_index"]))] = next_defect_id
                    next_defect_id += 1
        if next_defect_id != 3361:
            errors.append(f"expected defect annotation inventory mismatch: {next_defect_id-1}")

        for output_index in range(int(config["scene_count"])):
            source_index = output_index // int(config["variants_per_source_scene"])
            variant_index = output_index % int(config["variants_per_source_scene"])
            source_scene = sorted_source_scenes[source_index]
            expected_scene_id = f"{config['sample_id_prefix']}-{output_index:04d}"
            scene = scenes_by_id.get(expected_scene_id)
            prefix = f"scene {expected_scene_id}"
            if scene is None:
                errors.append(f"{prefix}: missing manifest row")
                continue
            condition = assignment[(source_scene["scene_id"], variant_index)]
            expected_static = {
                "image_id": str(output_index + 1),
                "source_scene_id": source_scene["scene_id"],
                "source_variant_index": str(variant_index),
                "source_image_path": source_scene["image_path"],
                "pixel_base_method": "v4_geometry_replay_neutral_added_light_shadow_sensor_pre_jpeg",
                "source_lighting_profile": source_scene["lighting_profile"],
                "source_image_sha256": source_scene["image_sha256"],
                "source_component_mask_sha256": source_scene["component_mask_sha256"],
                "source_defect_mask_sha256": source_scene["defect_mask_sha256"],
                "split": "train",
                "task_type": config["task_type"],
                "training_use": config["training_use"],
                "evaluation_eligible": "NO",
                "classification_eligible": "NO",
                "photometry_domain": "SYNTHETIC_PROXY",
                "absolute_lux_eligible": "NO",
                "measured_illuminance_lux": "",
                "photometric_calibration_status": config["photometric_calibration_status"],
                "synthetic_illuminance_proxy_bin": condition["lux"]["id"],
                "capture_plan_target_lux": str(
                    condition["lux"]["capture_plan_target_lux"]
                ),
                "synthetic_relative_light_power": str(condition["lux"]["relative_light_power"]),
                "relative_light_power_unit": "1",
                "calibrated_to_lux": "NO",
                "multi_light_rig_id": condition["rig"]["id"],
                "shadow_regime_id": condition["shadow"]["id"],
                "condition_cell_id": condition["condition_cell_id"],
                "condition_cell_index": str(condition["condition_cell_index"]),
                "light_source_count": str(len(condition["rig"]["sources"])),
                "coordinate_frame_name": config["coordinate_frame"]["name"],
                "instance_count": "5",
                "component_status_labels": source_scene["component_status_labels"],
                "source_parent_ids": source_scene["source_parent_ids"],
                "composition_family_id": source_scene["scene_id"],
                "width": source_scene["width"],
                "height": source_scene["height"],
                "config_sha256": EXPECTED_CONFIG_SHA256,
                "generator_version": config["generator_version"],
                "qc_gate_version": config["qc_gate_version"],
                "qc_status": EXPECTED_QC_STATUS,
                "human_verified": "NO",
            }
            for field, expected in expected_static.items():
                if scene.get(field) != expected:
                    errors.append(f"{prefix}: manifest mismatch {field}")
            if scene.get("light_sources_json") != canonical_json(condition["rig"]["sources"]):
                errors.append(f"{prefix}: manifest rig source JSON mismatch")
            try:
                scene_seed = integer(scene["scene_seed"], "scene_seed")
                attempt = integer(scene["attempt"], "attempt")
                expected_seed = generator.stable_seed(
                    config["global_seed"],
                    config["release"],
                    source_scene["scene_id"],
                    condition["condition_cell_id"],
                    attempt,
                )
                if scene_seed != expected_seed:
                    errors.append(f"{prefix}: deterministic scene seed mismatch")
                if not 0 <= attempt < 30:
                    errors.append(f"{prefix}: attempt outside 0..29")
            except (ValueError, KeyError) as error:
                errors.append(f"{prefix}: invalid scene seed/attempt: {error}")

            relative_paths = {
                "image_path": f"images/train/{expected_scene_id}.jpg",
                "component_instance_mask_path": (
                    f"masks/component_visible_instances/train/{expected_scene_id}.png"
                ),
                "defect_semantic_mask_path": (
                    f"masks/defect_semantic/train/{expected_scene_id}.png"
                ),
                "shadow_attenuation_mask_path": (
                    f"masks/shadow_attenuation/train/{expected_scene_id}.png"
                ),
                "component_yolo_path": (
                    f"labels/yolo_component_status/train/{expected_scene_id}.txt"
                ),
                "defect_yolo_path": f"labels/yolo_defects/train/{expected_scene_id}.txt",
            }
            resolved: dict[str, Path] = {}
            for field, relative in relative_paths.items():
                expected_path = release_path(release_root, relative, f"{prefix}.{field}")
                expected_inventory.add(expected_path.resolve())
                try:
                    actual_path = repository_path(scene[field], f"{prefix}.{field}")
                    resolved[field] = actual_path
                    if actual_path.resolve() != expected_path.resolve():
                        errors.append(f"{prefix}: canonical path mismatch {field}")
                    if not actual_path.is_file():
                        errors.append(f"{prefix}: missing file {field}")
                except (KeyError, ValidationSetupError) as error:
                    errors.append(f"{prefix}: invalid {field}: {error}")
            if len(resolved) != len(relative_paths) or any(
                not path.is_file() for path in resolved.values()
            ):
                continue
            hash_map = {
                "image_path": "image_sha256",
                "component_instance_mask_path": "component_mask_sha256",
                "defect_semantic_mask_path": "defect_mask_sha256",
                "shadow_attenuation_mask_path": "shadow_mask_sha256",
                "component_yolo_path": "component_yolo_sha256",
                "defect_yolo_path": "defect_yolo_sha256",
            }
            for path_field, hash_field in hash_map.items():
                if sha256_file(resolved[path_field]) != scene.get(hash_field):
                    errors.append(f"{prefix}: hash mismatch {hash_field}")

            source_copy_pairs = {
                "component_instance_mask_path": "component_instance_mask_path",
                "defect_semantic_mask_path": "defect_semantic_mask_path",
                "component_yolo_path": "component_yolo_path",
                "defect_yolo_path": "defect_yolo_path",
            }
            source_hash_fields = {
                "component_instance_mask_path": "component_mask_sha256",
                "defect_semantic_mask_path": "defect_mask_sha256",
                "component_yolo_path": "component_yolo_sha256",
                "defect_yolo_path": "defect_yolo_sha256",
            }
            for target_field, source_field in source_copy_pairs.items():
                source_path = repository_path(source_scene[source_field], f"{prefix}.source")
                source_hash_field = source_hash_fields[source_field]
                actual_source_sha = sha256_file(source_path)
                if actual_source_sha != source_scene.get(source_hash_field):
                    errors.append(
                        f"{prefix}: source file SHA differs from v4 manifest "
                        f"{source_hash_field}"
                    )
                if scene.get(source_hash_field) != source_scene.get(source_hash_field):
                    errors.append(
                        f"{prefix}: published source SHA field differs "
                        f"{source_hash_field}"
                    )
                if resolved[target_field].read_bytes() != source_path.read_bytes():
                    errors.append(f"{prefix}: inherited geometry bytes differ {target_field}")
            try:
                with Image.open(resolved["image_path"]) as image:
                    if image.format != "JPEG" or image.mode != "RGB" or image.size != (1280, 720):
                        errors.append(f"{prefix}: image format/mode/size mismatch")
                    if image.getexif():
                        errors.append(f"{prefix}: image contains EXIF")
                    image.load()
                    image_array = np.asarray(image, dtype=np.uint8).copy()
                with Image.open(resolved["component_instance_mask_path"]) as image:
                    if image.format != "PNG" or image.mode != "I;16" or image.size != (1280, 720):
                        errors.append(f"{prefix}: component mask format/mode/size mismatch")
                    component_mask = np.asarray(image, dtype=np.uint16).copy()
                with Image.open(resolved["defect_semantic_mask_path"]) as image:
                    if image.format != "PNG" or image.mode != "L" or image.size != (1280, 720):
                        errors.append(f"{prefix}: defect mask format/mode/size mismatch")
                    defect_mask = np.asarray(image, dtype=np.uint8).copy()
                with Image.open(resolved["shadow_attenuation_mask_path"]) as image:
                    if image.format != "PNG" or image.mode != "L" or image.size != (1280, 720):
                        errors.append(f"{prefix}: shadow mask format/mode/size mismatch")
                    if image.getexif() or image.info:
                        errors.append(f"{prefix}: shadow mask has metadata")
                    shadow_mask = np.asarray(image, dtype=np.uint8).copy()
            except Exception as error:
                errors.append(f"{prefix}: image/mask decode failed: {error}")
                continue
            if set(np.unique(component_mask).tolist()) != {0, 1, 2, 3, 4, 5}:
                errors.append(f"{prefix}: component mask ID inventory mismatch")

            lighting = lighting_by_scene.get(expected_scene_id)
            if lighting is None:
                errors.append(f"{prefix}: missing lighting scene metadata")
                continue
            expected_lighting = {
                "lighting_scene_id": expected_scene_id,
                "scene_id": expected_scene_id,
                "source_scene_id": source_scene["scene_id"],
                "source_variant_index": variant_index,
                "pixel_base_method": "v4_geometry_replay_neutral_added_light_shadow_sensor_pre_jpeg",
                "condition_cell_index": int(condition["condition_cell_index"]),
                "condition_cell_id": condition["condition_cell_id"],
                "photometry_domain": "SYNTHETIC_PROXY",
                "absolute_lux_eligible": "NO",
                "measured_illuminance_lux": None,
                "photometric_calibration_status": config["photometric_calibration_status"],
                "synthetic_illuminance_proxy_bin": condition["lux"]["id"],
                "capture_plan_target_lux": int(
                    condition["lux"]["capture_plan_target_lux"]
                ),
                "multi_light_rig_id": condition["rig"]["id"],
                "shadow_regime_id": condition["shadow"]["id"],
                "source_count": len(condition["rig"]["sources"]),
                "coordinate_frame": config["coordinate_frame"],
            }
            for field, expected in expected_lighting.items():
                if canonical_json(lighting.get(field)) != canonical_json(expected):
                    errors.append(f"{prefix}: lighting metadata mismatch {field}")
            expected_proxy = {
                "relative_light_power": float(condition["lux"]["relative_light_power"]),
                "unit": "1",
                "calibrated_to_lux": False,
                "method": config["rendering"]["model"],
            }
            if canonical_json(lighting.get("synthetic_lux_proxy")) != canonical_json(
                expected_proxy
            ):
                errors.append(f"{prefix}: synthetic lux proxy contract mismatch")
            sensor_manifest = parse_json_field(
                scene["sensor_params_json"], "sensor_params_json", prefix, errors
            )
            if canonical_json(sensor_manifest) != canonical_json(lighting.get("sensor_params")):
                errors.append(f"{prefix}: sensor metadata copies differ")
            validate_sensor_params(lighting.get("sensor_params"), condition["lux"], prefix, errors)
            cast_manifest = parse_json_field(
                scene["cast_shadow_params_json"], "cast_shadow_params_json", prefix, errors
            )
            if canonical_json(cast_manifest) != canonical_json(lighting.get("cast_shadow_rows")):
                errors.append(f"{prefix}: cast-shadow metadata copies differ")
            validate_cast_shadow_rows(
                lighting.get("cast_shadow_rows"),
                condition["rig"],
                condition["shadow"],
                config,
                expected_scene_id,
                errors,
            )
            validate_light_source_rows(
                lights_by_scene.get(expected_scene_id, []),
                expected_scene_id,
                condition["rig"],
                config["coordinate_frame"],
                errors,
            )
            validate_scene_metrics(
                scene,
                lighting,
                shadow_mask,
                component_mask,
                config,
                prefix,
                errors,
            )

            source_scene_instances = sorted(
                source_instances_by_scene[source_scene["scene_id"]],
                key=lambda row: int(row["instance_index"]),
            )
            published_instances = sorted(
                instances_by_scene.get(expected_scene_id, []),
                key=lambda row: int(row.get("instance_index", 0)),
            )
            if len(source_scene_instances) != 5 or len(published_instances) != 5:
                errors.append(f"{prefix}: source/published instance count mismatch")
                continue
            visibility_by_instance = validate_defect_visibility(
                scene,
                lighting.get("paired_clean_defect_visibility"),
                published_instances,
                config,
                prefix,
                errors,
            )
            validate_instance_local_contrast(
                image_array,
                component_mask,
                defect_mask,
                published_instances,
                source["v4_config"],
                prefix,
                contrast_statistics,
                errors,
            )
            missing_classes = set(
                source["v4_config"]["component_alpha"]["missing_material_classes"]
            )
            nominal_union = component_mask > 0
            for published in published_instances:
                if published["component_status_class"] in missing_classes:
                    nominal_union |= defect_mask == int(published["defect_semantic_id"])
            if np.any(shadow_mask[nominal_union] != 0):
                errors.append(f"{prefix}: shadow attenuation overlaps nominal component support")
            for source_instance, published in zip(
                source_scene_instances, published_instances, strict=True
            ):
                instance_index = int(source_instance["instance_index"])
                instance_prefix = f"{prefix}/instance-{instance_index}"
                expected_instance_fields = set(source_instance) | {
                    "source_scene_id",
                    "source_variant_index",
                    "source_component_annotation_id",
                    "source_defect_annotation_id",
                    "source_composition_family_id",
                    "lighting_scene_id",
                    "photometry_domain",
                    "absolute_lux_eligible",
                    "measured_illuminance_lux",
                    "photometric_calibration_status",
                    "synthetic_illuminance_proxy_bin",
                    "capture_plan_target_lux",
                    "synthetic_relative_light_power",
                    "multi_light_rig_id",
                    "shadow_regime_id",
                    "source_lighting_profile",
                    "lighting_effect",
                    "paired_clean_defect_visibility",
                }
                if set(published) != expected_instance_fields:
                    errors.append(f"{instance_prefix}: exact instance schema mismatch")
                for field, source_value in source_instance.items():
                    expected_value = source_value
                    if field == "scene_id":
                        expected_value = expected_scene_id
                    elif field == "image_id":
                        expected_value = output_index + 1
                    elif field == "component_annotation_id":
                        expected_value = output_index * 5 + instance_index
                    elif field == "defect_annotation_id":
                        expected_value = expected_defect_ids.get(
                            (expected_scene_id, instance_index)
                        )
                    elif field == "training_use":
                        expected_value = config["training_use"]
                    elif field == "evaluation_eligible":
                        expected_value = "NO"
                    elif field == "classification_eligible":
                        expected_value = "NO"
                    if canonical_json(published.get(field)) != canonical_json(expected_value):
                        errors.append(f"{instance_prefix}: inherited field mismatch {field}")
                added_expected = {
                    "source_scene_id": source_scene["scene_id"],
                    "source_variant_index": variant_index,
                    "source_component_annotation_id": int(
                        source_instance["component_annotation_id"]
                    ),
                    "source_defect_annotation_id": source_instance[
                        "defect_annotation_id"
                    ],
                    "source_composition_family_id": source_instance["composition_family_id"],
                    "lighting_scene_id": expected_scene_id,
                    "photometry_domain": "SYNTHETIC_PROXY",
                    "absolute_lux_eligible": "NO",
                    "measured_illuminance_lux": None,
                    "photometric_calibration_status": config["photometric_calibration_status"],
                    "synthetic_illuminance_proxy_bin": condition["lux"]["id"],
                    "capture_plan_target_lux": int(
                        condition["lux"]["capture_plan_target_lux"]
                    ),
                    "synthetic_relative_light_power": float(
                        condition["lux"]["relative_light_power"]
                    ),
                    "multi_light_rig_id": condition["rig"]["id"],
                    "shadow_regime_id": condition["shadow"]["id"],
                    "source_lighting_profile": source_scene["lighting_profile"],
                    "evaluation_eligible": "NO",
                    "classification_eligible": "NO",
                    "paired_clean_defect_visibility": visibility_by_instance.get(
                        instance_index
                    ),
                }
                for field, expected in added_expected.items():
                    if canonical_json(published.get(field)) != canonical_json(expected):
                        errors.append(f"{instance_prefix}: v5 metadata mismatch {field}")
                parent_id = published.get("source_parent_sample_id")
                v5_parent_usage[str(parent_id)] += 1
                if parent_id not in source["gradient_ids"]:
                    errors.append(f"{instance_prefix}: transitive parent is not gradient-train")
                validate_effect(
                    published.get("lighting_effect"),
                    source_instance,
                    condition["rig"],
                    condition["lux"],
                    instance_prefix,
                    errors,
                )
        expected_v5_parent_usage = Counter(
            {key: value * int(config["variants_per_source_scene"]) for key, value in source_parent_usage.items()}
        )
        if v5_parent_usage != expected_v5_parent_usage:
            errors.append("v5 transitive parent usage differs from v4 source usage")
        if set(v5_parent_usage) & source["forbidden_ids"]:
            errors.append("v5 contains validation/test transitive parent leakage")

        counters = validate_condition_balance(config, scenes, instances, errors)

        expected_summary: list[dict[str, str]] = []
        for name in CLASSES:
            expected_summary.append(
                {"dimension": "component_status_class", "value": name, "count": "480"}
            )
        for lux in config["target_illuminance_bins"]:
            expected_summary.append(
                {
                    "dimension": "synthetic_illuminance_proxy_bin_scene",
                    "value": lux["id"],
                    "count": "128",
                }
            )
        for rig in config["multi_light_rigs"]:
            expected_summary.append(
                {"dimension": "multi_light_rig_scene", "value": rig["id"], "count": "192"}
            )
        for shadow in config["shadow_regimes"]:
            expected_summary.append(
                {
                    "dimension": "shadow_regime_scene",
                    "value": shadow["id"],
                    "count": "384",
                }
            )
        if summary_fields != ["dimension", "value", "count"] or summary_rows != expected_summary:
            errors.append("summary exact schema/content mismatch")

        expected_condition_fields = [
            "condition_cell_id",
            "multi_light_rig_id",
            "synthetic_illuminance_proxy_bin",
            "capture_plan_target_lux",
            "shadow_regime_id",
            "scene_count",
            *[f"class_{name}" for name in CLASSES],
        ]
        expected_condition_rows: list[dict[str, str]] = []
        for rig in config["multi_light_rigs"]:
            for lux in config["target_illuminance_bins"]:
                for shadow in config["shadow_regimes"]:
                    cell_id = f"{rig['id']}__{lux['id']}__{shadow['id']}"
                    row = {
                        "condition_cell_id": cell_id,
                        "multi_light_rig_id": rig["id"],
                        "synthetic_illuminance_proxy_bin": lux["id"],
                        "capture_plan_target_lux": str(lux["capture_plan_target_lux"]),
                        "shadow_regime_id": shadow["id"],
                        "scene_count": str(counters["scene_cell"][cell_id]),
                    }
                    for class_name in CLASSES:
                        row[f"class_{class_name}"] = str(
                            counters["class_cell"][(class_name, cell_id)]
                        )
                    expected_condition_rows.append(row)
        if (
            condition_matrix_fields != expected_condition_fields
            or condition_matrix_rows != expected_condition_rows
        ):
            errors.append("condition matrix exact schema/content mismatch")

        expected_component_coco = build_expected_coco(
            source["component_coco"],
            False,
            config,
            sorted_source_scenes,
            scenes_by_id,
            assignment,
            source_instances_by_scene,
            instances_by_scene,
        )
        expected_defect_coco = build_expected_coco(
            source["defect_coco"],
            True,
            config,
            sorted_source_scenes,
            scenes_by_id,
            assignment,
            source_instances_by_scene,
            instances_by_scene,
        )
        if canonical_json(component_coco) != canonical_json(expected_component_coco):
            errors.append("component COCO differs from exact v4-geometry derivation")
        if canonical_json(defect_coco) != canonical_json(expected_defect_coco):
            errors.append("defect COCO differs from exact v4-geometry derivation")

        raw_contact = contact_sheet_payload(config, scenes, instances_by_scene, False)
        overlay_contact = contact_sheet_payload(config, scenes, instances_by_scene, True)
        if raw_contact != release_files["contact_sheet"].read_bytes():
            errors.append("raw contact sheet deterministic reconstruction differs")
        if overlay_contact != release_files["contact_sheet_overlay"].read_bytes():
            errors.append("overlay contact sheet deterministic reconstruction differs")
        paired_contact = paired_comparison_payload(scenes)
        if paired_contact != release_files["paired_condition_comparison"].read_bytes():
            errors.append("paired condition sheet deterministic reconstruction differs")

        actual_inventory = {
            path.resolve() for path in release_root.rglob("*") if path.is_file()
        }
        if actual_inventory != expected_inventory:
            missing = sorted(str(path) for path in expected_inventory - actual_inventory)
            extra = sorted(str(path) for path in actual_inventory - expected_inventory)
            errors.append(
                f"release inventory mismatch missing={missing[:5]} extra={extra[:5]} "
                f"expected={len(expected_inventory)} actual={len(actual_inventory)}"
            )
        payload = sum(path.stat().st_size for path in actual_inventory)
        validate_release_metadata(
            metadata,
            config,
            config_path,
            generator_path,
            source,
            source["paths"],
            counters,
            release_files,
            payload,
            errors,
        )

        # Full replay: neutral v4 geometry is reconstructed once per source, then
        # both recorded variants are replayed through their first passing attempt.
        print("starting deterministic full replay of 768 v5 scenes", flush=True)
        neutral_context = generator.load_neutral_v4_context(config)
        neutral_plan_by_scene = {
            plan["scene_id"]: plan for plan in neutral_context.plans
        }
        replay_count = 0
        for source_index, source_scene in enumerate(sorted_source_scenes):
            source_scene_id = source_scene["scene_id"]
            source_scene_instances = sorted(
                source_instances_by_scene[source_scene_id],
                key=lambda row: int(row["instance_index"]),
            )
            try:
                (
                    neutral_defect_array,
                    neutral_clean_array,
                    neutral_component_mask,
                    neutral_defect_mask,
                ) = generator.render_neutral_v4_scene(
                    neutral_context,
                    neutral_plan_by_scene[source_scene_id],
                    source_scene,
                )
            except Exception as error:
                errors.append(f"neutral replay {source_scene_id}: exception: {error}")
                continue
            for variant_index in range(int(config["variants_per_source_scene"])):
                output_index = (
                    source_index * int(config["variants_per_source_scene"])
                    + variant_index
                )
                replay_count += 1
                scene_id = f"{config['sample_id_prefix']}-{output_index:04d}"
                scene = scenes_by_id.get(scene_id)
                if scene is None:
                    continue
                condition = assignment[(source_scene_id, variant_index)]
                try:
                    recorded_attempt = integer(scene["attempt"], "attempt")
                    replay: dict[str, Any] | None = None
                    for attempt in range(recorded_attempt + 1):
                        candidate = generator.render_variant(
                            config,
                            source_scene,
                            source_scene_instances,
                            condition,
                            neutral_defect_array,
                            neutral_clean_array,
                            neutral_component_mask,
                            neutral_defect_mask,
                            attempt,
                        )
                        if attempt < recorded_attempt and not candidate.get("failures"):
                            errors.append(
                                f"replay {scene_id}: earlier attempt {attempt} unexpectedly passes"
                            )
                        if attempt == recorded_attempt:
                            replay = candidate
                    if replay is None:
                        errors.append(f"replay {scene_id}: no replay result")
                        continue
                    if replay.get("failures"):
                        errors.append(
                            f"replay {scene_id}: recorded attempt fails {replay['failures']}"
                        )
                    image_path = repository_path(scene["image_path"], "replay image")
                    shadow_path = repository_path(
                        scene["shadow_attenuation_mask_path"], "replay shadow"
                    )
                    component_path = repository_path(
                        scene["component_instance_mask_path"], "replay component mask"
                    )
                    defect_path = repository_path(
                        scene["defect_semantic_mask_path"], "replay defect mask"
                    )
                    if replay["image_payload"] != image_path.read_bytes():
                        errors.append(f"replay {scene_id}: JPEG bytes differ")
                    if png_l_payload(replay["shadow_mask"]) != shadow_path.read_bytes():
                        errors.append(f"replay {scene_id}: shadow PNG bytes differ")
                    with Image.open(component_path) as image:
                        published_component = np.asarray(image, dtype=np.uint16).copy()
                    with Image.open(defect_path) as image:
                        published_defect = np.asarray(image, dtype=np.uint8).copy()
                    if not np.array_equal(replay["component_mask"], published_component):
                        errors.append(f"replay {scene_id}: component mask differs")
                    if not np.array_equal(replay["defect_mask"], published_defect):
                        errors.append(f"replay {scene_id}: defect mask differs")
                    if not np.array_equal(
                        replay["component_mask"], neutral_component_mask
                    ) or not np.array_equal(replay["defect_mask"], neutral_defect_mask):
                        errors.append(f"replay {scene_id}: neutral v4 geometry diverged")
                    if integer(scene["scene_seed"], "scene seed") != int(replay["scene_seed"]):
                        errors.append(f"replay {scene_id}: scene seed differs")
                    lighting = lighting_by_scene[scene_id]
                    if canonical_json(lighting.get("sensor_params")) != canonical_json(
                        replay["sensor_params"]
                    ):
                        errors.append(f"replay {scene_id}: sensor parameters differ")
                    if canonical_json(lighting.get("cast_shadow_rows")) != canonical_json(
                        replay["cast_shadow_rows"]
                    ):
                        errors.append(f"replay {scene_id}: cast shadow rows differ")
                    if canonical_json(
                        lighting.get("paired_clean_defect_visibility")
                    ) != canonical_json(replay["defect_visibility"]):
                        errors.append(f"replay {scene_id}: paired-clean visibility differs")
                    expected_metrics = {
                        key: round(float(value), 8)
                        for key, value in replay["metrics"].items()
                    }
                    if canonical_json(lighting.get("metrics")) != canonical_json(
                        expected_metrics
                    ):
                        errors.append(f"replay {scene_id}: lighting metrics differ")
                    replay_effects = {
                        int(row["instance_index"]): row
                        for row in replay["instance_effects"]
                    }
                    replay_visibility = {
                        int(row["instance_index"]): row
                        for row in replay["defect_visibility"]
                    }
                    for published in instances_by_scene[scene_id]:
                        index = int(published["instance_index"])
                        if canonical_json(published.get("lighting_effect")) != canonical_json(
                            replay_effects.get(index)
                        ):
                            errors.append(
                                f"replay {scene_id}/instance-{index}: lighting effect differs"
                            )
                        if canonical_json(
                            published.get("paired_clean_defect_visibility")
                        ) != canonical_json(replay_visibility.get(index)):
                            errors.append(
                                f"replay {scene_id}/instance-{index}: paired-clean visibility differs"
                            )
                    for field, replay_value in replay["metrics"].items():
                        same_float(
                            scene.get(field),
                            replay_value,
                            field,
                            f"replay {scene_id}",
                            errors,
                            2.1e-6,
                        )
                except Exception as error:
                    errors.append(f"replay {scene_id}: exception: {error}")
                if replay_count % 24 == 0:
                    print(f"replayed scenes={replay_count}/768", flush=True)

        if contrast_statistics.get("local_luma_delta"):
            delta_values = np.asarray(
                contrast_statistics["local_luma_delta"], dtype=np.float64
            )
            ratio_values = np.asarray(
                contrast_statistics["local_luma_ratio"], dtype=np.float64
            )
            component_values = np.asarray(
                contrast_statistics["component_mean_luma"], dtype=np.float64
            )
            print(
                "local_contrast_audit: "
                f"instances={len(delta_values)} "
                f"component_luma_min={component_values.min():.6f} "
                f"delta_min={delta_values.min():.6f} "
                f"delta_p01={np.percentile(delta_values, 1):.6f} "
                f"ratio_min={ratio_values.min():.6f} "
                f"ratio_p01={np.percentile(ratio_values, 1):.6f} "
                "threshold_source=pinned_v4_qc",
                flush=True,
            )
        if errors:
            print(f"FAIL: errors={len(errors)}", flush=True)
            for error in errors[:200]:
                print(f"- {error}", flush=True)
            if len(errors) > 200:
                print(f"- ... {len(errors) - 200} additional errors", flush=True)
            return 1
        print(
            "PASS: scenes=768, source_scenes=384x2, components=3840, defects=3360, "
            "proxy_bins=P0-P5x128, rigs=4x192, shadow_regimes=2x384, "
            "condition_cells=48x16, class_cells=9-11, "
            "measured_lux=NULL, calibrated_to_lux=FALSE, lights=2-3, "
            "contact_directional_shadow_mask=PASS, v4_geometry_yolo_copy=PASS, "
            "paired_visibility=PASS, local_v4_contrast=PASS, val_test_parents=0, "
            "deterministic_full_replay=PASS, payload_inventory_all_sheets=PASS",
            flush=True,
        )
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError, ValidationSetupError) as error:
        print(f"SETUP_FAIL: {error}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
