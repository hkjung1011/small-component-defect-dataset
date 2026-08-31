"""Independently validate the synthetic-v4 black-conveyor release.

The validator treats COCO/YOLO files and the scene manifest as untrusted
derivatives.  It pins every source asset, rebuilds the deterministic scene
plan, checks all published files and annotations, then replays every selected
scene attempt through ``generate_synthetic_v4_conveyor.render_scene`` and
requires byte-exact JPEG plus pixel-exact mask equality.
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
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont, features


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "synthetic_v4_conveyor.json"
DEFAULT_RELEASE = ROOT / "synthetic" / "v4_conveyor"
EXPECTED_QC_STATUS = "AUTO_PASS_MULTI_INSTANCE_PAIRED_CLEAN_1024"
EXPECTED_BACKGROUND_LUMA_REFERENCE = "SHADOWLESS_SAME_SENSOR_JPEG"
EXPECTED_CONFIG_SHA256 = "fa5adb968da368a28b85a711cd7b3d0aab6829453361eb686e915af6797cf717"
EXPECTED_GENERATOR_SHA256 = "177cc44d3f14b4466a2b6b4a1cb0329d0eb57c9094b129f0544511be63c05524"
EXPECTED_REQUIREMENTS_SHA256 = "445952014c05a210088d847236be8fab262f66d28109ff8abb6d168d157c2b21"
EXPECTED_HELPER_SHA256 = {
    "scripts/generate_synthetic_v1_450.py": "40f2d733e6f4778fd715c2d9afc9af15745d5b83beebeb60d17a05e9676e6e77",
    "scripts/generate_synthetic_v2_700.py": "abbfdd2ce9fc2538f497d867007c6e1805bf03bf957e8797adefe51af1910ea3",
    "scripts/generate_synthetic_v3_conditions.py": "21194506114004772e28987b82ac8531a0291141f2fc0e665d0532a510bd5098",
}
FLOAT_TOLERANCE = 1.1e-6


def validate_runtime_contract(errors: list[str]) -> None:
    actual = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "libjpeg": features.version("jpg"),
        "zlib": features.version("zlib"),
    }
    expected = {
        "python": "3.14.6",
        "numpy": "2.5.1",
        "pillow": "12.3.0",
        "libjpeg": "8.0",
        "zlib": "1.3.1.zlib-ng",
    }
    for field, expected_value in expected.items():
        if actual[field] != expected_value:
            errors.append(
                f"runtime contract mismatch {field}: "
                f"expected={expected_value!r} actual={actual[field]!r}"
            )


class ValidationSetupError(RuntimeError):
    """Raised when the validator cannot establish a trusted input state."""


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            line = raw.strip()
            if not line:
                raise ValidationSetupError(f"blank JSONL row: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValidationSetupError(
                    f"JSONL row is not an object: {path}:{line_number}"
                )
            records.append(value)
    return records


def repository_path(value: str, field: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValidationSetupError(
            f"{field} must be a non-empty repository-relative path: {value!r}"
        )
    candidate = (ROOT / value).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValidationSetupError(f"{field} escapes repository root: {value}") from error
    return candidate


def require_pinned_file(
    block: dict[str, Any], path_key: str, sha_key: str, label: str
) -> Path:
    raw_path = block.get(path_key)
    expected_sha = block.get(sha_key)
    if not isinstance(raw_path, str):
        raise ValidationSetupError(f"{label}.{path_key} must be a string")
    if (
        not isinstance(expected_sha, str)
        or len(expected_sha) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in expected_sha)
    ):
        raise ValidationSetupError(f"{label}.{sha_key} is not a SHA-256 value")
    path = repository_path(raw_path, f"{label}.{path_key}")
    if not path.is_file():
        raise ValidationSetupError(f"missing pinned file: {path}")
    actual = sha256_file(path)
    if actual != expected_sha.lower():
        raise ValidationSetupError(
            f"pin mismatch {label}.{path_key}: expected={expected_sha.lower()} actual={actual}"
        )
    return path


def integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} is boolean, not integer")
    try:
        result = int(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} is not an integer: {value!r}") from error
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


def boolean(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in ("true", "True", "TRUE", "1", 1):
        return True
    if value in ("false", "False", "FALSE", "0", 0):
        return False
    raise ValueError(f"{field} is not boolean: {value!r}")


def json_field(value: Any, field: str) -> Any:
    if not isinstance(value, str):
        raise ValueError(f"{field} is not a JSON string")
    return json.loads(value)


def int_bbox(value: Any, field: str) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{field} is not a four-element bbox: {value!r}")
    bbox = tuple(integer(item, field) for item in value)
    if bbox[2] <= 0 or bbox[3] <= 0:
        raise ValueError(f"{field} has non-positive dimensions: {bbox}")
    return bbox


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    return x0, y0, x1 - x0, y1 - y0


def encode_coco_uncompressed_rle(mask: np.ndarray) -> dict[str, Any]:
    """Encode a binary mask using COCO's column-major uncompressed RLE."""

    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2:
        raise ValueError("RLE mask must be two-dimensional")
    flat = binary.T.reshape(-1)
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    boundaries = np.concatenate(([0], changes, [flat.size]))
    counts = np.diff(boundaries).astype(np.int64).tolist()
    if int(flat[0]) == 1:
        counts.insert(0, 0)
    return {"size": [int(binary.shape[0]), int(binary.shape[1])], "counts": counts}


def decode_coco_uncompressed_rle(value: Any, field: str) -> np.ndarray:
    """Decode and structurally validate a COCO uncompressed RLE object."""

    if not isinstance(value, dict) or set(value) != {"size", "counts"}:
        raise ValueError(f"{field} is not an uncompressed RLE object")
    size = value["size"]
    counts = value["counts"]
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError(f"{field}.size must contain height,width")
    height = integer(size[0], f"{field}.size[0]")
    width = integer(size[1], f"{field}.size[1]")
    if height <= 0 or width <= 0:
        raise ValueError(f"{field}.size is non-positive")
    if not isinstance(counts, list) or not counts:
        raise ValueError(f"{field}.counts must be a non-empty integer list")
    runs = [integer(item, f"{field}.counts") for item in counts]
    if any(run < 0 for run in runs):
        raise ValueError(f"{field}.counts contains a negative run")
    if sum(runs) != height * width:
        raise ValueError(
            f"{field}.counts total {sum(runs)} != mask size {height * width}"
        )
    flat = np.empty(height * width, dtype=np.uint8)
    cursor = 0
    value_bit = 0
    for run in runs:
        flat[cursor : cursor + run] = value_bit
        cursor += run
        value_bit = 1 - value_bit
    return flat.reshape((width, height)).T.astype(bool)


def bbox_inside(
    bbox: tuple[int, int, int, int], width: int, height: int
) -> bool:
    x, y, box_width, box_height = bbox
    return (
        x >= 0
        and y >= 0
        and box_width > 0
        and box_height > 0
        and x + box_width <= width
        and y + box_height <= height
    )


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if not intersection:
        return 0.0
    return intersection / float(aw * ah + bw * bh - intersection)


def bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    dx = max(ax0 - bx1, bx0 - ax1, 0)
    dy = max(ay0 - by1, by0 - ay1, 0)
    return math.hypot(dx, dy)


def yolo_box_line(
    class_id: int, bbox: tuple[int, int, int, int], width: int, height: int
) -> str:
    x, y, box_width, box_height = bbox
    return (
        f"{class_id} {(x + box_width / 2.0) / width:.10f} "
        f"{(y + box_height / 2.0) / height:.10f} "
        f"{box_width / width:.10f} {box_height / height:.10f}"
    )


def parse_yolo(path: Path, expected_rows: int, errors: list[str]) -> list[str]:
    raw_text = path.read_text(encoding="ascii")
    if not raw_text.endswith("\n"):
        errors.append(f"YOLO file has no terminal newline: {path}")
    lines = [line for line in raw_text.splitlines() if line.strip()]
    if len(lines) != expected_rows:
        errors.append(
            f"YOLO row count mismatch {path}: expected={expected_rows} actual={len(lines)}"
        )
    for line_number, line in enumerate(lines, 1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"invalid YOLO row {path}:{line_number}: {line!r}")
            continue
        try:
            class_id = integer(parts[0], "YOLO class ID")
            values = [finite_float(item, "YOLO coordinate") for item in parts[1:]]
        except ValueError as error:
            errors.append(f"invalid YOLO row {path}:{line_number}: {error}")
            continue
        if class_id < 0:
            errors.append(f"negative YOLO class ID {path}:{line_number}")
        if not all(0.0 <= item <= 1.0 for item in values):
            errors.append(f"YOLO coordinate outside [0,1] {path}:{line_number}")
        if values[2] <= 0.0 or values[3] <= 0.0:
            errors.append(f"YOLO bbox is empty {path}:{line_number}")
        if values[0] - values[2] / 2.0 < -1e-9 or values[0] + values[2] / 2.0 > 1.0 + 1e-9:
            errors.append(f"YOLO x extent outside frame {path}:{line_number}")
        if values[1] - values[3] / 2.0 < -1e-9 or values[1] + values[3] / 2.0 > 1.0 + 1e-9:
            errors.append(f"YOLO y extent outside frame {path}:{line_number}")
    return lines


def require_columns(
    rows: list[dict[str, Any]], required: set[str], label: str, errors: list[str]
) -> None:
    if not rows:
        errors.append(f"{label} is empty")
        return
    missing = required - set(rows[0])
    if missing:
        errors.append(f"{label} missing columns: {sorted(missing)}")


def same_float(
    recorded: Any,
    expected: float,
    field: str,
    prefix: str,
    errors: list[str],
    tolerance: float = FLOAT_TOLERANCE,
) -> None:
    try:
        actual = finite_float(recorded, field)
    except ValueError as error:
        errors.append(f"{prefix}: {error}")
        return
    if not math.isclose(actual, float(expected), rel_tol=0.0, abs_tol=tolerance):
        errors.append(
            f"{prefix}: {field} mismatch recorded={actual!r} expected={expected!r}"
        )


def validate_config_contract(config: dict[str, Any], errors: list[str]) -> None:
    exact = {
        "release": "synthetic-v4-conveyor",
        "generator_version": "4.0.0",
        "qc_gate_version": "multi-instance-paired-clean-1024-v1",
        "task_type": "multi_instance_detection_segmentation",
        "scene_count": 384,
        "scene_width": 1280,
        "scene_height": 720,
        "instances_per_scene": 5,
        "split": "train",
        "training_use": "TRAIN_ONLY_MULTI_SCENE_SYNTHETIC",
        "evaluation_eligible": "NO",
        "classification_eligible": "NO",
        "expected_instances_per_class": 240,
        "expected_instances_per_class_profile": 60,
    }
    for field, expected in exact.items():
        if config.get(field) != expected:
            errors.append(
                f"config contract mismatch {field}: expected={expected!r} actual={config.get(field)!r}"
            )
    expected_runtime_contract = {
        "python": "3.14.6",
        "numpy": "2.5.1",
        "pillow": "12.3.0",
        "libjpeg": "8.0",
        "zlib": "1.3.1.zlib-ng",
        "requirements_path": "requirements-synthetic.txt",
        "requirements_sha256": EXPECTED_REQUIREMENTS_SHA256,
        "helper_scripts": [
            {"path": path, "sha256": sha256}
            for path, sha256 in EXPECTED_HELPER_SHA256.items()
        ],
    }
    if canonical_json(config.get("runtime_contract")) != canonical_json(
        expected_runtime_contract
    ):
        errors.append("config runtime/helper/requirements contract mismatch")
    expected_classes = [
        "normal_proxy",
        "scratch",
        "surface_spot",
        "discoloration",
        "contamination",
        "lead_breakage",
        "body_chip",
        "body_crack",
    ]
    if config.get("classes") != expected_classes:
        errors.append("config class order mismatch")
    expected_yolo = {name: index for index, name in enumerate(expected_classes)}
    expected_coco = {name: index + 1 for index, name in enumerate(expected_classes)}
    expected_semantic = {
        name: index for index, name in enumerate(expected_classes[1:], 1)
    }
    if config.get("yolo_class_ids") != expected_yolo:
        errors.append("config YOLO class mapping mismatch")
    if config.get("coco_category_ids") != expected_coco:
        errors.append("config COCO class mapping mismatch")
    if config.get("defect_semantic_ids") != expected_semantic:
        errors.append("config semantic class mapping mismatch")
    profiles = config.get("lighting_profiles")
    if not isinstance(profiles, list) or len(profiles) != 4 or len(set(profiles)) != 4:
        errors.append("config must contain exactly four unique lighting profiles")
    if config.get("normal_status") != "synthetic_normal_proxy_not_confirmed_ok":
        errors.append("normal_proxy status must explicitly deny confirmed OK")
    source = config.get("source", {})
    if source.get("required_parent_model_split") != "gradient_train":
        errors.append("source parent model split must be gradient_train")
    for field, expected in {
        "expected_parent_count": 168,
        "expected_parent_count_per_defect_class": 24,
        "expected_parent_reuse_count": 10,
        "expected_parent_reuse_per_lighting_profile": [2, 3],
        "expected_parent_reuse_per_placement_slot": 2,
    }.items():
        if source.get(field) != expected:
            errors.append(f"source contract mismatch {field}")
    if config.get("expected_instances_per_class_profile_grid_cell") != [7, 8]:
        errors.append("class/profile/grid-cell expected range must be [7,8]")
    if config.get("background", {}).get("source_domain") != "synthetic_background":
        errors.append("background source_domain mismatch")
    alpha = config.get("component_alpha", {})
    if alpha.get("human_verified") != "YES":
        errors.append("nominal component alpha must be human verified")
    if set(alpha.get("missing_material_classes", [])) != {
        "lead_breakage",
        "body_chip",
    }:
        errors.append("missing-material class inventory mismatch")
    if config.get("layout", {}).get("component_final_visible_long_side_px") != [230, 285]:
        errors.append("final visible component long-side range must be [230,285]")
    qc = config.get("qc", {})
    if qc.get("detector_input_size") != 1024:
        errors.append("detector QC input must be 1024")
    if float(qc.get("maximum_component_overlap_iou", -1.0)) != 0.0:
        errors.append("component overlap gate must be zero")
    for field, expected in {
        "minimum_instance_component_background_luma_delta": 30.0,
        "minimum_instance_component_background_luma_ratio": 1.55,
        "local_belt_annulus_inner_radius_px": 4,
        "local_belt_annulus_outer_radius_px": 24,
    }.items():
        if qc.get(field) != expected:
            errors.append(f"local component/belt QC contract mismatch {field}")


def validate_source_pins(
    config: dict[str, Any], errors: list[str]
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
    set[str],
    set[str],
]:
    source = config["source"]
    manifest_path = require_pinned_file(source, "manifest_path", "manifest_sha256", "source")
    source_config_path = require_pinned_file(source, "config_path", "config_sha256", "source")
    split_path = require_pinned_file(
        source, "split_assignments_path", "split_assignments_sha256", "source"
    )
    require_pinned_file(source, "clean_base_path", "clean_base_sha256", "source")
    background = config["background"]
    background_path = require_pinned_file(background, "path", "sha256", "background")
    prompt_path = require_pinned_file(
        background, "prompt_path", "prompt_sha256", "background"
    )
    alpha = config["component_alpha"]
    alpha_path = require_pinned_file(
        alpha, "nominal_asset_path", "nominal_asset_sha256", "component_alpha"
    )
    overlay_path = require_pinned_file(
        alpha,
        "verification_overlay_path",
        "verification_overlay_sha256",
        "component_alpha",
    )
    source_config = load_json(source_config_path)
    if source_config.get("release") != source.get("release"):
        errors.append("pinned source release/config name mismatch")
    source_rows_list = read_csv(manifest_path)
    split_rows_list = read_csv(split_path)
    source_rows = {row["sample_id"]: row for row in source_rows_list}
    split_rows = {row["sample_id"]: row for row in split_rows_list}
    if len(source_rows) != len(source_rows_list):
        errors.append("source manifest contains duplicate sample IDs")
    if len(split_rows) != len(split_rows_list):
        errors.append("source split assignments contain duplicate sample IDs")
    if set(source_rows) != set(split_rows):
        errors.append("source manifest and split assignment sample inventories differ")
    gradient_ids = {
        sample_id
        for sample_id, row in split_rows.items()
        if row.get("model_split") == source["required_parent_model_split"]
    }
    forbidden_ids = set(split_rows) - gradient_ids
    if len(gradient_ids) != 168:
        errors.append(f"gradient parent count mismatch: {len(gradient_ids)}")
    counts = Counter(source_rows[sample_id]["primary_class"] for sample_id in gradient_ids)
    if set(counts.values()) != {24} or len(counts) != 7:
        errors.append(f"gradient parent class balance mismatch: {dict(counts)}")
    for sample_id in sorted(gradient_ids):
        source_row = source_rows[sample_id]
        split_row = split_rows[sample_id]
        prefix = f"source {sample_id}"
        for field in ("primary_class", "severity", "image_sha256"):
            if source_row.get(field) != split_row.get(field):
                errors.append(f"{prefix}: source/split {field} mismatch")
        try:
            image_path = repository_path(source_row["image_path"], f"{prefix}.image_path")
            mask_path = repository_path(source_row["mask_path"], f"{prefix}.mask_path")
            if not image_path.is_file() or sha256_file(image_path) != source_row["image_sha256"]:
                errors.append(f"{prefix}: source image path/hash mismatch")
            if not mask_path.is_file() or sha256_file(mask_path) != source_row["mask_sha256"]:
                errors.append(f"{prefix}: source mask path/hash mismatch")
        except (KeyError, ValidationSetupError) as error:
            errors.append(f"{prefix}: invalid source path: {error}")
    for path, label, expected_size in (
        (background_path, "background", None),
        (alpha_path, "nominal alpha", (512, 512)),
        (overlay_path, "alpha verification overlay", (512, 512)),
    ):
        try:
            with Image.open(path) as image:
                image.load()
                if expected_size is not None and image.size != expected_size:
                    errors.append(f"{label} dimensions mismatch: {image.size}")
                if image.getexif():
                    errors.append(f"{label} contains EXIF metadata")
                if label == "nominal alpha":
                    values = set(np.unique(np.asarray(image.convert("L"))).tolist())
                    if values - {0, 255}:
                        errors.append(f"nominal alpha is not binary: {sorted(values)[:20]}")
        except Exception as error:  # Pillow raises several decode-specific errors.
            errors.append(f"cannot decode {label}: {error}")
    if not prompt_path.read_text(encoding="utf-8").strip():
        errors.append("background prompt is empty")
    return source_rows, split_rows, gradient_ids, forbidden_ids


def compare_instance_to_source(
    row: dict[str, Any],
    source_rows: dict[str, dict[str, str]],
    gradient_ids: set[str],
    forbidden_ids: set[str],
    config: dict[str, Any],
    errors: list[str],
) -> None:
    scene_id = str(row.get("scene_id", "?"))
    instance_index = row.get("instance_index", "?")
    prefix = f"{scene_id}/instance-{instance_index}"
    parent_id = row.get("source_parent_sample_id")
    if parent_id not in gradient_ids:
        if parent_id in forbidden_ids:
            errors.append(f"{prefix}: validation/test parent leaked: {parent_id}")
        else:
            errors.append(f"{prefix}: unknown/non-gradient parent: {parent_id}")
        return
    source = source_rows[parent_id]
    expected = {
        "source_parent_image_path": source["image_path"],
        "source_parent_mask_path": source["mask_path"],
        "source_parent_image_sha256": source["image_sha256"],
        "source_parent_mask_sha256": source["mask_sha256"],
        "source_parent_class": source["primary_class"],
        "source_parent_severity": source["severity"],
        "source_parent_model_split": "gradient_train",
        "base_group_id": source["base_group_id"],
        "source_specimen_group": source["source_specimen_group"],
        "view": source["view"],
        "family_split_id": parent_id,
        "composition_family_id": scene_id,
        "training_use": config["training_use"],
        "evaluation_eligible": "NO",
        "classification_eligible": "NO",
    }
    for field, value in expected.items():
        if row.get(field) != value:
            errors.append(
                f"{prefix}: {field} mismatch expected={value!r} actual={row.get(field)!r}"
            )
    class_name = row.get("component_status_class")
    envelope_fraction: float | None = None
    normal = class_name == "normal_proxy"
    try:
        normal_flag = boolean(row.get("normal_proxy_from_paired_clean"), "normal flag")
    except ValueError as error:
        errors.append(f"{prefix}: {error}")
        normal_flag = not normal
    if normal_flag != normal:
        errors.append(f"{prefix}: normal proxy flag/class mismatch")
    if normal:
        if row.get("defect_class") is not None or row.get("defect_semantic_id") is not None:
            errors.append(f"{prefix}: normal proxy must not have a defect class")
        if row.get("defect_annotation_id") is not None:
            errors.append(f"{prefix}: normal proxy must not have a defect annotation")
        if row.get("normal_status") != config["normal_status"]:
            errors.append(f"{prefix}: normal proxy status mismatch")
    else:
        if row.get("defect_class") != class_name:
            errors.append(f"{prefix}: defect/status class mismatch")
        if row.get("normal_status") is not None:
            errors.append(f"{prefix}: defect instance has normal status")
        if row.get("defect_semantic_id") != config["defect_semantic_ids"].get(class_name):
            errors.append(f"{prefix}: semantic ID mismatch")
    if row.get("component_yolo_class_id") != config["yolo_class_ids"].get(class_name):
        errors.append(f"{prefix}: component YOLO class mismatch")
    if row.get("component_coco_category_id") != config["coco_category_ids"].get(class_name):
        errors.append(f"{prefix}: component COCO category mismatch")


def validate_instance_photometry(
    row: dict[str, Any], config: dict[str, Any], errors: list[str]
) -> None:
    scene_id = str(row.get("scene_id", "?"))
    index = row.get("instance_index", "?")
    prefix = f"{scene_id}/instance-{index}"
    profile = row.get("lighting_profile")
    ranges = config.get("lighting_profile_ranges", {}).get(profile)
    params = row.get("component_light_params")
    if not isinstance(ranges, dict) or not isinstance(params, dict):
        errors.append(f"{prefix}: missing lighting profile ranges/parameters")
    else:
        if params.get("profile") != profile or params.get("component_only") is not True:
            errors.append(f"{prefix}: component-only lighting metadata mismatch")
        for field, bounds in ranges.items():
            if field not in params:
                errors.append(f"{prefix}: missing lighting parameter {field}")
                continue
            try:
                value = finite_float(params[field], field)
                lower, upper = float(bounds[0]), float(bounds[1])
                if not lower <= value <= upper:
                    errors.append(
                        f"{prefix}: lighting parameter {field}={value} outside [{lower},{upper}]"
                    )
            except (ValueError, TypeError, IndexError) as error:
                errors.append(f"{prefix}: invalid lighting parameter {field}: {error}")
    shadow = row.get("component_shadow_params")
    shadow_config = config["component_shadow"]
    if not isinstance(shadow, dict):
        errors.append(f"{prefix}: missing component shadow parameters")
    else:
        try:
            opacity = finite_float(shadow.get("opacity"), "shadow opacity")
            offset_x = integer(shadow.get("offset_x_px"), "shadow offset_x_px")
            offset_y = integer(shadow.get("offset_y_px"), "shadow offset_y_px")
            if not float(shadow_config["opacity"][0]) <= opacity <= float(
                shadow_config["opacity"][1]
            ):
                errors.append(f"{prefix}: shadow opacity outside config range")
            if not int(shadow_config["offset_x_px"][0]) <= offset_x <= int(
                shadow_config["offset_x_px"][1]
            ):
                errors.append(f"{prefix}: shadow x offset outside config range")
            if not int(shadow_config["offset_y_px"][0]) <= offset_y <= int(
                shadow_config["offset_y_px"][1]
            ):
                errors.append(f"{prefix}: shadow y offset outside config range")
        except ValueError as error:
            errors.append(f"{prefix}: invalid shadow parameters: {error}")
    qc = config["qc"]
    try:
        annulus_area = integer(row.get("local_belt_annulus_area"), "local annulus area")
        local_mean = finite_float(
            row.get("local_background_mean_luma"), "local background mean luma"
        )
        component_mean = finite_float(
            row.get("instance_component_mean_luma"), "instance component mean luma"
        )
        delta = finite_float(
            row.get("instance_component_background_luma_delta"),
            "instance component/background luma delta",
        )
        ratio = finite_float(
            row.get("instance_component_background_luma_ratio"),
            "instance component/background luma ratio",
        )
        if annulus_area < 100:
            errors.append(f"{prefix}: local belt annulus area below 100 pixels")
        if not math.isclose(component_mean - local_mean, delta, abs_tol=2.1e-6):
            errors.append(f"{prefix}: local luma delta arithmetic mismatch")
        expected_ratio = component_mean / max(local_mean, 1e-6)
        if not math.isclose(expected_ratio, ratio, rel_tol=0.0, abs_tol=2.1e-6):
            errors.append(f"{prefix}: local luma ratio arithmetic mismatch")
        if delta < float(qc["minimum_instance_component_background_luma_delta"]):
            errors.append(f"{prefix}: local component/background delta gate failed")
        if ratio < float(qc["minimum_instance_component_background_luma_ratio"]):
            errors.append(f"{prefix}: local component/background ratio gate failed")
    except ValueError as error:
        errors.append(f"{prefix}: invalid local component/belt metric: {error}")

    class_name = row.get("component_status_class")
    try:
        visible_bbox = int_bbox(row.get("visible_bbox_xywh"), "visible_bbox_xywh")
        if visible_bbox[2] < int(qc["minimum_component_width_px"]):
            errors.append(f"{prefix}: visible component width gate failed")
        if visible_bbox[3] < int(qc["minimum_component_height_px"]):
            errors.append(f"{prefix}: visible component height gate failed")
        visible_fraction = finite_float(
            row.get("component_visible_fraction"), "component visible fraction"
        )
        boundary_change = finite_float(
            row.get("visible_render_boundary_change_fraction"),
            "visible render boundary change fraction",
        )
        envelope_fraction = finite_float(
            row.get("defect_envelope_fraction"), "defect envelope fraction"
        )
        if not 0.0 < visible_fraction <= 1.0:
            errors.append(f"{prefix}: component visible fraction outside (0,1]")
        if not 0.0 <= boundary_change <= 1.0:
            errors.append(f"{prefix}: visible render boundary change outside [0,1]")
        if not 0.0 <= envelope_fraction <= 1.0:
            errors.append(f"{prefix}: defect envelope fraction outside [0,1]")
        if (
            class_name not in config["component_alpha"]["missing_material_classes"]
            and visible_fraction < float(qc["minimum_component_visible_fraction"])
        ):
            errors.append(f"{prefix}: component visible fraction gate failed")
    except (ValueError, TypeError, KeyError) as error:
        errors.append(f"{prefix}: invalid component visibility geometry: {error}")

    metrics = row.get("visibility_metrics_1024")
    if class_name == "normal_proxy":
        if metrics is not None:
            errors.append(f"{prefix}: normal proxy has defect visibility metrics")
        if row.get("defect_envelope_fraction") != 1.0:
            errors.append(f"{prefix}: normal proxy defect envelope fraction must be 1.0")
        return
    if class_name not in config["classes"][1:]:
        return
    if not isinstance(metrics, dict):
        errors.append(f"{prefix}: defect visibility_metrics_1024 is missing")
        return
    required_metrics = {
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
    }
    missing_metrics = required_metrics - set(metrics)
    if missing_metrics:
        errors.append(
            f"{prefix}: visibility_metrics_1024 missing fields {sorted(missing_metrics)}"
        )
        return
    try:
        area = integer(metrics["area"], "visibility area")
        x = integer(metrics["bbox_x"], "visibility bbox_x")
        y = integer(metrics["bbox_y"], "visibility bbox_y")
        width = integer(metrics["bbox_w"], "visibility bbox_w")
        height = integer(metrics["bbox_h"], "visibility bbox_h")
        major = integer(metrics["major"], "visibility major")
        minor = integer(metrics["minor"], "visibility minor")
        diagonal = finite_float(metrics["diag"], "visibility diagonal")
        mean_abs_delta = finite_float(
            metrics["mean_abs_delta"], "visibility mean_abs_delta"
        )
        delta_e = finite_float(metrics["delta_e76_p50"], "visibility delta_e76_p50")
        changed_fraction = finite_float(
            metrics["changed_fraction"], "visibility changed_fraction"
        )
        detector_width = int(qc["detector_input_size"])
        detector_height = max(
            1,
            int(round(config["scene_height"] * detector_width / config["scene_width"])),
        )
        if width <= 0 or height <= 0 or area <= 0:
            errors.append(f"{prefix}: empty 1024-input defect visibility geometry")
        if x < 0 or y < 0 or x + width > detector_width or y + height > detector_height:
            errors.append(f"{prefix}: 1024-input defect visibility bbox is truncated")
        if area > width * height:
            errors.append(f"{prefix}: 1024-input defect area exceeds its bbox")
        if major != max(width, height) or minor != min(width, height):
            errors.append(f"{prefix}: 1024-input major/minor arithmetic mismatch")
        if not math.isclose(
            diagonal, round(math.hypot(width, height), 6), rel_tol=0.0, abs_tol=1e-6
        ):
            errors.append(f"{prefix}: 1024-input defect diagonal arithmetic mismatch")
        if area < int(qc["minimum_defect_area_at_detector_input_px"]):
            errors.append(f"{prefix}: 1024-input defect area gate failed")
        if diagonal < float(qc["minimum_defect_diagonal_at_detector_input_px"]):
            errors.append(f"{prefix}: 1024-input defect diagonal gate failed")
        if mean_abs_delta < float(qc["minimum_mean_abs_delta"][class_name]):
            errors.append(f"{prefix}: 1024-input mean absolute delta gate failed")
        if not 0.0 <= delta_e:
            errors.append(f"{prefix}: negative delta E metric")
        if not 0.0 <= changed_fraction <= 1.0:
            errors.append(f"{prefix}: changed fraction outside [0,1]")
        elif changed_fraction < float(qc["minimum_changed_fraction"][class_name]):
            errors.append(f"{prefix}: 1024-input changed fraction gate failed")
        if envelope_fraction is not None and envelope_fraction < 0.98:
            errors.append(f"{prefix}: defect envelope fraction gate failed")
    except (ValueError, TypeError, KeyError, ZeroDivisionError) as error:
        errors.append(f"{prefix}: invalid 1024-input visibility metrics: {error}")


def validate_scene_photometry(
    row: dict[str, Any], config: dict[str, Any], errors: list[str]
) -> None:
    scene_id = str(row.get("scene_id", "?"))
    prefix = f"scene {scene_id}"
    qc = config["qc"]
    try:
        background_mean = finite_float(row.get("background_mean_luma"), "background mean")
        background_std = finite_float(row.get("background_std_luma"), "background std")
        background_p99 = finite_float(row.get("background_p99_luma"), "background p99")
        component_mean = finite_float(row.get("component_mean_luma"), "component mean")
        delta = finite_float(
            row.get("component_background_luma_delta"), "component/background delta"
        )
        ratio = finite_float(
            row.get("component_background_luma_ratio"), "component/background ratio"
        )
        spill_p99 = finite_float(row.get("component_light_spill_p99"), "light spill p99")
        spill_max = finite_float(row.get("component_light_spill_max"), "light spill max")
        if not 0.0 <= background_mean <= 255.0 or not 0.0 <= component_mean <= 255.0:
            errors.append(f"{prefix}: luma mean outside [0,255]")
        if background_std < 0.0 or not 0.0 <= background_p99 <= 255.0:
            errors.append(f"{prefix}: invalid background luma distribution")
        if background_mean > float(qc["maximum_background_mean_luma"]):
            errors.append(f"{prefix}: black-belt background mean gate failed")
        if background_p99 > float(qc["maximum_background_p99_luma"]):
            errors.append(f"{prefix}: black-belt background p99 gate failed")
        if not math.isclose(component_mean - background_mean, delta, abs_tol=2.1e-6):
            errors.append(f"{prefix}: component/background delta arithmetic mismatch")
        expected_ratio = component_mean / max(background_mean, 1e-6)
        if not math.isclose(expected_ratio, ratio, rel_tol=0.0, abs_tol=2.1e-6):
            errors.append(f"{prefix}: component/background ratio arithmetic mismatch")
        if delta < float(qc["minimum_component_background_luma_delta"]):
            errors.append(f"{prefix}: component/background delta gate failed")
        if ratio < float(qc["minimum_component_background_luma_ratio"]):
            errors.append(f"{prefix}: component/background ratio gate failed")
        if not 0.0 <= spill_p99 <= spill_max + 1e-6:
            errors.append(f"{prefix}: invalid component-light spill distribution")
        if spill_p99 > 1.0 or spill_max > 3.0:
            errors.append(f"{prefix}: component-only light spill gate failed")
    except (ValueError, TypeError, KeyError, ZeroDivisionError) as error:
        errors.append(f"{prefix}: invalid scene photometry: {error}")


def compare_replay_instance(
    published: dict[str, Any], replayed: dict[str, Any], config: dict[str, Any], errors: list[str]
) -> None:
    scene_id = published["scene_id"]
    index = published["instance_index"]
    prefix = f"replay {scene_id}/instance-{index}"
    expected_exact = {
        "instance_index": int(replayed["instance_index"]),
        "component_status_class": replayed["class_name"],
        "source_parent_sample_id": replayed["source_parent"]["sample_id"],
        "normal_proxy_from_paired_clean": bool(replayed["normal_proxy_from_paired_clean"]),
        "lighting_profile": replayed["light_params"]["profile"],
        "placement_slot": int(replayed["placement_slot"]),
        "grid_cell": int(replayed["grid_cell"]),
        "left": int(replayed["left"]),
        "top": int(replayed["top"]),
        "nominal_bbox_xywh": list(replayed["nominal_bbox"]),
        "visible_bbox_xywh": list(replayed["visible_bbox"]),
        "nominal_area": int(replayed["nominal_area"]),
        "visible_area": int(replayed["visible_area"]),
        "final_visible_long_side_px": int(replayed["final_visible_long_side_px"]),
        "pre_erosion_nominal_bbox_xywh": (
            None
            if replayed["pre_erosion_nominal_bbox"] is None
            else list(replayed["pre_erosion_nominal_bbox"])
        ),
        "pre_erosion_visible_bbox_xywh": (
            None
            if replayed["pre_erosion_visible_bbox"] is None
            else list(replayed["pre_erosion_visible_bbox"])
        ),
        "pre_erosion_nominal_area": int(replayed["pre_erosion_nominal_area"]),
        "pre_erosion_visible_area": int(replayed["pre_erosion_visible_area"]),
        "defect_bbox_xywh": (
            None if replayed["defect_bbox"] is None else list(replayed["defect_bbox"])
        ),
        "defect_area": int(replayed["defect_area"]),
        "visibility_metrics_1024": replayed["visibility_metrics_1024"],
        "component_light_params": replayed["light_params"],
        "component_shadow_params": replayed["shadow_params"],
        "local_belt_annulus_area": int(replayed["local_belt_annulus_area"]),
    }
    for field, expected in expected_exact.items():
        if canonical_json(published.get(field)) != canonical_json(expected):
            errors.append(f"{prefix}: replay mismatch {field}")
    rounded = {
        "scale": round(float(replayed["scale"]), 10),
        "target_long_side_px": round(float(replayed["target_long_side_px"]), 6),
        "rotation_degrees": round(float(replayed["rotation_degrees"]), 8),
        "component_visible_fraction": round(float(replayed["component_visible_fraction"]), 8),
        "visible_render_boundary_change_fraction": round(
            float(replayed["visible_render_boundary_change_fraction"]), 8
        ),
        "defect_envelope_fraction": round(float(replayed["defect_envelope_fraction"]), 8),
        "local_background_mean_luma": round(
            float(replayed["local_background_mean_luma"]), 6
        ),
        "instance_component_mean_luma": round(
            float(replayed["instance_component_mean_luma"]), 6
        ),
        "instance_component_background_luma_delta": round(
            float(replayed["instance_component_background_luma_delta"]), 6
        ),
        "instance_component_background_luma_ratio": round(
            float(replayed["instance_component_background_luma_ratio"]), 6
        ),
    }
    for field, expected in rounded.items():
        same_float(published.get(field), expected, field, prefix, errors, tolerance=1e-10)


def validate_release_metadata(
    metadata: dict[str, Any],
    config: dict[str, Any],
    config_path: Path,
    release_root: Path,
    manifest_path: Path,
    instances_path: Path,
    component_coco_path: Path,
    defect_coco_path: Path,
    summary_path: Path,
    class_counts: Counter[str],
    profile_counts: Counter[str],
    class_profile_counts: Counter[tuple[str, str]],
    defect_parent_usage: Counter[str],
    normal_parent_usage: Counter[str],
    image_hashes: set[str],
    errors: list[str],
) -> None:
    expected_metadata_keys = {
        "release",
        "generator_version",
        "qc_gate_version",
        "task_type",
        "split",
        "training_use",
        "evaluation_eligible",
        "classification_eligible",
        "config_path",
        "config_sha256",
        "generator_script",
        "generator_script_sha256",
        "runtime_contract",
        "requirements_path",
        "requirements_sha256",
        "helper_script_sha256",
        "scene_count",
        "instances_per_scene",
        "component_instance_count",
        "defect_annotation_count",
        "normal_proxy_instance_count",
        "class_counts",
        "profile_scene_counts",
        "class_profile_counts",
        "class_profile_grid_cell_count_range",
        "profile_grid_cell_counts",
        "defect_parent_count",
        "defect_parent_reuse_range",
        "defect_parent_profile_reuse_range",
        "defect_parent_placement_slot_reuse_range",
        "normal_parent_count",
        "normal_parent_reuse_range",
        "source_release",
        "source_manifest_sha256",
        "source_split_assignments_sha256",
        "background_asset_id",
        "background_sha256",
        "background_prompt_sha256",
        "nominal_component_alpha_sha256",
        "manifest_sha256",
        "instances_sha256",
        "component_coco_sha256",
        "defect_coco_sha256",
        "summary_sha256",
        "overview_contact_sheet_sha256",
        "contact_sheet_sha256",
        "unique_image_sha256_count",
        "tracked_payload_bytes_before_release_json",
        "tracked_payload_bytes",
        "limitations",
    }
    if set(metadata) != expected_metadata_keys:
        errors.append("release metadata key inventory mismatch")
    expected_exact: dict[str, Any] = {
        "release": config["release"],
        "generator_version": config["generator_version"],
        "qc_gate_version": config["qc_gate_version"],
        "task_type": config["task_type"],
        "split": "train",
        "training_use": config["training_use"],
        "evaluation_eligible": "NO",
        "classification_eligible": "NO",
        "config_path": config_path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "config_sha256": sha256_file(config_path),
        "generator_script": "scripts/generate_synthetic_v4_conveyor.py",
        "generator_script_sha256": sha256_file(ROOT / "scripts/generate_synthetic_v4_conveyor.py"),
        "runtime_contract": {
            key: str(config["runtime_contract"][key])
            for key in ("python", "numpy", "pillow", "libjpeg", "zlib")
        },
        "requirements_path": config["runtime_contract"]["requirements_path"],
        "requirements_sha256": config["runtime_contract"]["requirements_sha256"],
        "helper_script_sha256": {
            helper["path"]: helper["sha256"]
            for helper in config["runtime_contract"]["helper_scripts"]
        },
        "scene_count": 384,
        "instances_per_scene": 5,
        "component_instance_count": 1920,
        "defect_annotation_count": 1680,
        "normal_proxy_instance_count": 240,
        "class_counts": {name: class_counts[name] for name in config["classes"]},
        "profile_scene_counts": {
            profile: profile_counts[profile] for profile in config["lighting_profiles"]
        },
        "class_profile_counts": {
            name: {
                profile: class_profile_counts[(name, profile)]
                for profile in config["lighting_profiles"]
            }
            for name in config["classes"]
        },
        "defect_parent_count": 168,
        "defect_parent_reuse_range": [10, 10],
        "normal_parent_count": 168,
        "normal_parent_reuse_range": [1, 2],
        "class_profile_grid_cell_count_range": [7, 8],
        "profile_grid_cell_counts": {
            profile: {str(cell): 60 for cell in range(8)}
            for profile in config["lighting_profiles"]
        },
        "defect_parent_profile_reuse_range": [2, 3],
        "defect_parent_placement_slot_reuse_range": [2, 2],
        "source_release": config["source"]["release"],
        "source_manifest_sha256": config["source"]["manifest_sha256"],
        "source_split_assignments_sha256": config["source"]["split_assignments_sha256"],
        "background_asset_id": config["background"]["asset_id"],
        "background_sha256": config["background"]["sha256"],
        "background_prompt_sha256": config["background"]["prompt_sha256"],
        "nominal_component_alpha_sha256": config["component_alpha"]["nominal_asset_sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "instances_sha256": sha256_file(instances_path),
        "component_coco_sha256": sha256_file(component_coco_path),
        "defect_coco_sha256": sha256_file(defect_coco_path),
        "summary_sha256": sha256_file(summary_path),
        "unique_image_sha256_count": len(image_hashes),
    }
    for field, expected in expected_exact.items():
        if canonical_json(metadata.get(field)) != canonical_json(expected):
            errors.append(
                f"release metadata mismatch {field}: expected={expected!r} actual={metadata.get(field)!r}"
            )
    overview_path = release_root / "contact_sheet.jpg"
    if not overview_path.is_file():
        errors.append("missing overview contact sheet")
    elif metadata.get("overview_contact_sheet_sha256") != sha256_file(overview_path):
        errors.append("overview contact sheet SHA mismatch")
    else:
        try:
            with Image.open(overview_path) as image:
                image.load()
                if image.format != "JPEG" or image.mode != "RGB" or image.size != (1280, 792):
                    errors.append(
                        "overview contact sheet format/mode/dimensions mismatch"
                    )
                if image.getexif():
                    errors.append("overview contact sheet contains EXIF")
                if image.info != {
                    "jfif": 257,
                    "jfif_version": (1, 1),
                    "jfif_unit": 0,
                    "jfif_density": (1, 1),
                }:
                    errors.append("overview contact sheet has ancillary JPEG metadata")
        except OSError as error:
            errors.append(f"cannot decode overview contact sheet: {error}")
    expected_sheets = {
        (
            release_root
            / "annotations"
            / "contact_sheets"
            / f"{profile}_{kind}_96_at_160.jpg"
        ).resolve().relative_to(ROOT.resolve()).as_posix()
        for profile in config["lighting_profiles"]
        for kind in ("raw", "overlay")
    }
    inventory = metadata.get("contact_sheet_sha256")
    if not isinstance(inventory, dict) or set(inventory) != expected_sheets:
        errors.append("release contact-sheet inventory mismatch")
    else:
        for relative, expected_hash in inventory.items():
            try:
                path = repository_path(relative, "contact_sheet_sha256")
                if not path.is_file() or sha256_file(path) != expected_hash:
                    errors.append(f"contact sheet SHA mismatch: {relative}")
                else:
                    with Image.open(path) as image:
                        image.load()
                        if (
                            image.format != "JPEG"
                            or image.mode != "RGB"
                            or image.size != (1920, 832)
                        ):
                            errors.append(
                                f"contact sheet format/mode/dimensions mismatch: {relative}"
                            )
                        if image.getexif():
                            errors.append(f"contact sheet contains EXIF: {relative}")
                        if image.info != {
                            "jfif": 257,
                            "jfif_version": (1, 1),
                            "jfif_unit": 0,
                            "jfif_density": (1, 1),
                        }:
                            errors.append(
                                f"contact sheet has ancillary JPEG metadata: {relative}"
                            )
            except (ValidationSetupError, OSError) as error:
                errors.append(f"invalid contact sheet {relative}: {error}")
    files_before_release = [
        path
        for path in release_root.rglob("*")
        if path.is_file() and path.resolve() != (release_root / "annotations/release.json").resolve()
    ]
    payload_before = sum(path.stat().st_size for path in files_before_release)
    if metadata.get("tracked_payload_bytes_before_release_json") != payload_before:
        errors.append(
            "release tracked_payload_bytes_before_release_json does not match filesystem"
        )
    total_payload = sum(path.stat().st_size for path in release_root.rglob("*") if path.is_file())
    if metadata.get("tracked_payload_bytes") != total_payload:
        errors.append("release tracked_payload_bytes does not match final filesystem size")
    maximum = float(config["qc"]["maximum_new_payload_mib"]) * 1024 * 1024
    if total_payload > maximum:
        errors.append(
            f"release payload exceeds gate: {total_payload / 1024 / 1024:.2f} MiB"
        )
    expected_limitations = [
        "All scenes are train-only composites derived from one synthetic-restored physical base family.",
        "normal_proxy is paired-clean synthetic data and is not confirmed real OK data.",
        "This balanced pilot uses five distinct status classes per scene; it does not model normal-heavy or repeated-status production prevalence.",
        "These scenes must never be repartitioned into validation or test data.",
        "Existing ResNet-18 checkpoints do not process full multi-instance scenes.",
    ]
    if metadata.get("limitations") != expected_limitations:
        errors.append("release limitations/disclosure contract mismatch")


def validate_release_inventory(
    release_root: Path,
    scene_ids: Iterable[str],
    lighting_profiles: Iterable[str],
    errors: list[str],
) -> None:
    expected = {
        ".synthetic_v4_conveyor_marker",
        "contact_sheet.jpg",
        "annotations/manifest.csv",
        "annotations/instances.jsonl",
        "annotations/summary.csv",
        "annotations/release.json",
        "annotations/coco/component_status_train.json",
        "annotations/coco/defects_train.json",
    }
    for profile in lighting_profiles:
        for kind in ("raw", "overlay"):
            expected.add(
                f"annotations/contact_sheets/{profile}_{kind}_96_at_160.jpg"
            )
    for scene_id in scene_ids:
        expected.update(
            {
                f"images/train/{scene_id}.jpg",
                f"masks/component_visible_instances/train/{scene_id}.png",
                f"masks/defect_semantic/train/{scene_id}.png",
                f"labels/yolo_component_status/train/{scene_id}.txt",
                f"labels/yolo_defects/train/{scene_id}.txt",
            }
        )
    if len(expected) != 1936:
        errors.append(f"internal expected release inventory count mismatch: {len(expected)}")
    entries = list(release_root.rglob("*"))
    for entry in entries:
        if entry.is_symlink():
            errors.append(
                "release inventory contains a symlink: "
                f"{entry.relative_to(release_root).as_posix()}"
            )
    actual = {
        entry.relative_to(release_root).as_posix()
        for entry in entries
        if entry.is_file()
    }
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing:
        errors.append(
            f"release inventory missing {len(missing)} file(s): {missing[:12]}"
        )
    if unexpected:
        errors.append(
            f"release inventory has {len(unexpected)} unexpected file(s): {unexpected[:12]}"
        )
    marker = release_root / ".synthetic_v4_conveyor_marker"
    if marker.is_file() and marker.read_bytes() != b"synthetic-v4-conveyor\n":
        errors.append("release marker content mismatch")


def validate_contact_sheet_reconstruction(
    generator: Any,
    release_root: Path,
    scenes: list[dict[str, str]],
    instances_by_scene: dict[str, list[dict[str, Any]]],
    lighting_profiles: Iterable[str],
    errors: list[str],
) -> None:
    font = ImageFont.load_default()
    profiles = list(lighting_profiles)
    for profile in profiles:
        selected = [row for row in scenes if row["lighting_profile"] == profile]
        if len(selected) != 96:
            errors.append(
                f"cannot reconstruct contact sheets for {profile}: {len(selected)} scenes"
            )
            continue
        for overlay in (False, True):
            sheet = Image.new("RGB", (1920, 832), (18, 18, 18))
            draw = ImageDraw.Draw(sheet)
            for index, row in enumerate(selected):
                source_path = repository_path(row["image_path"], "contact sheet image")
                with Image.open(source_path) as source:
                    tile = source.convert("RGB")
                if overlay:
                    overlay_instances = sorted(
                        instances_by_scene[row["scene_id"]],
                        key=lambda item: item["instance_index"],
                    )
                    tile = generator.overlay_scene(tile, overlay_instances, 4)
                tile = tile.resize((160, 90), Image.Resampling.LANCZOS)
                left = (index % 12) * 160
                top = (index // 12) * 104
                sheet.paste(tile, (left, top))
                draw.text(
                    (left + 2, top + 91),
                    row["scene_id"],
                    fill=(225, 225, 225),
                    font=font,
                )
            suffix = "overlay" if overlay else "raw"
            published_path = (
                release_root
                / "annotations"
                / "contact_sheets"
                / f"{profile}_{suffix}_96_at_160.jpg"
            )
            payload = io.BytesIO()
            sheet.save(
                payload,
                format="JPEG",
                quality=90,
                subsampling=1,
                optimize=True,
            )
            if not published_path.is_file() or payload.getvalue() != published_path.read_bytes():
                errors.append(f"contact sheet byte reconstruction mismatch: {published_path}")

    overview = Image.new("RGB", (1280, 792), (15, 15, 15))
    overview_draw = ImageDraw.Draw(overview)
    overview_ready = True
    for profile_index, profile in enumerate(profiles):
        profile_rows = [row for row in scenes if row["lighting_profile"] == profile]
        if len(profile_rows) != 96:
            overview_ready = False
            continue
        for pair_index, row in enumerate((profile_rows[8], profile_rows[56])):
            source_path = repository_path(row["image_path"], "overview image")
            with Image.open(source_path) as source:
                raw = source.convert("RGB")
            overlay_instances = sorted(
                instances_by_scene[row["scene_id"]],
                key=lambda item: item["instance_index"],
            )
            boxed = generator.overlay_scene(raw, overlay_instances, 5)
            for variant_index, tile in enumerate((raw, boxed)):
                column = pair_index * 2 + variant_index
                left = column * 320
                top = profile_index * 198
                tile = tile.resize((320, 180), Image.Resampling.LANCZOS)
                overview.paste(tile, (left, top))
                variant = "raw" if variant_index == 0 else "boxes"
                overview_draw.text(
                    (left + 3, top + 182),
                    f"{profile} | {row['scene_id']} | {variant}",
                    fill=(235, 235, 235),
                    font=font,
                )
    if overview_ready:
        payload = io.BytesIO()
        overview.save(
            payload,
            format="JPEG",
            quality=92,
            subsampling=1,
            optimize=True,
        )
        published_path = release_root / "contact_sheet.jpg"
        if not published_path.is_file() or payload.getvalue() != published_path.read_bytes():
            errors.append("overview contact sheet byte reconstruction mismatch")


def print_failure(errors: list[str]) -> int:
    unique = list(dict.fromkeys(errors))
    print(f"FAIL: {len(unique)} validation error(s)")
    for error in unique[:250]:
        print(f"- {error}")
    if len(unique) > 250:
        print(f"- ... {len(unique) - 250} additional error(s) omitted")
    return 1


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    validate_runtime_contract(errors)
    if errors:
        return print_failure(errors)
    config_path = args.config.resolve()
    release_root = args.release.resolve()
    try:
        config_path.relative_to(ROOT.resolve())
        release_root.relative_to((ROOT / "synthetic").resolve())
    except ValueError as error:
        return print_failure([f"config/release path is outside repository scope: {error}"])
    if not config_path.is_file():
        return print_failure([f"missing config: {config_path}"])
    generator_path = ROOT / "scripts" / "generate_synthetic_v4_conveyor.py"
    pin_errors: list[str] = []
    actual_config_sha256 = sha256_file(config_path)
    if actual_config_sha256 != EXPECTED_CONFIG_SHA256:
        pin_errors.append(
            "immutable config SHA mismatch: "
            f"expected={EXPECTED_CONFIG_SHA256} actual={actual_config_sha256}"
        )
    if not generator_path.is_file():
        pin_errors.append(f"missing pinned generator: {generator_path}")
    else:
        actual_generator_sha256 = sha256_file(generator_path)
        if actual_generator_sha256 != EXPECTED_GENERATOR_SHA256:
            pin_errors.append(
                "immutable generator SHA mismatch: "
                f"expected={EXPECTED_GENERATOR_SHA256} actual={actual_generator_sha256}"
            )
    trusted_import_inputs = {
        "requirements-synthetic.txt": EXPECTED_REQUIREMENTS_SHA256,
        **EXPECTED_HELPER_SHA256,
    }
    for relative, expected_sha256 in trusted_import_inputs.items():
        trusted_path = ROOT / relative
        if not trusted_path.is_file():
            pin_errors.append(f"missing pinned import/runtime input: {trusted_path}")
            continue
        actual_sha256 = sha256_file(trusted_path)
        if actual_sha256 != expected_sha256:
            pin_errors.append(
                f"immutable import/runtime SHA mismatch {relative}: "
                f"expected={expected_sha256} actual={actual_sha256}"
            )
    if pin_errors:
        return print_failure(pin_errors)
    try:
        generator = importlib.import_module("generate_synthetic_v4_conveyor")
        imported_local_modules = {
            "scripts/generate_synthetic_v4_conveyor.py": generator,
            "scripts/generate_synthetic_v2_700.py": generator.v2,
            "scripts/generate_synthetic_v3_conditions.py": generator.v3,
            "scripts/generate_synthetic_v1_450.py": generator.v2.legacy,
        }
        for relative, module in imported_local_modules.items():
            if Path(module.__file__).resolve() != (ROOT / relative).resolve():
                raise ValidationSetupError(
                    f"imported local module path mismatch: {relative} -> {module.__file__}"
                )
        if generator.v3.v2 is not generator.v2 or generator.v3.legacy is not generator.v2.legacy:
            raise ValidationSetupError("local generator import graph is not canonical")
        config = load_json(config_path)
        if not isinstance(config, dict):
            raise TypeError("config must be a JSON object")
        validate_config_contract(config, errors)
        runtime_contract = config["runtime_contract"]
        require_pinned_file(
            runtime_contract,
            "requirements_path",
            "requirements_sha256",
            "runtime_contract",
        )
        for index, helper in enumerate(runtime_contract["helper_scripts"]):
            require_pinned_file(helper, "path", "sha256", f"runtime helper {index}")
        source_rows, split_rows, gradient_ids, forbidden_ids = validate_source_pins(
            config, errors
        )
        context = generator.load_context(config_path, release_root)
    except Exception as error:
        return print_failure(errors + [f"validation setup failed: {error}"])

    if context.config_sha256 != sha256_file(config_path):
        errors.append("generator context config SHA mismatch")
    plan_by_scene = {plan["scene_id"]: plan for plan in context.plans}
    if len(plan_by_scene) != 384:
        errors.append(f"deterministic plan count mismatch: {len(plan_by_scene)}")

    manifest_path = release_root / "annotations" / "manifest.csv"
    instances_path = release_root / "annotations" / "instances.jsonl"
    component_coco_path = (
        release_root / "annotations" / "coco" / "component_status_train.json"
    )
    defect_coco_path = release_root / "annotations" / "coco" / "defects_train.json"
    summary_path = release_root / "annotations" / "summary.csv"
    metadata_path = release_root / "annotations" / "release.json"
    required_files = (
        manifest_path,
        instances_path,
        component_coco_path,
        defect_coco_path,
        summary_path,
        metadata_path,
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        return print_failure(errors + [f"missing release file: {path}" for path in missing])
    validate_release_inventory(
        release_root,
        (plan["scene_id"] for plan in context.plans),
        config["lighting_profiles"],
        errors,
    )
    try:
        scenes = read_csv(manifest_path)
        instances = read_jsonl(instances_path)
        component_coco = load_json(component_coco_path)
        defect_coco = load_json(defect_coco_path)
        summary_rows = read_csv(summary_path)
        metadata = load_json(metadata_path)
    except Exception as error:
        return print_failure(errors + [f"cannot parse release annotations: {error}"])
    if not isinstance(metadata, dict):
        errors.append("release metadata root is not an object")
        metadata = {}

    expected_scene_columns = [
        "scene_id", "image_id", "image_path", "component_instance_mask_path",
        "defect_semantic_mask_path", "component_yolo_path", "defect_yolo_path",
        "domain", "split", "task_type", "training_use", "evaluation_eligible",
        "classification_eligible", "lighting_profile", "background_asset_id",
        "scene_seed", "attempt", "instance_count", "distinct_class_count",
        "component_status_labels", "source_parent_ids", "composition_family_id",
        "width", "height", "image_sha256", "component_mask_sha256",
        "defect_mask_sha256", "component_yolo_sha256", "defect_yolo_sha256",
        "background_mean_luma", "background_std_luma", "background_p99_luma",
        "component_mean_luma", "component_background_luma_delta",
        "component_background_luma_ratio", "component_light_spill_p99",
        "component_light_spill_max", "background_luma_reference", "config_sha256",
        "generator_version",
        "qc_gate_version", "qc_status", "human_verified", "background_params_json",
        "sensor_params_json", "shadow_params_json",
    ]
    require_columns(scenes, set(expected_scene_columns), "manifest", errors)
    manifest_schema_valid = bool(scenes)
    if scenes and list(scenes[0]) != expected_scene_columns:
        errors.append("manifest header/order is not the exact public schema")
        manifest_schema_valid = False
    for row_number, scene in enumerate(scenes, 1):
        if list(scene) != expected_scene_columns:
            errors.append(f"manifest row {row_number} key inventory/order mismatch")
            manifest_schema_valid = False
    expected_instance_keys = {
        "scene_id",
        "image_id",
        "instance_index",
        "component_annotation_id",
        "defect_annotation_id",
        "component_status_class",
        "component_yolo_class_id",
        "component_coco_category_id",
        "defect_class",
        "defect_semantic_id",
        "source_parent_sample_id",
        "source_parent_image_path",
        "source_parent_mask_path",
        "source_parent_image_sha256",
        "source_parent_mask_sha256",
        "source_parent_class",
        "source_parent_severity",
        "source_parent_model_split",
        "normal_proxy_from_paired_clean",
        "normal_status",
        "base_group_id",
        "source_specimen_group",
        "view",
        "family_split_id",
        "composition_family_id",
        "lighting_profile",
        "placement_slot",
        "grid_cell",
        "scale",
        "target_long_side_px",
        "final_visible_long_side_px",
        "rotation_degrees",
        "left",
        "top",
        "nominal_bbox_xywh",
        "visible_bbox_xywh",
        "nominal_area",
        "visible_area",
        "pre_erosion_nominal_bbox_xywh",
        "pre_erosion_visible_bbox_xywh",
        "pre_erosion_nominal_area",
        "pre_erosion_visible_area",
        "visible_render_boundary_change_fraction",
        "component_visible_fraction",
        "defect_bbox_xywh",
        "defect_area",
        "defect_envelope_fraction",
        "visibility_metrics_1024",
        "component_light_params",
        "component_shadow_params",
        "local_belt_annulus_area",
        "local_background_mean_luma",
        "instance_component_mean_luma",
        "instance_component_background_luma_delta",
        "instance_component_background_luma_ratio",
        "training_use",
        "evaluation_eligible",
        "classification_eligible",
    }
    instance_schema_valid = bool(instances)
    for row_number, instance in enumerate(instances, 1):
        if set(instance) != expected_instance_keys:
            errors.append(f"instance JSONL row {row_number} key inventory mismatch")
            instance_schema_valid = False
    if not manifest_schema_valid or not instance_schema_valid:
        return print_failure(errors)
    if len(scenes) != 384:
        errors.append(f"manifest scene count mismatch: {len(scenes)}")
    if len(instances) != 1920:
        errors.append(f"instance row count mismatch: {len(instances)}")
    scene_ids = [row.get("scene_id", "") for row in scenes]
    if len(set(scene_ids)) != len(scene_ids):
        errors.append("manifest contains duplicate scene IDs")
    if set(scene_ids) != set(plan_by_scene):
        errors.append("manifest scene inventory differs from deterministic plan")
    image_ids: list[int] = []
    image_hashes: set[str] = set()
    scene_by_id = {row["scene_id"]: row for row in scenes}
    instances_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    component_annotation_ids: set[int] = set()
    defect_annotation_ids: set[int] = set()
    class_counts: Counter[str] = Counter()
    profile_counts: Counter[str] = Counter()
    class_profile_counts: Counter[tuple[str, str]] = Counter()
    class_profile_slot_counts: Counter[tuple[str, str, int]] = Counter()
    class_profile_grid_counts: Counter[tuple[str, str, int]] = Counter()
    profile_grid_counts: Counter[tuple[str, int]] = Counter()
    defect_parent_usage: Counter[str] = Counter()
    defect_parent_profile_usage: Counter[tuple[str, str]] = Counter()
    defect_parent_slot_usage: Counter[tuple[str, int]] = Counter()
    normal_parent_usage: Counter[str] = Counter()

    for row_number, instance in enumerate(instances, 1):
        scene_id = instance.get("scene_id")
        if scene_id not in scene_by_id:
            errors.append(f"instance row {row_number}: unknown scene_id {scene_id!r}")
            continue
        instances_by_scene[scene_id].append(instance)
        for field in (
            "image_id",
            "instance_index",
            "component_annotation_id",
            "component_yolo_class_id",
            "component_coco_category_id",
        ):
            if type(instance.get(field)) is not int:
                errors.append(
                    f"instance row {row_number}: {field} must be a JSON integer"
                )
        for field in ("defect_annotation_id", "defect_semantic_id"):
            value = instance.get(field)
            if value is not None and type(value) is not int:
                errors.append(
                    f"instance row {row_number}: {field} must be integer or null"
                )
        compare_instance_to_source(
            instance, source_rows, gradient_ids, forbidden_ids, config, errors
        )
        validate_instance_photometry(instance, config, errors)
        class_name = instance.get("component_status_class")
        profile = instance.get("lighting_profile")
        if class_name not in config["classes"]:
            errors.append(f"{scene_id}: unknown component class {class_name!r}")
            continue
        if profile not in config["lighting_profiles"]:
            errors.append(f"{scene_id}: unknown lighting profile {profile!r}")
            continue
        class_counts[class_name] += 1
        class_profile_counts[(class_name, profile)] += 1
        try:
            slot = integer(instance.get("placement_slot"), "placement_slot")
            grid_cell = integer(instance.get("grid_cell"), "grid_cell")
            index = integer(instance.get("instance_index"), "instance_index")
            image_id = integer(instance.get("image_id"), "image_id")
            component_annotation_id = integer(
                instance.get("component_annotation_id"), "component_annotation_id"
            )
            if component_annotation_id in component_annotation_ids:
                errors.append(f"duplicate component annotation ID {component_annotation_id}")
            component_annotation_ids.add(component_annotation_id)
            if instance.get("defect_annotation_id") is not None:
                defect_annotation_id = integer(
                    instance["defect_annotation_id"], "defect_annotation_id"
                )
                if defect_annotation_id in defect_annotation_ids:
                    errors.append(f"duplicate defect annotation ID {defect_annotation_id}")
                defect_annotation_ids.add(defect_annotation_id)
            if not 0 <= slot < 5 or not 0 <= grid_cell < 8 or not 1 <= index <= 5:
                errors.append(
                    f"{scene_id}: invalid slot/grid/index {slot}/{grid_cell}/{index}"
                )
            if image_id != integer(scene_by_id[scene_id]["image_id"], "scene image_id"):
                errors.append(f"{scene_id}: instance image ID mismatch")
            class_profile_slot_counts[(class_name, profile, slot)] += 1
            class_profile_grid_counts[(class_name, profile, grid_cell)] += 1
            profile_grid_counts[(profile, grid_cell)] += 1
        except ValueError as error:
            errors.append(f"{scene_id}: invalid instance numeric field: {error}")
        parent_id = instance.get("source_parent_sample_id")
        if class_name == "normal_proxy":
            normal_parent_usage[parent_id] += 1
        else:
            defect_parent_usage[parent_id] += 1
            defect_parent_profile_usage[(parent_id, profile)] += 1
            try:
                defect_parent_slot_usage[(parent_id, int(instance["placement_slot"]))] += 1
            except (KeyError, TypeError, ValueError):
                pass

    if len(component_annotation_ids) != 1920 or component_annotation_ids != set(range(1, 1921)):
        errors.append("component annotation ID inventory must be exactly 1..1920")
    if len(defect_annotation_ids) != 1680 or defect_annotation_ids != set(range(1, 1681)):
        errors.append("defect annotation ID inventory must be exactly 1..1680")
    for class_name in config["classes"]:
        if class_counts[class_name] != 240:
            errors.append(f"class count mismatch {class_name}: {class_counts[class_name]}")
        for profile in config["lighting_profiles"]:
            if class_profile_counts[(class_name, profile)] != 60:
                errors.append(
                    f"class/profile count mismatch {class_name}/{profile}: "
                    f"{class_profile_counts[(class_name, profile)]}"
                )
            for slot in range(5):
                if class_profile_slot_counts[(class_name, profile, slot)] != 12:
                    errors.append(
                        f"class/profile/slot count mismatch {class_name}/{profile}/{slot}"
                    )
            grid_values = [
                class_profile_grid_counts[(class_name, profile, cell)]
                for cell in range(8)
            ]
            if set(grid_values) - {7, 8} or Counter(grid_values) != Counter({7: 4, 8: 4}):
                errors.append(
                    f"class/profile/grid balance mismatch {class_name}/{profile}: {grid_values}"
                )
    for profile in config["lighting_profiles"]:
        for cell in range(8):
            if profile_grid_counts[(profile, cell)] != 60:
                errors.append(
                    f"profile/grid count mismatch {profile}/{cell}: "
                    f"{profile_grid_counts[(profile, cell)]}"
                )
    if len(defect_parent_usage) != 168 or set(defect_parent_usage.values()) != {10}:
        errors.append(
            f"defect parent reuse mismatch: parents={len(defect_parent_usage)} "
            f"range={min(defect_parent_usage.values(), default=0)}.."
            f"{max(defect_parent_usage.values(), default=0)}"
        )
    if set(defect_parent_usage) != gradient_ids:
        errors.append("defect parent inventory does not equal gradient-train allowlist")
    if (
        len(defect_parent_profile_usage) != 168 * 4
        or set(defect_parent_profile_usage.values()) != {2, 3}
    ):
        errors.append(
            "defect parent/profile reuse must cover 168x4 combinations with counts 2..3"
        )
    if (
        len(defect_parent_slot_usage) != 168 * 5
        or set(defect_parent_slot_usage.values()) != {2}
    ):
        errors.append(
            "defect parent/placement-slot reuse must cover 168x5 combinations exactly twice"
        )
    if len(normal_parent_usage) != 168 or set(normal_parent_usage.values()) != {1, 2}:
        errors.append(
            f"normal parent reuse mismatch: parents={len(normal_parent_usage)} "
            f"range={min(normal_parent_usage.values(), default=0)}.."
            f"{max(normal_parent_usage.values(), default=0)}"
        )
    if Counter(normal_parent_usage.values()) != Counter({1: 96, 2: 72}):
        errors.append("normal parent reuse distribution must be 96x1 plus 72x2")
    if set(normal_parent_usage) != gradient_ids:
        errors.append("normal parent inventory does not equal gradient-train allowlist")
    if (set(defect_parent_usage) | set(normal_parent_usage)) & forbidden_ids:
        errors.append("validation/test source parent leakage detected")

    expected_dirs = {
        "image_path": release_root / "images" / "train",
        "component_instance_mask_path": release_root / "masks/component_visible_instances/train",
        "defect_semantic_mask_path": release_root / "masks/defect_semantic/train",
        "component_yolo_path": release_root / "labels/yolo_component_status/train",
        "defect_yolo_path": release_root / "labels/yolo_defects/train",
    }
    expected_suffixes = {
        "image_path": ".jpg",
        "component_instance_mask_path": ".png",
        "defect_semantic_mask_path": ".png",
        "component_yolo_path": ".txt",
        "defect_yolo_path": ".txt",
    }
    seen_paths: dict[str, set[Path]] = {field: set() for field in expected_dirs}
    pair_counts_by_profile: dict[str, Counter[tuple[str, str]]] = {
        profile: Counter() for profile in config["lighting_profiles"]
    }
    for scene_index, scene in enumerate(scenes):
        scene_id = scene["scene_id"]
        prefix = f"scene {scene_id}"
        validate_scene_photometry(scene, config, errors)
        plan = plan_by_scene.get(scene_id)
        if plan is None:
            continue
        scene_instances = sorted(
            instances_by_scene.get(scene_id, []), key=lambda item: item.get("instance_index", 0)
        )
        if len(scene_instances) != 5:
            errors.append(f"{prefix}: expected 5 instances, got {len(scene_instances)}")
            continue
        try:
            instance_indices = [
                integer(item.get("instance_index"), "instance_index")
                for item in scene_instances
            ]
            if instance_indices != [1, 2, 3, 4, 5]:
                errors.append(
                    f"{prefix}: instance index inventory/order mismatch {instance_indices}"
                )
        except ValueError as error:
            errors.append(f"{prefix}: invalid instance index inventory: {error}")
        try:
            image_id = integer(scene["image_id"], "image_id")
            image_ids.append(image_id)
            if image_id != int(plan["image_id"]):
                errors.append(f"{prefix}: deterministic image ID mismatch")
            if integer(scene["scene_seed"], "scene_seed") != int(plan["scene_seed"]):
                errors.append(f"{prefix}: deterministic scene seed mismatch")
            if not 0 <= integer(scene["attempt"], "attempt") < int(config["max_scene_attempts"]):
                errors.append(f"{prefix}: attempt outside configured range")
            if integer(scene["instance_count"], "instance_count") != 5:
                errors.append(f"{prefix}: instance_count must be 5")
            if integer(scene["distinct_class_count"], "distinct_class_count") != 5:
                errors.append(f"{prefix}: distinct_class_count must be 5")
            if (integer(scene["width"], "width"), integer(scene["height"], "height")) != (1280, 720):
                errors.append(f"{prefix}: manifest dimensions mismatch")
        except ValueError as error:
            errors.append(f"{prefix}: invalid manifest numeric field: {error}")
        static_expected = {
            "domain": "synthetic_black_conveyor_multi_instance",
            "split": "train",
            "task_type": config["task_type"],
            "training_use": config["training_use"],
            "evaluation_eligible": "NO",
            "classification_eligible": "NO",
            "lighting_profile": plan["lighting_profile"],
            "background_asset_id": config["background"]["asset_id"],
            "composition_family_id": scene_id,
            "config_sha256": sha256_file(config_path),
            "generator_version": config["generator_version"],
            "qc_gate_version": config["qc_gate_version"],
            "qc_status": EXPECTED_QC_STATUS,
            "human_verified": "NO",
            "background_luma_reference": EXPECTED_BACKGROUND_LUMA_REFERENCE,
        }
        for field, expected in static_expected.items():
            if scene.get(field) != expected:
                errors.append(f"{prefix}: {field} mismatch")
        profile_counts[scene["lighting_profile"]] += 1
        labels = [item["component_status_class"] for item in scene_instances]
        parent_ids = [item["source_parent_sample_id"] for item in scene_instances]
        for item in scene_instances:
            if item.get("lighting_profile") != scene["lighting_profile"]:
                errors.append(f"{prefix}: instance/scene lighting profile mismatch")
            if item.get("composition_family_id") != scene_id:
                errors.append(f"{prefix}: instance composition family mismatch")
        if len(set(labels)) != 5:
            errors.append(f"{prefix}: component classes are not distinct")
        if len(set(parent_ids)) != 5:
            errors.append(f"{prefix}: duplicate source parent within scene")
        if scene["component_status_labels"].split("|") != labels:
            errors.append(f"{prefix}: manifest component label order mismatch")
        if scene["source_parent_ids"].split("|") != parent_ids:
            errors.append(f"{prefix}: manifest parent order mismatch")
        planned_labels = [item["class_name"] for item in plan["instances"]]
        planned_parents = [item["source_parent_sample_id"] for item in plan["instances"]]
        if labels != planned_labels or parent_ids != planned_parents:
            errors.append(f"{prefix}: manifest/instance rows differ from deterministic plan")
        pair_counts_by_profile[scene["lighting_profile"]].update(
            combinations(sorted(labels), 2)
        )
        resolved: dict[str, Path] = {}
        for field, expected_dir in expected_dirs.items():
            try:
                path = repository_path(scene[field], f"{prefix}.{field}")
                resolved[field] = path
                expected_path = (expected_dir / f"{scene_id}{expected_suffixes[field]}").resolve()
                if path.resolve() != expected_path:
                    errors.append(f"{prefix}: {field} canonical path mismatch")
                if path in seen_paths[field]:
                    errors.append(f"{prefix}: duplicate {field} path")
                seen_paths[field].add(path)
                if not path.is_file():
                    errors.append(f"{prefix}: missing {field} file")
            except ValidationSetupError as error:
                errors.append(f"{prefix}: {error}")
        if len(resolved) != len(expected_dirs) or any(not path.is_file() for path in resolved.values()):
            continue
        hash_fields = {
            "image_path": "image_sha256",
            "component_instance_mask_path": "component_mask_sha256",
            "defect_semantic_mask_path": "defect_mask_sha256",
            "component_yolo_path": "component_yolo_sha256",
            "defect_yolo_path": "defect_yolo_sha256",
        }
        for path_field, hash_field in hash_fields.items():
            actual_hash = sha256_file(resolved[path_field])
            if scene.get(hash_field) != actual_hash:
                errors.append(f"{prefix}: {hash_field} mismatch")
            if path_field == "image_path":
                if actual_hash in image_hashes:
                    errors.append(f"{prefix}: duplicate image SHA-256")
                image_hashes.add(actual_hash)
        try:
            with Image.open(resolved["image_path"]) as image:
                if image.format != "JPEG" or image.size != (1280, 720) or image.mode != "RGB":
                    errors.append(
                        f"{prefix}: invalid scene image format/size/mode "
                        f"{image.format}/{image.size}/{image.mode}"
                    )
                if image.getexif():
                    errors.append(f"{prefix}: scene image contains EXIF")
                image.load()
            with Image.open(resolved["component_instance_mask_path"]) as mask_image:
                if (
                    mask_image.format != "PNG"
                    or mask_image.mode != "I;16"
                    or mask_image.size != (1280, 720)
                ):
                    errors.append(f"{prefix}: invalid component mask format/mode/size")
                if mask_image.getexif():
                    errors.append(f"{prefix}: component mask contains EXIF")
                if mask_image.info:
                    errors.append(f"{prefix}: component mask has ancillary PNG metadata")
                if mask_image.mode == "I;16":
                    component_mask = np.asarray(mask_image, dtype=np.uint16).copy()
                else:
                    component_mask = np.zeros((720, 1280), dtype=np.uint16)
            with Image.open(resolved["defect_semantic_mask_path"]) as mask_image:
                if (
                    mask_image.format != "PNG"
                    or mask_image.mode != "L"
                    or mask_image.size != (1280, 720)
                ):
                    errors.append(f"{prefix}: invalid defect mask format/mode/size")
                if mask_image.getexif():
                    errors.append(f"{prefix}: defect mask contains EXIF")
                if mask_image.info:
                    errors.append(f"{prefix}: defect mask has ancillary PNG metadata")
                if mask_image.mode == "L":
                    defect_mask = np.asarray(mask_image, dtype=np.uint8).copy()
                else:
                    defect_mask = np.zeros((720, 1280), dtype=np.uint8)
        except Exception as error:
            errors.append(f"{prefix}: image/mask decode failed: {error}")
            continue
        component_values = set(np.unique(component_mask).tolist())
        if component_values != {0, 1, 2, 3, 4, 5}:
            errors.append(f"{prefix}: component mask IDs mismatch: {sorted(component_values)}")
        expected_semantic_values = {0} | {
            int(config["defect_semantic_ids"][name])
            for name in labels
            if name != "normal_proxy"
        }
        semantic_values = set(np.unique(defect_mask).tolist())
        if semantic_values != expected_semantic_values:
            errors.append(
                f"{prefix}: semantic mask IDs mismatch expected={sorted(expected_semantic_values)} "
                f"actual={sorted(semantic_values)}"
            )
        nominal_bboxes: list[tuple[int, int, int, int]] = []
        expected_component_lines: list[str] = []
        expected_defect_lines: list[str] = []
        for item in scene_instances:
            index = int(item["instance_index"])
            class_name = item["component_status_class"]
            try:
                nominal_bbox = int_bbox(item["nominal_bbox_xywh"], "nominal_bbox_xywh")
                visible_bbox = int_bbox(item["visible_bbox_xywh"], "visible_bbox_xywh")
            except ValueError as error:
                errors.append(f"{prefix}/instance-{index}: {error}")
                continue
            nominal_bboxes.append(nominal_bbox)
            if not bbox_inside(nominal_bbox, 1280, 720) or not bbox_inside(visible_bbox, 1280, 720):
                errors.append(f"{prefix}/instance-{index}: component bbox is truncated")
            visible_pixels = component_mask == index
            actual_visible_bbox = mask_bbox(visible_pixels)
            if actual_visible_bbox != visible_bbox:
                errors.append(f"{prefix}/instance-{index}: visible bbox/mask mismatch")
            if int(visible_pixels.sum()) != int(item["visible_area"]):
                errors.append(f"{prefix}/instance-{index}: visible area/mask mismatch")
            final_long_side = max(visible_bbox[2], visible_bbox[3])
            if int(item.get("final_visible_long_side_px", -1)) != final_long_side:
                errors.append(
                    f"{prefix}/instance-{index}: final visible long-side value mismatch"
                )
            final_bounds = config["layout"]["component_final_visible_long_side_px"]
            if not int(final_bounds[0]) <= final_long_side <= int(final_bounds[1]):
                errors.append(
                    f"{prefix}/instance-{index}: final visible long side "
                    f"{final_long_side} outside {final_bounds}"
                )
            expected_component_lines.append(
                yolo_box_line(config["yolo_class_ids"][class_name], visible_bbox, 1280, 720)
            )
            semantic_id = config["defect_semantic_ids"].get(class_name)
            class_pixels = np.zeros_like(defect_mask, dtype=bool)
            if semantic_id is not None:
                class_pixels = defect_mask == semantic_id
                try:
                    defect_bbox = int_bbox(item["defect_bbox_xywh"], "defect_bbox_xywh")
                except ValueError as error:
                    errors.append(f"{prefix}/instance-{index}: {error}")
                    continue
                if mask_bbox(class_pixels) != defect_bbox:
                    errors.append(f"{prefix}/instance-{index}: defect bbox/mask mismatch")
                if int(class_pixels.sum()) != int(item["defect_area"]):
                    errors.append(f"{prefix}/instance-{index}: defect area/mask mismatch")
                if not bbox_inside(defect_bbox, 1280, 720):
                    errors.append(f"{prefix}/instance-{index}: defect bbox truncated")
                if class_name not in config["component_alpha"]["missing_material_classes"]:
                    outside_visible = int(np.logical_and(class_pixels, ~visible_pixels).sum())
                    if outside_visible:
                        errors.append(
                            f"{prefix}/instance-{index}: surface defect has "
                            f"{outside_visible} pixels outside final rendered component mask"
                        )
                expected_defect_lines.append(
                    yolo_box_line(semantic_id - 1, defect_bbox, 1280, 720)
                )
            else:
                if item.get("defect_bbox_xywh") is not None or int(item.get("defect_area", 0)) != 0:
                    errors.append(f"{prefix}/instance-{index}: normal proxy has defect geometry")
        for left, right in combinations(nominal_bboxes, 2):
            if bbox_iou(left, right) > 0.0:
                errors.append(f"{prefix}: nominal component bboxes overlap")
            minimum_gap = float(config["layout"].get("minimum_gap_px", 0))
            if bbox_gap(left, right) + 1e-9 < minimum_gap:
                errors.append(
                    f"{prefix}: component bbox gap {bbox_gap(left, right):.3f} < {minimum_gap}"
                )
        component_lines = parse_yolo(
            resolved["component_yolo_path"], 5, errors
        )
        defect_lines = parse_yolo(
            resolved["defect_yolo_path"], len(expected_defect_lines), errors
        )
        if component_lines != expected_component_lines:
            errors.append(f"{prefix}: component YOLO does not roundtrip from instance bboxes")
        if defect_lines != expected_defect_lines:
            errors.append(f"{prefix}: defect YOLO does not roundtrip from semantic mask bboxes")

    if image_ids != list(range(1, 385)):
        errors.append("manifest image IDs/order must be exactly 1..384")
    if len(image_hashes) != 384:
        errors.append(f"unique image SHA count mismatch: {len(image_hashes)}")
    for profile in config["lighting_profiles"]:
        if profile_counts[profile] != 96:
            errors.append(f"profile scene count mismatch {profile}: {profile_counts[profile]}")
        pair_values = list(pair_counts_by_profile[profile].values())
        if len(pair_values) != 28 or min(pair_values, default=0) != 34 or max(pair_values, default=0) != 35:
            errors.append(
                f"class pair balance mismatch {profile}: "
                f"pairs={len(pair_values)} range={min(pair_values, default=0)}..{max(pair_values, default=0)}"
            )
    for field, expected_dir in expected_dirs.items():
        actual_files = {path.resolve() for path in expected_dir.glob("*") if path.is_file()}
        if actual_files != seen_paths[field]:
            errors.append(
                f"canonical directory inventory mismatch {field}: "
                f"manifest={len(seen_paths[field])} disk={len(actual_files)}"
            )

    try:
        validate_contact_sheet_reconstruction(
            generator,
            release_root,
            scenes,
            instances_by_scene,
            config["lighting_profiles"],
            errors,
        )
    except Exception as error:
        errors.append(f"contact sheet reconstruction exception: {error}")

    # COCO is independently compared against the canonical instance rows.
    for label, document, expected_categories in (
        (
            "component COCO",
            component_coco,
            [
                {"id": config["coco_category_ids"][name], "name": name}
                for name in config["classes"]
            ],
        ),
        (
            "defect COCO",
            defect_coco,
            [
                {"id": config["defect_semantic_ids"][name], "name": name}
                for name in config["classes"][1:]
            ],
        ),
    ):
        if not isinstance(document, dict):
            errors.append(f"{label} root is not an object")
            continue
        if set(document) != {"info", "licenses", "images", "annotations", "categories"}:
            errors.append(f"{label} root key inventory mismatch")
        expected_info = {
            "description": "Synthetic black-conveyor multi-instance train-only release",
            "version": config["generator_version"],
            "year": 2026,
        }
        if (
            canonical_json(document.get("info")) != canonical_json(expected_info)
            or document.get("licenses") != []
        ):
            errors.append(f"{label} info/license contract mismatch")
        if canonical_json(document.get("categories")) != canonical_json(expected_categories):
            errors.append(f"{label} category mapping mismatch")
        images = document.get("images")
        if not isinstance(images, list) or len(images) != 384:
            errors.append(f"{label} image count mismatch")
        else:
            for image, scene in zip(images, scenes, strict=False):
                expected_image = {
                    "id": int(scene["image_id"]),
                    "file_name": repository_path(
                        scene["image_path"], f"{label}.image_path"
                    ).relative_to(release_root).as_posix(),
                    "width": 1280,
                    "height": 720,
                    "scene_id": scene["scene_id"],
                }
                if canonical_json(image) != canonical_json(expected_image):
                    errors.append(f"{label} image row mismatch: {scene['scene_id']}")

    def safe_coco_annotations(document: Any, label: str) -> list[dict[str, Any]]:
        if not isinstance(document, dict):
            return []
        raw = document.get("annotations")
        if not isinstance(raw, list):
            errors.append(f"{label} annotations is not a list")
            return []
        if any(not isinstance(annotation, dict) for annotation in raw):
            errors.append(f"{label} annotations contains a non-object row")
            return [annotation for annotation in raw if isinstance(annotation, dict)]
        return raw

    component_annotations = safe_coco_annotations(component_coco, "component COCO")
    defect_annotations = safe_coco_annotations(defect_coco, "defect COCO")
    if len(component_annotations) != 1920:
        errors.append(f"component COCO annotation count mismatch: {len(component_annotations)}")
    if len(defect_annotations) != 1680:
        errors.append(f"defect COCO annotation count mismatch: {len(defect_annotations)}")
    component_by_id: dict[int, dict[str, Any]] = {}
    for row_number, annotation in enumerate(component_annotations, 1):
        annotation_id = annotation.get("id")
        if isinstance(annotation_id, bool) or not isinstance(annotation_id, int):
            errors.append(f"component COCO annotation {row_number} has non-integer ID")
            continue
        component_by_id[annotation_id] = annotation
    defect_by_id: dict[int, dict[str, Any]] = {}
    for row_number, annotation in enumerate(defect_annotations, 1):
        annotation_id = annotation.get("id")
        if isinstance(annotation_id, bool) or not isinstance(annotation_id, int):
            errors.append(f"defect COCO annotation {row_number} has non-integer ID")
            continue
        defect_by_id[annotation_id] = annotation
    if len(component_by_id) != len(component_annotations):
        errors.append("component COCO annotation IDs are not unique")
    if len(defect_by_id) != len(defect_annotations):
        errors.append("defect COCO annotation IDs are not unique")
    cached_coco_scene: str | None = None
    cached_component_mask: np.ndarray | None = None
    cached_defect_mask: np.ndarray | None = None
    for instance in instances:
        prefix = f"COCO {instance['scene_id']}/instance-{instance['instance_index']}"
        if instance["scene_id"] != cached_coco_scene:
            cached_coco_scene = instance["scene_id"]
            try:
                component_path = repository_path(
                    scene_by_id[cached_coco_scene]["component_instance_mask_path"],
                    "COCO component mask",
                )
                defect_path = repository_path(
                    scene_by_id[cached_coco_scene]["defect_semantic_mask_path"],
                    "COCO defect mask",
                )
                with Image.open(component_path) as image:
                    if image.format != "PNG" or image.mode != "I;16" or image.size != (1280, 720):
                        raise ValueError("component PNG schema mismatch")
                    cached_component_mask = np.asarray(image, dtype=np.uint16).copy()
                with Image.open(defect_path) as image:
                    if image.format != "PNG" or image.mode != "L" or image.size != (1280, 720):
                        raise ValueError("defect PNG schema mismatch")
                    cached_defect_mask = np.asarray(image, dtype=np.uint8).copy()
            except Exception as error:
                errors.append(f"{prefix}: cannot load canonical masks for COCO: {error}")
                cached_component_mask = None
                cached_defect_mask = None
        component_binary = (
            np.zeros((720, 1280), dtype=bool)
            if cached_component_mask is None
            else cached_component_mask == int(instance["instance_index"])
        )
        expected_component_rle = encode_coco_uncompressed_rle(component_binary)
        component = component_by_id.get(instance["component_annotation_id"])
        if component is None:
            errors.append(f"{prefix}: missing component annotation")
        else:
            expected_attributes = {
                "scene_instance_id": instance["instance_index"],
                "nominal_bbox": instance["nominal_bbox_xywh"],
                "nominal_area": instance["nominal_area"],
                "pre_erosion_nominal_bbox": instance[
                    "pre_erosion_nominal_bbox_xywh"
                ],
                "pre_erosion_nominal_area": instance[
                    "pre_erosion_nominal_area"
                ],
                "pre_erosion_visible_bbox": instance[
                    "pre_erosion_visible_bbox_xywh"
                ],
                "pre_erosion_visible_area": instance[
                    "pre_erosion_visible_area"
                ],
                "visible_bbox": instance["visible_bbox_xywh"],
                "visible_area": instance["visible_area"],
                "visible_instance_mask": Path(
                    repository_path(
                        scene_by_id[instance["scene_id"]][
                            "component_instance_mask_path"
                        ],
                        f"{prefix}.component_instance_mask_path",
                    )
                ).relative_to(release_root).as_posix(),
                "source_parent_sample_id": instance["source_parent_sample_id"],
                "normal_proxy_from_paired_clean": instance[
                    "normal_proxy_from_paired_clean"
                ],
            }
            expected_component = {
                "id": instance["component_annotation_id"],
                "image_id": instance["image_id"],
                "category_id": instance["component_coco_category_id"],
                "bbox": instance["visible_bbox_xywh"],
                "area": instance["visible_area"],
                "iscrowd": 0,
                "segmentation": expected_component_rle,
                "attributes": expected_attributes,
            }
            try:
                decoded = decode_coco_uncompressed_rle(
                    component.get("segmentation"), f"{prefix}.component.segmentation"
                )
                if not np.array_equal(decoded, component_binary):
                    errors.append(f"{prefix}: component COCO RLE differs from PNG mask")
            except ValueError as error:
                errors.append(f"{prefix}: {error}")
            if canonical_json(component) != canonical_json(expected_component):
                errors.append(f"{prefix}: component COCO mismatch")
        if instance["defect_annotation_id"] is None:
            continue
        defect = defect_by_id.get(instance["defect_annotation_id"])
        defect_binary = (
            np.zeros((720, 1280), dtype=bool)
            if cached_defect_mask is None
            else cached_defect_mask == int(instance["defect_semantic_id"])
        )
        expected_defect_rle = encode_coco_uncompressed_rle(defect_binary)
        expected_defect = {
            "id": instance["defect_annotation_id"],
            "image_id": instance["image_id"],
            "category_id": instance["defect_semantic_id"],
            "bbox": instance["defect_bbox_xywh"],
            "area": instance["defect_area"],
            "iscrowd": 0,
            "segmentation": expected_defect_rle,
            "attributes": {
                "component_annotation_id": instance["component_annotation_id"],
                "scene_instance_id": instance["instance_index"],
                "semantic_mask": Path(
                    repository_path(
                        scene_by_id[instance["scene_id"]]["defect_semantic_mask_path"],
                        f"{prefix}.defect_semantic_mask_path",
                    )
                ).relative_to(release_root).as_posix(),
            },
        }
        if defect is None:
            errors.append(f"{prefix}: missing defect annotation")
            continue
        try:
            decoded = decode_coco_uncompressed_rle(
                defect.get("segmentation"), f"{prefix}.defect.segmentation"
            )
            if not np.array_equal(decoded, defect_binary):
                errors.append(f"{prefix}: defect COCO RLE differs from PNG mask")
        except ValueError as error:
            errors.append(f"{prefix}: {error}")
        if canonical_json(defect) != canonical_json(expected_defect):
            errors.append(f"{prefix}: defect COCO mismatch")

    # Summary is small enough to reconstruct exactly.
    expected_summary: list[dict[str, str]] = []
    for class_name in config["classes"]:
        expected_summary.append(
            {
                "dimension": "component_status_class",
                "value": class_name,
                "count": str(class_counts[class_name]),
            }
        )
    for profile in config["lighting_profiles"]:
        expected_summary.append(
            {
                "dimension": "lighting_profile_scene",
                "value": profile,
                "count": str(profile_counts[profile]),
            }
        )
        pair_values = list(pair_counts_by_profile[profile].values())
        expected_summary.append(
            {
                "dimension": "class_pair_range",
                "value": profile,
                "count": f"{min(pair_values)}..{max(pair_values)}",
            }
        )
    if summary_rows != expected_summary:
        errors.append("summary.csv does not match reconstructed release summary")

    # Replay all scenes.  Earlier attempts must fail and the recorded attempt
    # must be the first passing attempt, matching generator selection semantics.
    print("starting deterministic full replay of 384 scenes", flush=True)
    for replay_index, plan in enumerate(context.plans, 1):
        scene_id = plan["scene_id"]
        scene = scene_by_id.get(scene_id)
        if scene is None:
            continue
        try:
            recorded_attempt = int(scene["attempt"])
            replay: dict[str, Any] | None = None
            for attempt in range(recorded_attempt + 1):
                candidate = generator.render_scene(context, plan, attempt)
                if attempt < recorded_attempt and not candidate.get("failures"):
                    errors.append(
                        f"replay {scene_id}: an earlier attempt {attempt} passes unexpectedly"
                    )
                if attempt == recorded_attempt:
                    replay = candidate
            if replay is None:
                errors.append(f"replay {scene_id}: no replay result")
                continue
            if replay.get("failures"):
                errors.append(
                    f"replay {scene_id}: recorded attempt now fails: {replay['failures']}"
                )
                continue
            image_path = repository_path(scene["image_path"], "replay image_path")
            component_mask_path = repository_path(
                scene["component_instance_mask_path"], "replay component mask"
            )
            defect_mask_path = repository_path(
                scene["defect_semantic_mask_path"], "replay defect mask"
            )
            if replay["image_payload"] != image_path.read_bytes():
                errors.append(f"replay {scene_id}: JPEG bytes differ")
            with Image.open(component_mask_path) as image:
                published_component_mask = np.asarray(image)
            with Image.open(defect_mask_path) as image:
                published_defect_mask = np.asarray(image.convert("L"), dtype=np.uint8)
            if not np.array_equal(
                np.asarray(replay["component_instance_mask"]), published_component_mask
            ):
                errors.append(f"replay {scene_id}: component instance mask differs")
            if not np.array_equal(
                np.asarray(replay["defect_semantic_mask"]), published_defect_mask
            ):
                errors.append(f"replay {scene_id}: defect semantic mask differs")
            component_png = io.BytesIO()
            Image.fromarray(
                np.asarray(replay["component_instance_mask"], dtype=np.uint16)
            ).save(component_png, format="PNG", optimize=True)
            if component_png.getvalue() != component_mask_path.read_bytes():
                errors.append(f"replay {scene_id}: component PNG bytes differ")
            defect_png = io.BytesIO()
            Image.fromarray(
                np.asarray(replay["defect_semantic_mask"], dtype=np.uint8), mode="L"
            ).save(defect_png, format="PNG", optimize=True)
            if defect_png.getvalue() != defect_mask_path.read_bytes():
                errors.append(f"replay {scene_id}: defect PNG bytes differ")
            for field in (
                "background_mean_luma",
                "background_std_luma",
                "background_p99_luma",
                "component_mean_luma",
                "component_background_luma_delta",
                "component_background_luma_ratio",
                "component_light_spill_p99",
                "component_light_spill_max",
            ):
                same_float(scene[field], replay[field], field, f"replay {scene_id}", errors)
            if replay.get("background_luma_reference") != scene.get(
                "background_luma_reference"
            ):
                errors.append(f"replay {scene_id}: background luma reference differs")
            expected_json = {
                "background_params_json": replay["background_params"],
                "sensor_params_json": replay["sensor_params"],
                "shadow_params_json": {
                    "blur_radius_px": replay["shadow_blur_radius_px"],
                    "instances": [
                        {
                            "instance_index": int(item["instance_index"]),
                            **item["shadow_params"],
                        }
                        for item in replay["instances"]
                    ],
                },
            }
            for field, expected in expected_json.items():
                try:
                    recorded = json_field(scene[field], field)
                except (ValueError, json.JSONDecodeError) as error:
                    errors.append(f"replay {scene_id}: invalid {field}: {error}")
                    continue
                if canonical_json(recorded) != canonical_json(expected):
                    errors.append(f"replay {scene_id}: {field} differs")
            published = sorted(
                instances_by_scene[scene_id], key=lambda item: item["instance_index"]
            )
            replayed = sorted(
                replay["instances"], key=lambda item: item["instance_index"]
            )
            if len(published) != len(replayed):
                errors.append(f"replay {scene_id}: instance count differs")
            else:
                for published_item, replayed_item in zip(published, replayed, strict=True):
                    compare_replay_instance(published_item, replayed_item, config, errors)
        except Exception as error:
            errors.append(f"replay {scene_id}: exception: {error}")
        if replay_index % 24 == 0:
            print(f"replayed scenes={replay_index}/384", flush=True)

    try:
        validate_release_metadata(
            metadata,
            config,
            config_path,
            release_root,
            manifest_path,
            instances_path,
            component_coco_path,
            defect_coco_path,
            summary_path,
            class_counts,
            profile_counts,
            class_profile_counts,
            defect_parent_usage,
            normal_parent_usage,
            image_hashes,
            errors,
        )
    except Exception as error:
        errors.append(f"release metadata validation exception: {error}")

    if errors:
        return print_failure(errors)
    total_payload = sum(path.stat().st_size for path in release_root.rglob("*") if path.is_file())
    print(
        "PASS: scenes=384, components=1920, classes=8x240, profiles=4x96, "
        "class_profile=60, parents=168x10, val_test_parents=0, normal_parent_max=2, "
        "bbox_mask_coco_yolo=PASS, deterministic_full_replay=PASS, "
        "train_only=YES, evaluation_eligible=NO, "
        f"payload_mib={total_payload / 1024 / 1024:.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
