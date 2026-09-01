"""Generate a train-only paired multi-light illuminance-proxy auxiliary release.

The renderer changes only photometric appearance.  Component, defect, COCO and
YOLO geometry is inherited byte-for-byte from synthetic-v4-conveyor.  Numeric
capture_plan_target_lux values are experiment targets for a future real capture
rig; they are never represented as measured synthetic illuminance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import generate_synthetic_v4_conveyor as v4


DEFAULT_CONFIG = ROOT / "configs" / "synthetic_v5_illumination.json"
DEFAULT_RELEASE = ROOT / "synthetic" / "v5_illumination"
MARKER = ".synthetic_v5_illumination_marker"
MARKER_TEXT = "synthetic-v5-illumination\n"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def repository_path(value: str) -> Path:
    candidate = (ROOT / value).resolve()
    try:
        candidate.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"path escapes repository: {value}") from error
    return candidate


def repository_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def release_relative(path: Path, release_root: Path) -> str:
    return path.resolve().relative_to(release_root.resolve()).as_posix()


def verify_file(value: str, expected_sha256: str) -> Path:
    path = repository_path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {value}: {actual}")
    return path


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def current_runtime_versions() -> dict[str, str]:
    return {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "numpy": np.__version__,
        "pillow": Image.__version__,
        "libjpeg": str(features.version("jpg")),
        "zlib": str(features.version("zlib")),
    }


def safe_prepare_release(path: Path, force: bool) -> None:
    path = path.resolve()
    try:
        path.relative_to((ROOT / "synthetic").resolve())
    except ValueError as error:
        raise ValueError("release must be inside repository synthetic directory") from error
    marker = path / MARKER
    if path.exists():
        if not force:
            raise FileExistsError(f"release exists; pass --force: {path}")
        if not marker.is_file() or marker.read_text(encoding="ascii") != MARKER_TEXT:
            raise RuntimeError(f"refusing to replace unmarked release: {path}")
        shutil.rmtree(path)
    for relative in (
        "images/train",
        "masks/component_visible_instances/train",
        "masks/defect_semantic/train",
        "masks/shadow_attenuation/train",
        "labels/yolo_component_status/train",
        "labels/yolo_defects/train",
        "annotations/coco",
        "annotations/contact_sheets",
    ):
        (path / relative).mkdir(parents=True, exist_ok=True)
    marker.write_text(MARKER_TEXT, encoding="ascii", newline="\n")


def require_runtime(config: dict[str, Any]) -> None:
    contract = config["runtime_contract"]
    expected = {
        key: str(contract[key])
        for key in ("python", "numpy", "pillow", "libjpeg", "zlib")
    }
    actual = current_runtime_versions()
    if actual != expected:
        raise RuntimeError(f"runtime contract mismatch: actual={actual} expected={expected}")
    verify_file(contract["requirements_path"], contract["requirements_sha256"])


def require_source(config: dict[str, Any]) -> dict[str, Path]:
    source = config["source"]
    pins = {
        "config": (source["config_path"], source["config_sha256"]),
        "generator": (source["generator_path"], source["generator_sha256"]),
        "manifest": (source["manifest_path"], source["manifest_sha256"]),
        "instances": (source["instances_path"], source["instances_sha256"]),
        "component_coco": (
            source["component_coco_path"],
            source["component_coco_sha256"],
        ),
        "defect_coco": (source["defect_coco_path"], source["defect_coco_sha256"]),
        "release": (
            source["release_metadata_path"],
            source["release_metadata_sha256"],
        ),
    }
    return {name: verify_file(path, digest) for name, (path, digest) in pins.items()}


def label_counter(row: dict[str, str]) -> Counter[str]:
    labels = row["component_status_labels"].split("|")
    if len(labels) != 5 or len(set(labels)) != 5:
        raise ValueError(f"unexpected source labels for {row['scene_id']}: {labels}")
    return Counter(labels)


def build_condition_assignment(
    config: dict[str, Any], source_rows: list[dict[str, str]]
) -> dict[tuple[str, int], dict[str, Any]]:
    """Assign two variants per source scene across a controlled 48-cell design.

    Every source scene is used twice.  Its paired conditions must differ in rig,
    target-illuminance bin and shadow regime so the release contains an actual
    controlled within-composition lighting comparison.
    """

    rigs = list(config["multi_light_rigs"])
    lux_bins = list(config["target_illuminance_bins"])
    shadows = list(config["shadow_regimes"])
    cells = [
        {"cell_index": index, "rig": rig, "lux": lux, "shadow": shadow}
        for index, (rig, lux, shadow) in enumerate(
            (r, l, s) for r in rigs for l in lux_bins for s in shadows
        )
    ]
    if len(cells) != 48:
        raise ValueError("v5 core requires exactly 48 condition cells")
    source_profiles = list(config["source"]["expected_source_lighting_profiles"])
    rows_by_profile: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        rows_by_profile[row["lighting_profile"]].append(row)
    if set(rows_by_profile) != set(source_profiles):
        raise ValueError("unexpected source lighting profile inventory")

    variants_per_source = int(config["variants_per_source_scene"])
    if variants_per_source != 2:
        raise ValueError("v5 paired core requires exactly two variants per source scene")
    assignments: dict[tuple[str, int], int] = {}
    scene_lookup = {row["scene_id"]: row for row in source_rows}
    per_profile_cell: dict[tuple[str, int], list[tuple[str, int]]] = {}

    def paired_cell(cell_index: int) -> int:
        shadow_count = len(shadows)
        lux_count = len(lux_bins)
        rig_index = cell_index // (lux_count * shadow_count)
        remainder = cell_index % (lux_count * shadow_count)
        lux_index = remainder // shadow_count
        shadow_index = remainder % shadow_count
        return (
            ((rig_index + 1) % len(rigs)) * lux_count * shadow_count
            + ((lux_index + 1) % lux_count) * shadow_count
            + ((shadow_index + 1) % shadow_count)
        )

    for profile in source_profiles:
        rows = sorted(rows_by_profile[profile], key=lambda row: row["scene_id"])
        if len(rows) != 96:
            raise ValueError(f"expected 96 source scenes for {profile}")
        rng = random.Random(stable_seed(config["global_seed"], profile, "initial"))
        rng.shuffle(rows)
        for cell in cells:
            per_profile_cell[(profile, int(cell["cell_index"]))] = []
        for cell in cells:
            start = int(cell["cell_index"]) * 2
            for row in (rows[start], rows[start + 1]):
                token = (row["scene_id"], 0)
                per_profile_cell[(profile, int(cell["cell_index"]))].append(token)
                assignments[token] = int(cell["cell_index"])
                second_cell = paired_cell(int(cell["cell_index"]))
                paired_token = (row["scene_id"], 1)
                per_profile_cell[(profile, second_cell)].append(paired_token)
                assignments[paired_token] = second_cell
        if any(len(per_profile_cell[(profile, index)]) != 4 for index in range(len(cells))):
            raise RuntimeError(f"initial paired inventory failed for {profile}")

    class_names = list(config["classes"])
    contract = config["balance_contract"]
    target = int(contract["instance_per_class_full_condition_cell_target"])
    allowed_low, allowed_high = [
        int(value) for value in contract["instance_per_class_full_condition_cell_range"]
    ]
    counts: dict[int, Counter[str]] = {index: Counter() for index in range(len(cells))}
    for (scene_id, _variant_index), cell_index in assignments.items():
        counts[cell_index].update(label_counter(scene_lookup[scene_id]))

    def pair_is_valid(scene_id: str, variant_index: int, candidate_cell: int) -> bool:
        paired_index = 1 - variant_index
        paired = assignments[(scene_id, paired_index)]
        left = cells[candidate_cell]
        right = cells[paired]
        return (
            left["rig"]["id"] != right["rig"]["id"]
            and left["lux"]["id"] != right["lux"]["id"]
            and left["shadow"]["id"] != right["shadow"]["id"]
        )

    def objective_for(cell_index: int) -> int:
        return sum((counts[cell_index][name] - target) ** 2 for name in class_names)

    objective = sum(objective_for(index) for index in counts)
    rng = random.Random(stable_seed(config["global_seed"], "condition-optimizer"))
    for iteration in range(2_000_000):
        if objective <= 8:
            break
        profile = rng.choice(source_profiles)
        first_cell, second_cell = rng.sample(range(len(cells)), 2)
        first_position = rng.randrange(4)
        second_position = rng.randrange(4)
        first_token = per_profile_cell[(profile, first_cell)][first_position]
        second_token = per_profile_cell[(profile, second_cell)][second_position]
        if first_token[0] == second_token[0]:
            continue
        if not pair_is_valid(first_token[0], first_token[1], second_cell):
            continue
        if not pair_is_valid(second_token[0], second_token[1], first_cell):
            continue
        first_scene = first_token[0]
        second_scene = second_token[0]
        old_local = objective_for(first_cell) + objective_for(second_cell)
        first_labels = label_counter(scene_lookup[first_scene])
        second_labels = label_counter(scene_lookup[second_scene])
        counts[first_cell].subtract(first_labels)
        counts[first_cell].update(second_labels)
        counts[second_cell].subtract(second_labels)
        counts[second_cell].update(first_labels)
        new_local = objective_for(first_cell) + objective_for(second_cell)
        temperature = max(0.005, 3.0 * (1.0 - iteration / 2_000_000.0))
        accept = new_local <= old_local or rng.random() < math.exp(
            (old_local - new_local) / temperature
        )
        if accept:
            per_profile_cell[(profile, first_cell)][first_position] = second_token
            per_profile_cell[(profile, second_cell)][second_position] = first_token
            assignments[first_token] = second_cell
            assignments[second_token] = first_cell
            objective += new_local - old_local
        else:
            counts[first_cell].subtract(second_labels)
            counts[first_cell].update(first_labels)
            counts[second_cell].subtract(first_labels)
            counts[second_cell].update(second_labels)
    distribution = Counter(counts[index][name] for index in counts for name in class_names)
    if objective > 8 or any(
        not allowed_low <= counts[index][name] <= allowed_high
        for index in counts
        for name in class_names
    ):
        raise RuntimeError(
            f"condition assignment did not converge: objective={objective} "
            f"distribution={distribution}"
        )

    result: dict[tuple[str, int], dict[str, Any]] = {}
    for token, cell_index in assignments.items():
        source_scene_id, variant_index = token
        cell = cells[cell_index]
        result[token] = {
            "variant_index": variant_index,
            "condition_cell_index": cell_index,
            "condition_cell_id": (
                f"{cell['rig']['id']}__{cell['lux']['id']}__{cell['shadow']['id']}"
            ),
            "rig": deepcopy(cell["rig"]),
            "lux": deepcopy(cell["lux"]),
            "shadow": deepcopy(cell["shadow"]),
        }
    for source_scene_id in scene_lookup:
        first = result[(source_scene_id, 0)]
        second = result[(source_scene_id, 1)]
        if (
            first["rig"]["id"] == second["rig"]["id"]
            or first["lux"]["id"] == second["lux"]["id"]
            or first["shadow"]["id"] == second["shadow"]["id"]
        ):
            raise RuntimeError(f"paired condition separation failed: {source_scene_id}")
    return result


def srgb_to_linear(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32) / 255.0
    return np.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(array: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)
    encoded = np.where(
        value <= 0.0031308,
        value * 12.92,
        1.055 * np.power(value, 1.0 / 2.4) - 0.055,
    )
    return np.clip(np.round(encoded * 255.0), 0, 255).astype(np.uint8)


def cct_proxy_gain(kelvin: float) -> np.ndarray:
    """Return an explicitly non-calibrated RGB appearance proxy."""

    anchors = [
        (3000.0, np.array([1.16, 1.02, 0.78], dtype=np.float32)),
        (4000.0, np.array([1.09, 1.02, 0.90], dtype=np.float32)),
        (5000.0, np.array([1.03, 1.01, 0.97], dtype=np.float32)),
        (6500.0, np.array([1.00, 1.00, 1.00], dtype=np.float32)),
        (7500.0, np.array([0.93, 1.00, 1.10], dtype=np.float32)),
    ]
    value = float(kelvin)
    if value <= anchors[0][0]:
        return anchors[0][1].copy()
    if value >= anchors[-1][0]:
        return anchors[-1][1].copy()
    for (left_k, left), (right_k, right) in zip(anchors, anchors[1:]):
        if left_k <= value <= right_k:
            fraction = (value - left_k) / (right_k - left_k)
            return left * (1.0 - fraction) + right * fraction
    raise AssertionError("unreachable CCT proxy interpolation")


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("empty component mask")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def dilate_bool(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    size = radius * 2 + 1
    return np.asarray(image.filter(ImageFilter.MaxFilter(size)), dtype=np.uint8) > 0


def shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    height, width = mask.shape
    result = np.zeros_like(mask, dtype=np.float32)
    source_x0 = max(0, -dx)
    source_y0 = max(0, -dy)
    source_x1 = min(width, width - dx) if dx >= 0 else width
    source_y1 = min(height, height - dy) if dy >= 0 else height
    target_x0 = source_x0 + dx
    target_y0 = source_y0 + dy
    target_x1 = source_x1 + dx
    target_y1 = source_y1 + dy
    if source_x0 < source_x1 and source_y0 < source_y1:
        result[target_y0:target_y1, target_x0:target_x1] = mask[
            source_y0:source_y1, source_x0:source_x1
        ]
    return result


def source_effective_weights(
    sources: list[dict[str, Any]], center_x_fraction: float, center_y_fraction: float
) -> np.ndarray:
    weights: list[float] = []
    for source in sources:
        anchor_x, anchor_y = source["anchor_xy_fraction"]
        distance_squared = (
            (float(anchor_x) - center_x_fraction) ** 2
            + (float(anchor_y) - center_y_fraction) ** 2
        )
        weights.append(float(source["relative_intensity"]) / (0.35 + distance_squared))
    array = np.asarray(weights, dtype=np.float32)
    return array / max(float(array.sum()), 1e-8)


def apply_component_lighting(
    linear: np.ndarray,
    application_mask: np.ndarray,
    instance: dict[str, Any],
    condition: dict[str, Any],
    config: dict[str, Any],
    seed: int,
    field_reference_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    rendering = config["rendering"]
    rng = random.Random(seed)
    reference_mask = application_mask if field_reference_mask is None else field_reference_mask
    if not np.any(application_mask) or not np.any(reference_mask):
        raise ValueError("component lighting masks must be non-empty")
    x0, y0, x1, y1 = mask_bbox(reference_mask)
    full_height, full_width = application_mask.shape
    crop_height, crop_width = y1 - y0, x1 - x0
    yy, xx = np.mgrid[0:crop_height, 0:crop_width]
    center_x_local = (crop_width - 1) * 0.5
    center_y_local = (crop_height - 1) * 0.5
    center_x = x0 + center_x_local
    center_y = y0 + center_y_local
    half_width = max(1.0, (x1 - x0) * 0.5)
    half_height = max(1.0, (y1 - y0) * 0.5)
    local_x = (xx - center_x_local) / half_width
    local_y = (yy - center_y_local) / half_height
    application_crop = application_mask[y0:y1, x0:x1]
    reference_crop = reference_mask[y0:y1, x0:x1]
    sources = condition["rig"]["sources"]
    weights = source_effective_weights(
        sources,
        center_x / max(1, full_width - 1),
        center_y / max(1, full_height - 1),
    )
    direction_strength = rng.uniform(*rendering["directional_gain_strength"])
    hotspot_strength = rng.uniform(*rendering["hotspot_strength"])
    hotspot_sigma = rng.uniform(*rendering["hotspot_sigma_fraction"])
    field = np.zeros((crop_height, crop_width), dtype=np.float32)
    color_gain = np.zeros(3, dtype=np.float32)
    source_rows: list[dict[str, Any]] = []
    component_rotation = float(instance["rotation_degrees"])
    for source, weight in zip(sources, weights, strict=True):
        azimuth = float(source["image_plane_azimuth_deg"])
        radians = math.radians(azimuth)
        projection = np.clip(
            0.5 + 0.5 * (math.cos(radians) * local_x + math.sin(radians) * local_y) / math.sqrt(2.0),
            0.0,
            1.0,
        )
        hotspot_x = 0.55 * math.cos(radians)
        hotspot_y = 0.55 * math.sin(radians)
        hotspot = np.exp(
            -((local_x - hotspot_x) ** 2 + (local_y - hotspot_y) ** 2)
            / max(2.0 * hotspot_sigma**2, 1e-6)
        )
        source_field = 1.0 + direction_strength * (projection - 0.5)
        source_field += hotspot_strength * hotspot
        field += float(weight) * source_field.astype(np.float32)
        color_gain += float(weight) * cct_proxy_gain(float(source["cct_proxy_kelvin"]))
        source_rows.append(
            {
                "light_id": source["light_id"],
                "effective_weight": round(float(weight), 8),
                "image_plane_azimuth_deg": azimuth,
                "local_azimuth_proxy_deg": round((azimuth - component_rotation) % 360.0, 8),
                "elevation_proxy_deg": float(source["elevation_proxy_deg"]),
                "cct_proxy_kelvin": int(source["cct_proxy_kelvin"]),
            }
        )
    mean_field = float(field[reference_crop].mean())
    field /= max(mean_field, 1e-6)
    color_luma = float(np.dot(color_gain, np.array([0.2126, 0.7152, 0.0722])))
    color_gain /= max(color_luma, 1e-6)

    occluder_applied = rng.random() < float(
        rendering["component_occluder_shadow_probability"]
    )
    occluder_angle = rng.uniform(0.0, 360.0)
    occluder_strength = 0.0
    occluder_width = 0.0
    occluder_center = 0.0
    occluder_map = np.ones((crop_height, crop_width), dtype=np.float32)
    if occluder_applied:
        occluder_strength = rng.uniform(*rendering["component_occluder_shadow_strength"])
        occluder_strength *= float(condition["shadow"]["component_occluder_strength_scale"])
        occluder_width = rng.uniform(*rendering["component_occluder_shadow_width_fraction"])
        occluder_center = rng.uniform(-0.35, 0.35)
        angle = math.radians(occluder_angle)
        coordinate = math.cos(angle) * local_x + math.sin(angle) * local_y
        distance = np.abs(coordinate - occluder_center)
        feather = max(0.04, occluder_width * 0.30)
        band = np.clip((occluder_width * 0.5 + feather - distance) / feather, 0.0, 1.0)
        occluder_map = 1.0 - float(occluder_strength) * band.astype(np.float32)

    base_gain = float(condition["lux"]["relative_light_power"])
    total_gain = base_gain * field * occluder_map
    linear_crop = linear[y0:y1, x0:x1]
    source_pixels = linear_crop[application_crop]
    transformed = source_pixels * total_gain[application_crop, None] * color_gain[None, :]
    linear_crop[application_crop] = np.clip(transformed, 0.0, 1.0)
    return {
        "instance_index": int(instance["instance_index"]),
        "source_rows": source_rows,
        "base_relative_light_power": base_gain,
        "directional_gain_strength": round(direction_strength, 8),
        "hotspot_strength": round(hotspot_strength, 8),
        "hotspot_sigma_fraction": round(hotspot_sigma, 8),
        "occluder_shadow_applied": occluder_applied,
        "occluder_shadow_angle_deg": round(occluder_angle, 8),
        "occluder_shadow_strength": round(float(occluder_strength), 8),
        "occluder_shadow_width_fraction": round(float(occluder_width), 8),
        "occluder_shadow_center": round(float(occluder_center), 8),
        "gain_min": round(float(total_gain[reference_crop].min()), 8),
        "gain_mean": round(float(total_gain[reference_crop].mean()), 8),
        "gain_max": round(float(total_gain[reference_crop].max()), 8),
        "color_gain_rgb": [round(float(value), 8) for value in color_gain],
    }


def create_cast_shadow(
    component_instance_mask: np.ndarray,
    condition: dict[str, Any],
    config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rendering = config["rendering"]
    regime = condition["shadow"]
    rng = random.Random(seed)
    union = component_instance_mask > 0
    attenuation = np.zeros(union.shape, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    instance_ids = sorted(int(value) for value in np.unique(component_instance_mask) if value)
    for instance_id in instance_ids:
        mask = (component_instance_mask == instance_id).astype(np.float32)
        contact_blur = rng.uniform(*rendering["contact_shadow_blur_radius_px"])
        contact_blur *= float(regime["contact_blur_scale"])
        contact_opacity = rng.uniform(*rendering["contact_shadow_opacity"])
        contact_opacity *= float(regime["contact_opacity_scale"])
        contact_offset_scale = float(regime["contact_offset_scale"])
        contact_dx = int(round(rng.uniform(*rendering["contact_shadow_offset_x_px"]) * contact_offset_scale))
        contact_dy = int(round(rng.uniform(*rendering["contact_shadow_offset_y_px"]) * contact_offset_scale))
        contact_shifted = shift_mask(mask, contact_dx, contact_dy)
        contact_image = Image.fromarray(
            np.clip(np.round(contact_shifted * 255.0), 0, 255).astype(np.uint8), mode="L"
        ).filter(ImageFilter.GaussianBlur(contact_blur))
        contact_layer = np.asarray(contact_image, dtype=np.float32) / 255.0
        contact_layer *= float(contact_opacity)
        attenuation = 1.0 - (1.0 - attenuation) * (1.0 - contact_layer)
        rows.append(
            {
                "instance_index": instance_id,
                "light_id": "CONTACT",
                "shadow_kind": "contact",
                "shadow_direction_deg": None,
                "offset_x_px": contact_dx,
                "offset_y_px": contact_dy,
                "length_px": 0.0,
                "blur_radius_px": round(float(contact_blur), 8),
                "opacity": round(float(contact_opacity), 8),
            }
        )
        for source in condition["rig"]["sources"]:
            azimuth = float(source["image_plane_azimuth_deg"])
            elevation = float(source["elevation_proxy_deg"])
            radians = math.radians(azimuth)
            base_length = rng.uniform(*rendering["cast_shadow_length_px"])
            length = base_length * float(regime["directional_length_scale"])
            length *= 1.25 - 0.55 * min(max(elevation / 90.0, 0.0), 1.0)
            dx = int(round(-math.cos(radians) * length))
            dy = int(round(-math.sin(radians) * length))
            blur = rng.uniform(*rendering["cast_shadow_blur_radius_px"])
            blur *= float(regime["directional_blur_scale"])
            opacity = rng.uniform(*rendering["cast_shadow_opacity"])
            opacity *= float(regime["directional_opacity_scale"])
            opacity *= 0.70 + 0.30 * float(source["relative_intensity"])
            shifted = shift_mask(mask, dx, dy)
            shifted_image = Image.fromarray(
                np.clip(np.round(shifted * 255.0), 0, 255).astype(np.uint8), mode="L"
            ).filter(ImageFilter.GaussianBlur(blur))
            layer = np.asarray(shifted_image, dtype=np.float32) / 255.0
            layer *= float(opacity)
            attenuation = 1.0 - (1.0 - attenuation) * (1.0 - layer)
            rows.append(
                {
                    "instance_index": instance_id,
                    "light_id": source["light_id"],
                    "shadow_kind": "directional_cast",
                    "shadow_direction_deg": round((azimuth + 180.0) % 360.0, 8),
                    "offset_x_px": dx,
                    "offset_y_px": dy,
                    "length_px": round(float(length), 8),
                    "blur_radius_px": round(float(blur), 8),
                    "opacity": round(float(opacity), 8),
                }
            )
    attenuation[union] = 0.0
    maximum = float(config["qc"]["maximum_shadow_attenuation"])
    return np.clip(attenuation, 0.0, maximum), rows


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


def apply_sensor_pair(
    rendered: np.ndarray,
    reference: np.ndarray,
    condition: dict[str, Any],
    seed: int,
) -> tuple[bytes, bytes, dict[str, Any]]:
    rng = random.Random(seed)
    lux = condition["lux"]
    sigma = rng.uniform(*lux["sensor_noise_sigma"])
    blur = rng.uniform(*lux["sensor_blur_radius_px"])
    quality = rng.randint(int(lux["jpeg_quality"][0]), int(lux["jpeg_quality"][1]))
    noise_seed = rng.randrange(0, 2**32)
    noise_rng = np.random.default_rng(noise_seed)
    noise = noise_rng.normal(0.0, sigma, rendered.shape).astype(np.float32)

    def process(array: np.ndarray) -> Image.Image:
        image = Image.fromarray(array, mode="RGB")
        if blur > 0:
            image = image.filter(ImageFilter.GaussianBlur(blur))
        value = np.asarray(image, dtype=np.float32) + noise
        return Image.fromarray(np.clip(np.round(value), 0, 255).astype(np.uint8), mode="RGB")

    return (
        jpeg_payload(process(rendered), quality),
        jpeg_payload(process(reference), quality),
        {
            "noise_sigma": round(float(sigma), 8),
            "noise_seed": noise_seed,
            "blur_radius_px": round(float(blur), 8),
            "jpeg_quality": quality,
        },
    )


def luma(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array, dtype=np.float32)
    return 0.2126 * value[..., 0] + 0.7152 * value[..., 1] + 0.0722 * value[..., 2]


def load_neutral_v4_context(config: dict[str, Any]) -> Any:
    """Replay v4 geometry while disabling its added light/shadow/sensor layer."""

    source = config["source"]
    context = v4.load_context(
        repository_path(source["config_path"]),
        repository_path(source["release_root"]),
    )
    for ranges in context.config["lighting_profile_ranges"].values():
        ranges["exposure_ev"] = [0.0, 0.0]
        ranges["contrast"] = [1.0, 1.0]
        ranges["channel_gain_r"] = [1.0, 1.0]
        ranges["channel_gain_g"] = [1.0, 1.0]
        ranges["channel_gain_b"] = [1.0, 1.0]
        ranges["gradient_strength"] = [0.0, 0.0]
        ranges["hotspot_strength"] = [0.0, 0.0]
    context.config["component_shadow"] = {
        "opacity": [0.0, 0.0],
        "blur_radius_px": [0.0, 0.0],
        "offset_x_px": [0, 0],
        "offset_y_px": [0, 0],
    }
    context.config["scene_sensor"] = {
        "noise_sigma": [0.0, 0.0],
        "blur_radius_px": [0.0, 0.0],
        "jpeg_quality": [100, 100],
    }
    return context


def render_neutral_v4_scene(
    context: Any,
    plan: dict[str, Any],
    source_row: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return neutral defect/clean composites and canonical v4 masks."""

    attempt = int(source_row["attempt"])
    width = int(context.config["scene_width"])
    height = int(context.config["scene_height"])
    layout = context.config["layout"]
    background, _ = v4.make_background(context, plan, attempt)
    transformed = [
        v4.transform_instance(context, plan, item, attempt) for item in plan["instances"]
    ]
    defect_scene = background.copy()
    clean_scene = background.copy()
    replay_component = np.zeros((height, width), dtype=np.uint16)
    replay_defect = np.zeros((height, width), dtype=np.uint8)
    for item in transformed:
        cell = int(item["grid_cell"])
        column = cell % int(layout["columns"])
        row_index = cell // int(layout["columns"])
        center_x = int(round((column + 0.5) * width / int(layout["columns"])))
        center_y = int(round((row_index + 0.5) * height / int(layout["rows"])))
        center_x += int(item["jitter_x"])
        center_y += int(item["jitter_y"])
        object_width, object_height = item["nominal_rendered"].size
        left = int(round(center_x - object_width / 2))
        top = int(round(center_y - object_height / 2))
        right = left + object_width
        bottom = top + object_height
        if left < 0 or top < 0 or right > width or bottom > height:
            raise RuntimeError(
                f"neutral replay frame truncation: {source_row['scene_id']} "
                f"instance={item['instance_index']}"
            )
        visible_local = np.asarray(item["visible_rendered"], dtype=np.uint8) >= 128
        semantic_local = np.asarray(item["semantic"], dtype=np.uint8) > 0
        if item["class_name"] not in context.config["component_alpha"][
            "missing_material_classes"
        ]:
            semantic_local &= visible_local
        instance_index = int(item["instance_index"])
        component_region = replay_component[top:bottom, left:right]
        component_region[visible_local] = instance_index
        if item["class_name"] != "normal_proxy":
            semantic_id = int(context.config["defect_semantic_ids"][item["class_name"]])
            defect_region = replay_defect[top:bottom, left:right]
            defect_region[semantic_local] = semantic_id
        location = (left, top)
        defect_scene.paste(item["defect_image"], location, item["visible_paste"])
        clean_scene.paste(item["clean_image"], location, item["nominal_paste"])

    source_component_path = repository_path(source_row["component_instance_mask_path"])
    source_defect_path = repository_path(source_row["defect_semantic_mask_path"])
    source_component = np.asarray(Image.open(source_component_path), dtype=np.uint16)
    source_defect = np.asarray(Image.open(source_defect_path), dtype=np.uint8)
    if not np.array_equal(replay_component, source_component):
        raise RuntimeError(f"neutral replay component mask mismatch: {source_row['scene_id']}")
    if not np.array_equal(replay_defect, source_defect):
        raise RuntimeError(f"neutral replay defect mask mismatch: {source_row['scene_id']}")
    return (
        np.asarray(defect_scene, dtype=np.uint8),
        np.asarray(clean_scene, dtype=np.uint8),
        replay_component,
        replay_defect,
    )


def render_variant(
    config: dict[str, Any],
    source_row: dict[str, str],
    source_instances: list[dict[str, Any]],
    condition: dict[str, Any],
    neutral_defect_array: np.ndarray,
    neutral_clean_array: np.ndarray,
    component_mask: np.ndarray,
    defect_mask: np.ndarray,
    attempt: int,
) -> dict[str, Any]:
    scene_seed = stable_seed(
        config["global_seed"],
        config["release"],
        source_row["scene_id"],
        condition["condition_cell_id"],
        attempt,
    )
    if (
        component_mask.shape != neutral_defect_array.shape[:2]
        or neutral_clean_array.shape != neutral_defect_array.shape
        or defect_mask.shape != component_mask.shape
    ):
        raise ValueError("source image/mask size mismatch")
    expected_ids = list(range(1, int(config["instances_per_scene"]) + 1))
    if sorted(int(value) for value in np.unique(component_mask) if value) != expected_ids:
        raise ValueError("source component instance IDs are not canonical")

    nominal_masks: dict[int, np.ndarray] = {}
    nominal_union = np.zeros(component_mask.shape, dtype=bool)
    missing_material_classes = {"lead_breakage", "body_chip"}
    for instance in source_instances:
        instance_index = int(instance["instance_index"])
        nominal = component_mask == instance_index
        if instance["component_status_class"] in missing_material_classes:
            semantic_id = int(instance["defect_semantic_id"])
            nominal = np.logical_or(nominal, defect_mask == semantic_id)
        nominal_masks[instance_index] = nominal
        nominal_union |= nominal

    shadow, cast_shadow_rows = create_cast_shadow(
        component_mask,
        condition,
        config,
        stable_seed(scene_seed, "cast-shadow"),
    )
    shadow = shadow.copy()
    shadow[nominal_union] = 0.0
    visible_union = component_mask > 0
    defect_linear = srgb_to_linear(neutral_defect_array)
    clean_linear = srgb_to_linear(neutral_clean_array)
    defect_linear[~nominal_union] *= (1.0 - shadow[~nominal_union, None])
    clean_linear[~nominal_union] *= (1.0 - shadow[~nominal_union, None])
    instance_effects: list[dict[str, Any]] = []
    for instance in sorted(source_instances, key=lambda row: int(row["instance_index"])):
        instance_index = int(instance["instance_index"])
        lighting_seed = stable_seed(scene_seed, "component-light", instance_index)
        effect_defect = apply_component_lighting(
            defect_linear,
            component_mask == instance_index,
            instance,
            condition,
            config,
            lighting_seed,
            nominal_masks[instance_index],
        )
        effect_clean = apply_component_lighting(
            clean_linear,
            nominal_masks[instance_index],
            instance,
            condition,
            config,
            lighting_seed,
            nominal_masks[instance_index],
        )
        if canonical_json(effect_defect) != canonical_json(effect_clean):
            raise RuntimeError("paired defect/clean lighting parameters diverged")
        instance_effects.append(effect_defect)
    rendered_pre = linear_to_srgb(defect_linear)
    clean_pre = linear_to_srgb(clean_linear)

    spill_exclusion = dilate_bool(
        nominal_union, int(config["qc"]["spill_exclusion_dilation_px"])
    )
    pre_positive = np.maximum(
        rendered_pre.astype(np.float32) - neutral_defect_array.astype(np.float32), 0.0
    ).max(axis=2)
    pre_spill_max = float(pre_positive[~spill_exclusion].max())

    image_payload, reference_payload, sensor_params = apply_sensor_pair(
        rendered_pre,
        clean_pre,
        condition,
        stable_seed(scene_seed, "sensor"),
    )
    final_array = np.asarray(Image.open(io.BytesIO(image_payload)).convert("RGB"), dtype=np.uint8)
    reference_array = np.asarray(
        Image.open(io.BytesIO(reference_payload)).convert("RGB"), dtype=np.uint8
    )
    post_difference = np.abs(
        final_array.astype(np.float32) - reference_array.astype(np.float32)
    ).max(axis=2)
    spill_values = post_difference[~spill_exclusion]
    threshold = float(config["qc"]["post_jpeg_paired_clean_spill_threshold"])
    background_values = luma(final_array)[~nominal_union]
    component_values = luma(final_array)[visible_union]
    shadow_u8 = np.clip(np.round(shadow * 255.0), 0, 255).astype(np.uint8)
    defect_visibility: list[dict[str, Any]] = []
    final_image = Image.open(io.BytesIO(image_payload)).convert("RGB")
    reference_image = Image.open(io.BytesIO(reference_payload)).convert("RGB")
    for instance in sorted(source_instances, key=lambda row: int(row["instance_index"])):
        class_name = instance["component_status_class"]
        if class_name == "normal_proxy":
            continue
        semantic_id = int(instance["defect_semantic_id"])
        semantic = defect_mask == semantic_id
        visibility = v4.detector_visibility_metrics(
            final_image,
            reference_image,
            semantic,
            int(config["qc"]["defect_visibility_input_width"]),
            float(config["qc"]["defect_visibility_pixel_threshold"]),
        )
        visibility.update(
            {
                "instance_index": int(instance["instance_index"]),
                "class_name": class_name,
                "semantic_id": semantic_id,
            }
        )
        defect_visibility.append(visibility)
    metrics = {
        "pre_sensor_positive_spill_max": pre_spill_max,
        "post_jpeg_paired_clean_spill_p99": float(np.percentile(spill_values, 99.0)),
        "post_jpeg_paired_clean_spill_max": float(spill_values.max()),
        "post_jpeg_paired_clean_spill_fraction": float(np.mean(spill_values > threshold)),
        "post_jpeg_paired_clean_spill_energy": float(spill_values.sum()),
        "background_mean_luma": float(background_values.mean()),
        "background_p99_luma": float(np.percentile(background_values, 99.0)),
        "component_mean_luma": float(component_values.mean()),
        "component_dark_fraction": float(np.mean(component_values < 16.0)),
        "component_saturated_fraction": float(np.mean(component_values > 250.0)),
        "shadow_nonzero_fraction": float(np.mean(shadow_u8 > 0)),
        "shadow_max_attenuation": float(shadow.max()),
    }
    qc = config["qc"]
    failures: list[str] = []
    gates = [
        (metrics["pre_sensor_positive_spill_max"] <= float(qc["pre_sensor_maximum_positive_spill"]), "pre_spill"),
        (metrics["post_jpeg_paired_clean_spill_p99"] <= float(qc["post_jpeg_paired_clean_spill_p99"]), "post_spill_p99"),
        (metrics["post_jpeg_paired_clean_spill_max"] <= float(qc["post_jpeg_paired_clean_spill_max"]), "post_spill_max"),
        (metrics["post_jpeg_paired_clean_spill_fraction"] <= float(qc["post_jpeg_paired_clean_spill_fraction"]), "post_spill_fraction"),
        (metrics["background_mean_luma"] <= float(qc["maximum_background_mean_luma"]), "background_mean"),
        (metrics["background_p99_luma"] <= float(qc["maximum_background_p99_luma"]), "background_p99"),
        (metrics["component_mean_luma"] >= float(qc["minimum_component_mean_luma"]), "component_mean_low"),
        (metrics["component_mean_luma"] <= float(qc["maximum_component_mean_luma"]), "component_mean_high"),
        (metrics["component_dark_fraction"] <= float(qc["maximum_component_dark_fraction"]), "component_dark"),
        (metrics["component_saturated_fraction"] <= float(qc["maximum_component_saturated_fraction"]), "component_saturated"),
        (metrics["shadow_nonzero_fraction"] >= float(qc["minimum_shadow_nonzero_fraction"]), "shadow_empty"),
        (metrics["shadow_max_attenuation"] <= float(qc["maximum_shadow_attenuation"]) + 1e-8, "shadow_max"),
    ]
    failures.extend(name for passed, name in gates if not passed)
    for visibility in defect_visibility:
        class_name = str(visibility["class_name"])
        if float(visibility["mean_abs_delta"]) < float(
            qc["minimum_defect_mean_abs_delta"][class_name]
        ):
            failures.append(
                f"defect_mad:{visibility['instance_index']}:{visibility['mean_abs_delta']}"
            )
        if float(visibility["changed_fraction"]) < float(
            qc["minimum_defect_changed_fraction"][class_name]
        ):
            failures.append(
                "defect_changed_fraction:"
                f"{visibility['instance_index']}:{visibility['changed_fraction']}"
            )
    return {
        "failures": failures,
        "scene_seed": scene_seed,
        "attempt": attempt,
        "image_payload": image_payload,
        "reference_payload": reference_payload,
        "component_mask": component_mask,
        "defect_mask": defect_mask,
        "shadow_mask": shadow_u8,
        "sensor_params": sensor_params,
        "metrics": metrics,
        "instance_effects": instance_effects,
        "defect_visibility": defect_visibility,
        "cast_shadow_rows": cast_shadow_rows,
    }


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


def create_contact_sheet(
    config: dict[str, Any],
    scene_rows: list[dict[str, Any]],
    instances_by_scene: dict[str, list[dict[str, Any]]],
    release_root: Path,
    overlay: bool,
) -> Path:
    rigs = [item["id"] for item in config["multi_light_rigs"]]
    lux_ids = [item["id"] for item in config["target_illuminance_bins"]]
    shadow_ids = [item["id"] for item in config["shadow_regimes"]]
    tile_width, tile_height, label_height = 160, 90, 22
    sheet = Image.new(
        "RGB",
        (len(rigs) * len(shadow_ids) * tile_width, len(lux_ids) * (tile_height + label_height)),
        (15, 15, 15),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    by_cell = {row["condition_cell_id"]: row for row in scene_rows}
    for row_index, lux_id in enumerate(lux_ids):
        for rig_index, rig_id in enumerate(rigs):
            for shadow_index, shadow_id in enumerate(shadow_ids):
                cell_id = f"{rig_id}__{lux_id}__{shadow_id}"
                row = by_cell[cell_id]
                image = Image.open(repository_path(row["image_path"])).convert("RGB")
                if overlay:
                    image = draw_overlay(image, instances_by_scene[row["scene_id"]])
                image = image.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
                left = (rig_index * len(shadow_ids) + shadow_index) * tile_width
                top = row_index * (tile_height + label_height)
                sheet.paste(image, (left, top))
                label = f"{lux_id} {rig_id[:9]} {shadow_id[:4]}"
                draw.text((left + 3, top + tile_height + 4), label, fill=(235, 235, 235), font=font)
    filename = "contact_sheet_overlay.jpg" if overlay else "contact_sheet.jpg"
    path = release_root / filename
    path.write_bytes(jpeg_payload(sheet, 92))
    return path


def create_paired_comparison_sheet(
    scene_rows: list[dict[str, Any]], release_root: Path
) -> Path:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scene_rows:
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
    rows = int(math.ceil(sample_count / columns))
    sheet = Image.new(
        "RGB",
        (columns * pair_width, rows * (tile_height + label_height)),
        (15, 15, 15),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for pair_index, source_scene_id in enumerate(selected):
        variants = sorted(
            by_source[source_scene_id], key=lambda row: int(row["source_variant_index"])
        )
        if len(variants) != 2:
            raise RuntimeError(f"paired comparison inventory mismatch: {source_scene_id}")
        left = (pair_index % columns) * pair_width
        top = (pair_index // columns) * (tile_height + label_height)
        for variant_index, row in enumerate(variants):
            image = Image.open(repository_path(row["image_path"])).convert("RGB")
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
        draw.text((left + 3, top + tile_height + 2), first_label, fill=(235, 235, 235), font=font)
        draw.text(
            (left + tile_width + 3, top + tile_height + 2),
            second_label,
            fill=(235, 235, 235),
            font=font,
        )
    path = release_root / "paired_condition_comparison.jpg"
    path.write_bytes(jpeg_payload(sheet, 92))
    return path


def generate_release(config_path: Path, release_root: Path, force: bool) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require_runtime(config)
    source_paths = require_source(config)
    config_sha = sha256_file(config_path)
    safe_prepare_release(release_root, force)

    source_rows = read_csv(source_paths["manifest"])
    source_instances = read_jsonl(source_paths["instances"])
    if len(source_rows) != int(config["source"]["expected_scene_count"]):
        raise ValueError("source scene count mismatch")
    if len(source_instances) != int(config["source"]["expected_instance_count"]):
        raise ValueError("source instance count mismatch")
    source_instances_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_instances:
        if row["source_parent_model_split"] != config["source"]["required_parent_model_split"]:
            raise ValueError("non-train parent found in source instances")
        source_instances_by_scene[row["scene_id"]].append(row)
    assignment = build_condition_assignment(config, source_rows)

    manifest_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    lighting_scene_rows: list[dict[str, Any]] = []
    light_source_rows: list[dict[str, Any]] = []
    instances_by_scene: dict[str, list[dict[str, Any]]] = {}
    component_coco_images: list[dict[str, Any]] = []
    component_coco_annotations: list[dict[str, Any]] = []
    defect_coco_annotations: list[dict[str, Any]] = []
    source_component_coco = json.loads(source_paths["component_coco"].read_text(encoding="utf-8"))
    source_defect_coco = json.loads(source_paths["defect_coco"].read_text(encoding="utf-8"))
    source_component_annotations = {
        int(row["id"]): row for row in source_component_coco["annotations"]
    }
    source_defect_annotations = {
        int(row["id"]): row for row in source_defect_coco["annotations"]
    }
    component_annotation_id = 1
    defect_annotation_id = 1
    neutral_context = load_neutral_v4_context(config)
    neutral_plan_by_scene = {plan["scene_id"]: plan for plan in neutral_context.plans}
    max_attempts = 30

    for source_index, source_row in enumerate(sorted(source_rows, key=lambda row: int(row["image_id"]))):
        source_scene_id = source_row["scene_id"]
        source_scene_instances = sorted(
            source_instances_by_scene[source_scene_id], key=lambda row: int(row["instance_index"])
        )
        (
            neutral_defect_array,
            neutral_clean_array,
            neutral_component_mask,
            neutral_defect_mask,
        ) = render_neutral_v4_scene(
            neutral_context, neutral_plan_by_scene[source_scene_id], source_row
        )
        for variant_index in range(int(config["variants_per_source_scene"])):
            output_index = source_index * int(config["variants_per_source_scene"]) + variant_index
            image_id = output_index + 1
            condition = assignment[(source_scene_id, variant_index)]
            scene_id = f"{config['sample_id_prefix']}-{output_index:04d}"
            result: dict[str, Any] | None = None
            last_failures: list[str] = []
            for attempt in range(max_attempts):
                candidate = render_variant(
                    config,
                    source_row,
                    source_scene_instances,
                    condition,
                    neutral_defect_array,
                    neutral_clean_array,
                    neutral_component_mask,
                    neutral_defect_mask,
                    attempt,
                )
                last_failures = list(candidate["failures"])
                if not last_failures:
                    result = candidate
                    break
            if result is None:
                raise RuntimeError(f"scene generation failed {scene_id}: {last_failures}")

            image_path = release_root / "images" / "train" / f"{scene_id}.jpg"
            component_mask_path = release_root / "masks" / "component_visible_instances" / "train" / f"{scene_id}.png"
            defect_mask_path = release_root / "masks" / "defect_semantic" / "train" / f"{scene_id}.png"
            shadow_mask_path = release_root / "masks" / "shadow_attenuation" / "train" / f"{scene_id}.png"
            component_yolo_path = release_root / "labels" / "yolo_component_status" / "train" / f"{scene_id}.txt"
            defect_yolo_path = release_root / "labels" / "yolo_defects" / "train" / f"{scene_id}.txt"
            image_path.write_bytes(result["image_payload"])
            source_component_mask_path = repository_path(source_row["component_instance_mask_path"])
            source_defect_mask_path = repository_path(source_row["defect_semantic_mask_path"])
            source_component_yolo = repository_path(source_row["component_yolo_path"])
            source_defect_yolo = repository_path(source_row["defect_yolo_path"])
            component_mask_path.write_bytes(source_component_mask_path.read_bytes())
            defect_mask_path.write_bytes(source_defect_mask_path.read_bytes())
            component_yolo_path.write_bytes(source_component_yolo.read_bytes())
            defect_yolo_path.write_bytes(source_defect_yolo.read_bytes())
            Image.fromarray(result["shadow_mask"], mode="L").save(shadow_mask_path, optimize=True)

            effect_by_instance = {int(row["instance_index"]): row for row in result["instance_effects"]}
            visibility_by_instance = {
                int(row["instance_index"]): row for row in result["defect_visibility"]
            }
            published_instances: list[dict[str, Any]] = []
            for source_instance in source_scene_instances:
                instance_index = int(source_instance["instance_index"])
                row = deepcopy(source_instance)
                source_component_annotation_id = int(source_instance["component_annotation_id"])
                component_annotation = deepcopy(source_component_annotations[source_component_annotation_id])
                component_annotation["id"] = component_annotation_id
                component_annotation["image_id"] = image_id
                component_annotation["attributes"]["source_component_annotation_id"] = source_component_annotation_id
                component_annotation["attributes"]["source_scene_id"] = source_scene_id
                component_annotation["attributes"]["source_variant_index"] = variant_index
                component_annotation["attributes"]["composition_family_id"] = source_scene_id
                component_annotation["attributes"]["lighting_scene_id"] = scene_id
                component_annotation["attributes"]["visible_instance_mask"] = release_relative(component_mask_path, release_root)
                component_coco_annotations.append(component_annotation)
                current_component_annotation_id = component_annotation_id
                component_annotation_id += 1

                current_defect_annotation_id: int | None = None
                if source_instance["defect_annotation_id"] is not None:
                    source_defect_annotation_id = int(source_instance["defect_annotation_id"])
                    defect_annotation = deepcopy(source_defect_annotations[source_defect_annotation_id])
                    defect_annotation["id"] = defect_annotation_id
                    defect_annotation["image_id"] = image_id
                    defect_annotation["attributes"]["source_defect_annotation_id"] = source_defect_annotation_id
                    defect_annotation["attributes"]["source_scene_id"] = source_scene_id
                    defect_annotation["attributes"]["source_variant_index"] = variant_index
                    defect_annotation["attributes"]["composition_family_id"] = source_scene_id
                    defect_annotation["attributes"]["lighting_scene_id"] = scene_id
                    defect_annotation["attributes"]["component_annotation_id"] = current_component_annotation_id
                    defect_annotation["attributes"]["semantic_mask"] = release_relative(defect_mask_path, release_root)
                    defect_annotation["attributes"]["paired_clean_defect_visibility"] = visibility_by_instance[instance_index]
                    defect_coco_annotations.append(defect_annotation)
                    current_defect_annotation_id = defect_annotation_id
                    defect_annotation_id += 1

                row["scene_id"] = scene_id
                row["image_id"] = image_id
                row["component_annotation_id"] = current_component_annotation_id
                row["defect_annotation_id"] = current_defect_annotation_id
                row["source_scene_id"] = source_scene_id
                row["source_variant_index"] = variant_index
                row["source_component_annotation_id"] = source_component_annotation_id
                row["source_defect_annotation_id"] = source_instance["defect_annotation_id"]
                row["source_composition_family_id"] = source_instance["composition_family_id"]
                row["composition_family_id"] = source_instance["composition_family_id"]
                row["lighting_scene_id"] = scene_id
                row["photometry_domain"] = config["photometry_domain"]
                row["absolute_lux_eligible"] = config["absolute_lux_eligible"]
                row["measured_illuminance_lux"] = None
                row["photometric_calibration_status"] = config["photometric_calibration_status"]
                row["synthetic_illuminance_proxy_bin"] = condition["lux"]["id"]
                row["capture_plan_target_lux"] = int(
                    condition["lux"]["capture_plan_target_lux"]
                )
                row["synthetic_relative_light_power"] = float(condition["lux"]["relative_light_power"])
                row["multi_light_rig_id"] = condition["rig"]["id"]
                row["shadow_regime_id"] = condition["shadow"]["id"]
                row["source_lighting_profile"] = source_row["lighting_profile"]
                row["lighting_effect"] = effect_by_instance[instance_index]
                row["paired_clean_defect_visibility"] = visibility_by_instance.get(
                    instance_index
                )
                row["training_use"] = config["training_use"]
                row["evaluation_eligible"] = config["evaluation_eligible"]
                row["classification_eligible"] = config["classification_eligible"]
                instance_rows.append(row)
                published_instances.append(row)
            instances_by_scene[scene_id] = published_instances

            lighting_scene = {
                "lighting_scene_id": scene_id,
                "scene_id": scene_id,
                "source_scene_id": source_scene_id,
                "source_variant_index": variant_index,
                "pixel_base_method": "v4_geometry_replay_neutral_added_light_shadow_sensor_pre_jpeg",
                "condition_cell_index": int(condition["condition_cell_index"]),
                "condition_cell_id": condition["condition_cell_id"],
                "photometry_domain": config["photometry_domain"],
                "absolute_lux_eligible": config["absolute_lux_eligible"],
                "measured_illuminance_lux": None,
                "photometric_calibration_status": config["photometric_calibration_status"],
                "synthetic_illuminance_proxy_bin": condition["lux"]["id"],
                "capture_plan_target_lux": int(
                    condition["lux"]["capture_plan_target_lux"]
                ),
                "synthetic_lux_proxy": {
                    "relative_light_power": float(condition["lux"]["relative_light_power"]),
                    "unit": condition["lux"]["relative_light_power_unit"],
                    "calibrated_to_lux": bool(condition["lux"]["calibrated_to_lux"]),
                    "method": config["rendering"]["model"],
                },
                "multi_light_rig_id": condition["rig"]["id"],
                "shadow_regime_id": condition["shadow"]["id"],
                "source_count": len(condition["rig"]["sources"]),
                "coordinate_frame": config["coordinate_frame"],
                "sensor_params": result["sensor_params"],
                "cast_shadow_rows": result["cast_shadow_rows"],
                "paired_clean_defect_visibility": result["defect_visibility"],
                "metrics": {key: round(float(value), 8) for key, value in result["metrics"].items()},
            }
            lighting_scene_rows.append(lighting_scene)
            for source in condition["rig"]["sources"]:
                azimuth = float(source["image_plane_azimuth_deg"])
                radians = math.radians(azimuth)
                light_source_rows.append({
                    "lighting_scene_id": scene_id,
                    "light_id": source["light_id"],
                    "multi_light_rig_id": condition["rig"]["id"],
                    "coordinate_frame_name": config["coordinate_frame"]["name"],
                    "image_plane_azimuth_deg": azimuth,
                    "elevation_proxy_deg": float(source["elevation_proxy_deg"]),
                    "direction_vector_image_xy": [round(math.cos(radians), 10), round(math.sin(radians), 10)],
                    "relative_intensity": float(source["relative_intensity"]),
                    "relative_intensity_unit": "1",
                    "cct_proxy_kelvin": int(source["cct_proxy_kelvin"]),
                    "cct_calibrated": False,
                    "anchor_xy_fraction": source["anchor_xy_fraction"],
                })

            metrics = result["metrics"]
            manifest_row = {
                "scene_id": scene_id,
                "image_id": image_id,
                "image_path": repository_relative(image_path),
                "component_instance_mask_path": repository_relative(component_mask_path),
                "defect_semantic_mask_path": repository_relative(defect_mask_path),
                "shadow_attenuation_mask_path": repository_relative(shadow_mask_path),
                "component_yolo_path": repository_relative(component_yolo_path),
                "defect_yolo_path": repository_relative(defect_yolo_path),
                "source_scene_id": source_scene_id,
                "source_variant_index": variant_index,
                "source_image_path": source_row["image_path"],
                "pixel_base_method": "v4_geometry_replay_neutral_added_light_shadow_sensor_pre_jpeg",
                "source_lighting_profile": source_row["lighting_profile"],
                "source_image_sha256": source_row["image_sha256"],
                "source_component_mask_sha256": source_row["component_mask_sha256"],
                "source_defect_mask_sha256": source_row["defect_mask_sha256"],
                "split": config["split"],
                "task_type": config["task_type"],
                "training_use": config["training_use"],
                "evaluation_eligible": config["evaluation_eligible"],
                "classification_eligible": config["classification_eligible"],
                "photometry_domain": config["photometry_domain"],
                "absolute_lux_eligible": config["absolute_lux_eligible"],
                "measured_illuminance_lux": "",
                "photometric_calibration_status": config["photometric_calibration_status"],
                "synthetic_illuminance_proxy_bin": condition["lux"]["id"],
                "capture_plan_target_lux": int(
                    condition["lux"]["capture_plan_target_lux"]
                ),
                "synthetic_relative_light_power": condition["lux"]["relative_light_power"],
                "relative_light_power_unit": condition["lux"]["relative_light_power_unit"],
                "calibrated_to_lux": "NO",
                "multi_light_rig_id": condition["rig"]["id"],
                "shadow_regime_id": condition["shadow"]["id"],
                "condition_cell_id": condition["condition_cell_id"],
                "condition_cell_index": int(condition["condition_cell_index"]),
                "light_source_count": len(condition["rig"]["sources"]),
                "coordinate_frame_name": config["coordinate_frame"]["name"],
                "scene_seed": int(result["scene_seed"]),
                "attempt": int(result["attempt"]),
                "instance_count": len(published_instances),
                "component_status_labels": source_row["component_status_labels"],
                "source_parent_ids": source_row["source_parent_ids"],
                "composition_family_id": source_scene_id,
                "width": int(source_row["width"]),
                "height": int(source_row["height"]),
                "image_sha256": sha256_file(image_path),
                "component_mask_sha256": sha256_file(component_mask_path),
                "defect_mask_sha256": sha256_file(defect_mask_path),
                "shadow_mask_sha256": sha256_file(shadow_mask_path),
                "component_yolo_sha256": sha256_file(component_yolo_path),
                "defect_yolo_sha256": sha256_file(defect_yolo_path),
                "background_mean_luma": f"{metrics['background_mean_luma']:.6f}",
                "background_p99_luma": f"{metrics['background_p99_luma']:.6f}",
                "component_mean_luma": f"{metrics['component_mean_luma']:.6f}",
                "component_dark_fraction": f"{metrics['component_dark_fraction']:.8f}",
                "component_saturated_fraction": f"{metrics['component_saturated_fraction']:.8f}",
                "pre_sensor_positive_spill_max": f"{metrics['pre_sensor_positive_spill_max']:.6f}",
                "post_jpeg_paired_clean_spill_p99": f"{metrics['post_jpeg_paired_clean_spill_p99']:.6f}",
                "post_jpeg_paired_clean_spill_max": f"{metrics['post_jpeg_paired_clean_spill_max']:.6f}",
                "post_jpeg_paired_clean_spill_fraction": f"{metrics['post_jpeg_paired_clean_spill_fraction']:.8f}",
                "post_jpeg_paired_clean_spill_energy": f"{metrics['post_jpeg_paired_clean_spill_energy']:.6f}",
                "minimum_defect_mean_abs_delta": f"{min(float(item['mean_abs_delta']) for item in result['defect_visibility']):.6f}",
                "minimum_defect_changed_fraction": f"{min(float(item['changed_fraction']) for item in result['defect_visibility']):.8f}",
                "shadow_nonzero_fraction": f"{metrics['shadow_nonzero_fraction']:.8f}",
                "shadow_max_attenuation": f"{metrics['shadow_max_attenuation']:.8f}",
                "config_sha256": config_sha,
                "generator_version": config["generator_version"],
                "qc_gate_version": config["qc_gate_version"],
                "qc_status": "AUTO_PASS_MULTI_LIGHT_LUX_PROXY_REPLAY",
                "human_verified": "NO",
                "sensor_params_json": canonical_json(result["sensor_params"]),
                "light_sources_json": canonical_json(condition["rig"]["sources"]),
                "cast_shadow_params_json": canonical_json(result["cast_shadow_rows"]),
            }
            manifest_rows.append(manifest_row)
            component_coco_images.append({
                "id": image_id,
                "file_name": release_relative(image_path, release_root),
                "width": int(source_row["width"]),
                "height": int(source_row["height"]),
                "scene_id": scene_id,
                "source_scene_id": source_scene_id,
                "source_variant_index": variant_index,
                "composition_family_id": source_scene_id,
                "synthetic_illuminance_proxy_bin": condition["lux"]["id"],
                "capture_plan_target_lux": int(
                    condition["lux"]["capture_plan_target_lux"]
                ),
                "synthetic_relative_light_power": float(condition["lux"]["relative_light_power"]),
                "multi_light_rig_id": condition["rig"]["id"],
                "shadow_regime_id": condition["shadow"]["id"],
                "absolute_lux_eligible": config["absolute_lux_eligible"],
                "measured_illuminance_lux": None,
                "photometry_domain": config["photometry_domain"],
            })
            if (output_index + 1) % 24 == 0:
                print(f"generated scenes={output_index + 1}/{config['scene_count']}", flush=True)

    manifest_fields = list(manifest_rows[0].keys())
    manifest_path = release_root / "annotations" / "manifest.csv"
    write_csv(manifest_path, manifest_rows, manifest_fields)
    instances_path = release_root / "annotations" / "instances.jsonl"
    with instances_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in instance_rows:
            stream.write(canonical_json(row) + "\n")
    lighting_scenes_path = release_root / "annotations" / "lighting_scenes.jsonl"
    with lighting_scenes_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in lighting_scene_rows:
            stream.write(canonical_json(row) + "\n")
    light_sources_path = release_root / "annotations" / "light_sources.jsonl"
    with light_sources_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in light_source_rows:
            stream.write(canonical_json(row) + "\n")

    component_coco_path = release_root / "annotations" / "coco" / "component_status_train.json"
    defect_coco_path = release_root / "annotations" / "coco" / "defects_train.json"
    common_info = {
        "description": "Synthetic paired multi-light illuminance-proxy train-only auxiliary release",
        "version": config["generator_version"],
        "year": 2026,
        "photometry_domain": config["photometry_domain"],
        "absolute_lux_eligible": config["absolute_lux_eligible"],
    }
    write_json(component_coco_path, {
        "info": common_info,
        "licenses": deepcopy(source_component_coco.get("licenses", [])),
        "images": component_coco_images,
        "annotations": component_coco_annotations,
        "categories": deepcopy(source_component_coco["categories"]),
    })
    write_json(defect_coco_path, {
        "info": common_info,
        "licenses": deepcopy(source_defect_coco.get("licenses", [])),
        "images": component_coco_images,
        "annotations": defect_coco_annotations,
        "categories": deepcopy(source_defect_coco["categories"]),
    })

    expected_scenes = int(config["scene_count"])
    expected_instances = expected_scenes * int(config["instances_per_scene"])
    expected_defects = sum(
        row["component_status_class"] != "normal_proxy" for row in instance_rows
    )
    cardinality_checks = {
        "manifest scenes": (len(manifest_rows), expected_scenes),
        "lighting scenes": (len(lighting_scene_rows), expected_scenes),
        "COCO images": (len(component_coco_images), expected_scenes),
        "instances": (len(instance_rows), expected_instances),
        "component annotations": (len(component_coco_annotations), expected_instances),
        "defect annotations": (len(defect_coco_annotations), expected_defects),
    }
    for label, (actual, expected) in cardinality_checks.items():
        if actual != expected:
            raise RuntimeError(f"{label} count mismatch: actual={actual} expected={expected}")

    class_counts = Counter(row["component_status_class"] for row in instance_rows)
    lux_scene_counts = Counter(
        row["synthetic_illuminance_proxy_bin"] for row in manifest_rows
    )
    rig_scene_counts = Counter(row["multi_light_rig_id"] for row in manifest_rows)
    rig_lux_scene_counts = Counter(
        (row["multi_light_rig_id"], row["synthetic_illuminance_proxy_bin"])
        for row in manifest_rows
    )
    shadow_scene_counts = Counter(row["shadow_regime_id"] for row in manifest_rows)
    cell_scene_counts = Counter(row["condition_cell_id"] for row in manifest_rows)
    source_profile_cell_counts = Counter(
        (row["source_lighting_profile"], row["condition_cell_id"])
        for row in manifest_rows
    )
    class_lux_counts = Counter(
        (row["component_status_class"], row["synthetic_illuminance_proxy_bin"])
        for row in instance_rows
    )
    class_rig_counts = Counter(
        (row["component_status_class"], row["multi_light_rig_id"])
        for row in instance_rows
    )
    class_shadow_counts = Counter(
        (row["component_status_class"], row["shadow_regime_id"])
        for row in instance_rows
    )
    class_rig_lux_counts = Counter(
        (
            row["component_status_class"],
            row["multi_light_rig_id"],
            row["synthetic_illuminance_proxy_bin"],
        )
        for row in instance_rows
    )
    class_cell_counts = Counter(
        (
            row["component_status_class"],
            f"{row['multi_light_rig_id']}__{row['synthetic_illuminance_proxy_bin']}__{row['shadow_regime_id']}",
        )
        for row in instance_rows
    )
    source_scene_reuse_counts = Counter(row["source_scene_id"] for row in manifest_rows)
    expected_cardinalities = {
        "classes": (len(class_counts), len(config["classes"])),
        "proxy bins": (len(lux_scene_counts), len(config["target_illuminance_bins"])),
        "rigs": (len(rig_scene_counts), len(config["multi_light_rigs"])),
        "rig proxy cells": (
            len(rig_lux_scene_counts),
            len(config["multi_light_rigs"]) * len(config["target_illuminance_bins"]),
        ),
        "shadow regimes": (len(shadow_scene_counts), len(config["shadow_regimes"])),
        "full condition cells": (
            len(cell_scene_counts),
            len(config["multi_light_rigs"])
            * len(config["target_illuminance_bins"])
            * len(config["shadow_regimes"]),
        ),
        "source scenes": (
            len(source_scene_reuse_counts),
            int(config["source"]["expected_scene_count"]),
        ),
    }
    for label, (actual, expected) in expected_cardinalities.items():
        if actual != expected:
            raise RuntimeError(
                f"{label} cardinality mismatch: actual={actual} expected={expected}"
            )
    contract = config["balance_contract"]
    checks: list[tuple[Iterable[int], int, str]] = [
        (class_counts.values(), int(contract["instance_per_class"]), "class"),
        (lux_scene_counts.values(), int(contract["scene_per_illuminance_bin"]), "lux scene"),
        (rig_scene_counts.values(), int(contract["scene_per_rig"]), "rig scene"),
        (rig_lux_scene_counts.values(), int(contract["scene_per_rig_illuminance_cell"]), "rig lux scene"),
        (shadow_scene_counts.values(), int(contract["scene_per_shadow_regime"]), "shadow scene"),
        (cell_scene_counts.values(), int(contract["scene_per_full_condition_cell"]), "condition cell scene"),
        (source_profile_cell_counts.values(), int(contract["source_profile_scene_per_full_condition_cell"]), "source profile cell"),
        (class_shadow_counts.values(), int(contract["instance_per_class_shadow_regime"]), "class shadow"),
        (source_scene_reuse_counts.values(), int(contract["source_scene_reuse"]), "source scene reuse"),
    ]
    for values, expected, label in checks:
        if set(values) != {expected}:
            raise RuntimeError(f"{label} balance mismatch: {Counter(values)}")
    ranged_checks = [
        (
            class_lux_counts.values(),
            contract["instance_per_class_illuminance_bin_range"],
            "class lux",
        ),
        (
            class_rig_counts.values(),
            contract["instance_per_class_rig_range"],
            "class rig",
        ),
        (
            class_rig_lux_counts.values(),
            contract["instance_per_class_rig_illuminance_cell_range"],
            "class rig lux",
        ),
    ]
    for values, allowed, label in ranged_checks:
        low, high = [int(value) for value in allowed]
        if any(not low <= value <= high for value in values):
            raise RuntimeError(f"{label} balance mismatch: {Counter(values)}")
    class_cell_low, class_cell_high = [
        int(value) for value in contract["instance_per_class_full_condition_cell_range"]
    ]
    if any(
        not class_cell_low <= value <= class_cell_high
        for value in class_cell_counts.values()
    ):
        raise RuntimeError(
            f"class condition cell balance mismatch: {Counter(class_cell_counts.values())}"
        )

    summary_rows: list[dict[str, Any]] = []
    for name in config["classes"]:
        summary_rows.append({"dimension": "component_status_class", "value": name, "count": class_counts[name]})
    for lux in config["target_illuminance_bins"]:
        summary_rows.append({"dimension": "synthetic_illuminance_proxy_bin_scene", "value": lux["id"], "count": lux_scene_counts[lux["id"]]})
    for rig in config["multi_light_rigs"]:
        summary_rows.append({"dimension": "multi_light_rig_scene", "value": rig["id"], "count": rig_scene_counts[rig["id"]]})
    for shadow in config["shadow_regimes"]:
        summary_rows.append({"dimension": "shadow_regime_scene", "value": shadow["id"], "count": shadow_scene_counts[shadow["id"]]})
    summary_path = release_root / "annotations" / "summary.csv"
    write_csv(summary_path, summary_rows, ["dimension", "value", "count"])
    condition_rows: list[dict[str, Any]] = []
    for rig in config["multi_light_rigs"]:
        for lux in config["target_illuminance_bins"]:
            for shadow in config["shadow_regimes"]:
                cell_id = f"{rig['id']}__{lux['id']}__{shadow['id']}"
                row: dict[str, Any] = {
                    "condition_cell_id": cell_id,
                    "multi_light_rig_id": rig["id"],
                    "synthetic_illuminance_proxy_bin": lux["id"],
                    "capture_plan_target_lux": int(lux["capture_plan_target_lux"]),
                    "shadow_regime_id": shadow["id"],
                    "scene_count": cell_scene_counts[cell_id],
                }
                for class_name in config["classes"]:
                    row[f"class_{class_name}"] = class_cell_counts[
                        (class_name, cell_id)
                    ]
                condition_rows.append(row)
    condition_matrix_path = release_root / "annotations" / "condition_matrix.csv"
    write_csv(
        condition_matrix_path,
        condition_rows,
        list(condition_rows[0].keys()),
    )

    contact_sheet = create_contact_sheet(config, manifest_rows, instances_by_scene, release_root, False)
    overlay_sheet = create_contact_sheet(config, manifest_rows, instances_by_scene, release_root, True)
    paired_sheet = create_paired_comparison_sheet(manifest_rows, release_root)

    release_path = release_root / "annotations" / "release.json"
    metadata = {
        "release": config["release"],
        "generator_version": config["generator_version"],
        "qc_gate_version": config["qc_gate_version"],
        "task_type": config["task_type"],
        "split": config["split"],
        "training_use": config["training_use"],
        "evaluation_eligible": config["evaluation_eligible"],
        "classification_eligible": config["classification_eligible"],
        "photometry_domain": config["photometry_domain"],
        "absolute_lux_eligible": config["absolute_lux_eligible"],
        "measured_illuminance_lux": None,
        "photometric_calibration_status": config["photometric_calibration_status"],
        "config_path": repository_relative(config_path),
        "config_sha256": config_sha,
        "generator_script": repository_relative(Path(__file__)),
        "generator_script_sha256": sha256_file(Path(__file__)),
        "runtime_contract": current_runtime_versions(),
        "source_release": config["source"]["release"],
        "source_pins": {
            key: sha256_file(path) for key, path in source_paths.items()
        },
        "scene_count": len(manifest_rows),
        "source_scene_count": len(source_scene_reuse_counts),
        "variants_per_source_scene": int(config["variants_per_source_scene"]),
        "composition_family_policy": "SOURCE_V4_SCENE_SHARED_BY_BOTH_VARIANTS",
        "pixel_base_method": "v4_geometry_replay_neutral_added_light_shadow_sensor_pre_jpeg",
        "instances_per_scene": int(config["instances_per_scene"]),
        "component_instance_count": len(instance_rows),
        "defect_instance_count": expected_defects,
        "class_counts": dict(sorted(class_counts.items())),
        "synthetic_illuminance_proxy_bin_scene_counts": dict(sorted(lux_scene_counts.items())),
        "multi_light_rig_scene_counts": dict(sorted(rig_scene_counts.items())),
        "shadow_regime_scene_counts": dict(sorted(shadow_scene_counts.items())),
        "condition_cell_count": len(cell_scene_counts),
        "condition_cell_scene_count_range": [min(cell_scene_counts.values()), max(cell_scene_counts.values())],
        "class_condition_cell_count_range": [min(class_cell_counts.values()), max(class_cell_counts.values())],
        "class_illuminance_proxy_bin_count_range": [min(class_lux_counts.values()), max(class_lux_counts.values())],
        "class_rig_count_range": [min(class_rig_counts.values()), max(class_rig_counts.values())],
        "class_rig_illuminance_proxy_count_range": [min(class_rig_lux_counts.values()), max(class_rig_lux_counts.values())],
        "capture_plan_target_lux_by_proxy_bin": {
            item["id"]: int(item["capture_plan_target_lux"])
            for item in config["target_illuminance_bins"]
        },
        "manifest_sha256": sha256_file(manifest_path),
        "instances_sha256": sha256_file(instances_path),
        "lighting_scenes_sha256": sha256_file(lighting_scenes_path),
        "light_sources_sha256": sha256_file(light_sources_path),
        "component_coco_sha256": sha256_file(component_coco_path),
        "defect_coco_sha256": sha256_file(defect_coco_path),
        "summary_sha256": sha256_file(summary_path),
        "condition_matrix_sha256": sha256_file(condition_matrix_path),
        "contact_sheet_sha256": sha256_file(contact_sheet),
        "contact_sheet_overlay_sha256": sha256_file(overlay_sheet),
        "paired_condition_comparison_sha256": sha256_file(paired_sheet),
        "limitations": [
            "capture_plan_target_lux is a future real-capture plan target, not measured synthetic lux.",
            "All photometric changes are deterministic 2D proxies and are not radiometrically calibrated.",
            "Each v4 composition is replayed twice under distinct proxy lighting conditions; both variants and their v4 source share one composition_family_id and must stay in one split.",
            "All scenes inherit the same train-only v4/v2 physical base family and must not enter validation or test.",
            "normal_proxy is paired-clean synthetic data and is not confirmed real OK data.",
            "This auxiliary release changes lighting appearance but adds no new real specimen diversity.",
        ],
    }
    maximum_payload = float(config["qc"]["maximum_payload_mib"]) * 1024 * 1024
    for _ in range(8):
        payload = sum(path.stat().st_size for path in release_root.rglob("*") if path.is_file())
        if metadata.get("tracked_payload_bytes") == payload:
            break
        metadata["tracked_payload_bytes"] = payload
        write_json(release_path, metadata)
    else:
        raise RuntimeError("release payload size did not stabilize")
    final_payload = sum(path.stat().st_size for path in release_root.rglob("*") if path.is_file())
    if final_payload != int(metadata["tracked_payload_bytes"]):
        raise RuntimeError("release payload byte count mismatch")
    if final_payload > maximum_payload:
        raise RuntimeError(
            f"release payload {final_payload / 1024 / 1024:.2f} MiB exceeds gate"
        )
    print(
        f"PASS generated_scenes={len(manifest_rows)} instances={len(instance_rows)} "
        f"lux_bins={len(config['target_illuminance_bins'])} rigs={len(config['multi_light_rigs'])} "
        f"shadow_regimes={len(config['shadow_regimes'])} payload_mib={final_payload / 1024 / 1024:.2f} "
        "measured_lux=NONE train_only=YES",
        flush=True,
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generate_release(args.config, args.release, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
