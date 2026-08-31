from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import generate_synthetic_v1_450 as legacy
import generate_synthetic_v2_700 as v2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "synthetic_v3_conditions.json"
DEFAULT_OUTPUT = ROOT / "synthetic" / "v3_conditions"
MARKER_NAME = ".synthetic_v3_conditions_marker"


def sha256_file(path: Path) -> str:
    return v2.sha256_file(path)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def stable_condition_seed(global_seed: int, release: str, parent_id: str, profile: str) -> int:
    material = f"{global_seed}|{release}|{parent_id}|{profile}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def reconstruct_parent(
    parent_row: dict[str, str], v2_config: dict[str, Any]
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Replay a v2 parent before its final JPEG and verify the published parent."""
    base_path = ROOT / v2_config["base"]["path"]
    base = Image.open(base_path).convert("RGB")
    size = int(v2_config["image_size"])
    rois = legacy.build_rois(v2_config)
    class_ids = {key: int(value) for key, value in v2_config["class_ids"].items()}
    class_name = parent_row["primary_class"]
    severity = parent_row["severity"]
    parameters = json.loads(parent_row["parameters_json"])
    rng = random.Random(int(parameters["effective_seed"]))
    defect, semantic, instance_params, _, _ = v2.apply_single_recipe(
        base, class_name, severity, rois, class_ids, rng
    )
    if instance_params != parameters["instance_pre_transform"]:
        raise ValueError(f"parent recipe replay mismatch: {parent_row['sample_id']}")
    geometry = parameters["geometry"]
    photometric = parameters["photometric"]
    defect, semantic = v2.apply_geometry(defect, semantic, geometry, size)
    clean, _ = v2.apply_geometry(base, Image.new("L", base.size, 0), geometry, size)
    defect = v2.apply_photometric(defect, photometric)
    clean = v2.apply_photometric(clean, photometric)

    _, parent_payload = v2.jpeg_roundtrip(defect, int(v2_config["jpeg_quality"]))
    if sha256_bytes(parent_payload) != parent_row["image_sha256"]:
        raise ValueError(f"parent image replay mismatch: {parent_row['sample_id']}")
    parent_mask = Image.open(ROOT / parent_row["mask_path"]).convert("L")
    parent_mask.load()
    if not np.array_equal(np.array(parent_mask), np.array(semantic)):
        raise ValueError(f"parent mask replay mismatch: {parent_row['sample_id']}")
    return defect, clean, semantic


def uniform_pair(rng: random.Random, bounds: list[float]) -> float:
    return rng.uniform(float(bounds[0]), float(bounds[1]))


def sample_condition(
    config: dict[str, Any], profile: str, class_name: str, seed: int
) -> dict[str, Any]:
    rng = random.Random(seed)
    ranges = config["profile_ranges"][profile]
    params: dict[str, Any] = {
        "profile": profile,
        "seed": seed,
        "exposure_ev": uniform_pair(rng, ranges["exposure_ev"]),
        "gamma": uniform_pair(rng, ranges["gamma"]),
        "contrast": uniform_pair(rng, ranges["contrast"]),
        "channel_gain_r": uniform_pair(rng, ranges.get("channel_gain_r", [0.98, 1.02])),
        "channel_gain_g": uniform_pair(rng, ranges.get("channel_gain_g", [0.98, 1.02])),
        "channel_gain_b": uniform_pair(rng, ranges.get("channel_gain_b", [0.98, 1.02])),
        "gradient_strength": uniform_pair(rng, ranges.get("gradient_strength", [0.0, 0.0])),
        "gradient_angle_deg": uniform_pair(rng, ranges.get("gradient_angle_deg", [0.0, 0.0])),
        "shadow_strength": uniform_pair(rng, ranges.get("shadow_strength", [0.0, 0.0])),
        "shadow_width": uniform_pair(rng, ranges.get("shadow_width", [0.25, 0.25])),
        "shadow_angle_deg": rng.uniform(0.0, 360.0),
        "shadow_offset": rng.uniform(-0.35, 0.35),
        "vignette_strength": uniform_pair(rng, ranges.get("vignette_strength", [0.0, 0.0])),
        "hotspot_strength": uniform_pair(rng, ranges.get("hotspot_strength", [0.0, 0.0])),
        "hotspot_sigma": uniform_pair(rng, ranges.get("hotspot_sigma", [0.18, 0.18])),
        "hotspot_center_x": rng.choice([rng.uniform(0.10, 0.30), rng.uniform(0.70, 0.90)]),
        "hotspot_center_y": rng.choice([rng.uniform(0.10, 0.30), rng.uniform(0.70, 0.90)]),
        "blur_radius": uniform_pair(rng, ranges["blur_radius"]),
        "noise_sigma": uniform_pair(rng, ranges["noise_sigma"]),
        "noise_seed": rng.randrange(0, 2**32),
        "jpeg_quality": rng.randint(int(ranges["jpeg_quality"][0]), int(ranges["jpeg_quality"][1])),
    }
    override = config.get("class_overrides", {}).get(class_name, {})
    if override:
        floor = float(override.get("channel_gain_floor", 0.0))
        ceiling = float(override.get("channel_gain_ceiling", 99.0))
        for channel in ("r", "g", "b"):
            key = f"channel_gain_{channel}"
            params[key] = min(ceiling, max(floor, float(params[key])))
        params["hotspot_strength"] = min(
            float(params["hotspot_strength"]),
            float(override.get("hotspot_strength_ceiling", 99.0)),
        )
        params["exposure_ev"] = max(
            float(params["exposure_ev"]),
            float(override.get("exposure_ev_floor", -99.0)),
        )
        params["gamma"] = min(
            float(params["gamma"]),
            float(override.get("gamma_ceiling", 99.0)),
        )
        params["contrast"] = max(
            float(params["contrast"]),
            float(override.get("contrast_floor", 0.0)),
        )
        params["blur_radius"] = min(
            float(params["blur_radius"]),
            float(override.get("blur_radius_ceiling", 99.0)),
        )
        params["noise_sigma"] = min(
            float(params["noise_sigma"]),
            float(override.get("noise_sigma_ceiling", 99.0)),
        )
        params["jpeg_quality"] = max(
            int(params["jpeg_quality"]),
            int(override.get("jpeg_quality_floor", 0)),
        )
    return params


def apply_condition(image: Image.Image, params: dict[str, Any]) -> Image.Image:
    """Apply only camera/illumination effects; geometry and defect pixels stay aligned."""
    array = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width = array.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x = xx / max(1, width - 1) * 2.0 - 1.0
    y = yy / max(1, height - 1) * 2.0 - 1.0

    array *= 2.0 ** float(params["exposure_ev"])
    array = np.clip(array, 0.0, 1.0) ** float(params["gamma"])
    array = (array - 0.5) * float(params["contrast"]) + 0.5
    gains = np.array(
        [params["channel_gain_r"], params["channel_gain_g"], params["channel_gain_b"]],
        dtype=np.float32,
    )
    array *= gains[None, None, :]

    field = np.ones((height, width), dtype=np.float32)
    gradient_strength = float(params["gradient_strength"])
    if gradient_strength:
        angle = math.radians(float(params["gradient_angle_deg"]))
        projection = (math.cos(angle) * x + math.sin(angle) * y) / math.sqrt(2.0)
        field *= 1.0 + gradient_strength * projection

    shadow_strength = float(params["shadow_strength"])
    if shadow_strength:
        angle = math.radians(float(params["shadow_angle_deg"]))
        distance = math.cos(angle) * x + math.sin(angle) * y - float(params["shadow_offset"])
        width_norm = max(0.04, float(params["shadow_width"]))
        band = np.exp(-0.5 * (distance / width_norm) ** 2)
        field *= 1.0 - shadow_strength * band

    vignette_strength = float(params["vignette_strength"])
    if vignette_strength:
        radius2 = np.clip((x * x + y * y) / 2.0, 0.0, 1.0)
        field *= 1.0 - vignette_strength * radius2

    hotspot_strength = float(params["hotspot_strength"])
    if hotspot_strength:
        cx = float(params["hotspot_center_x"]) * 2.0 - 1.0
        cy = float(params["hotspot_center_y"]) * 2.0 - 1.0
        sigma = max(0.04, float(params["hotspot_sigma"])) * 2.0
        hotspot = np.exp(-0.5 * (((x - cx) / sigma) ** 2 + ((y - cy) / sigma) ** 2))
        field *= 1.0 + hotspot_strength * hotspot

    array *= field[..., None]
    noise_sigma = float(params["noise_sigma"]) / 255.0
    if noise_sigma:
        noise_rng = np.random.default_rng(int(params["noise_seed"]))
        array += noise_rng.normal(0.0, noise_sigma, array.shape)
    output = Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    blur = float(params["blur_radius"])
    if blur > 0.04:
        output = output.filter(ImageFilter.GaussianBlur(blur))
    return output


def attenuate_condition(
    params: dict[str, Any], strength_scale: float, reference_jpeg_quality: int
) -> dict[str, Any]:
    """Move an over-hard camera condition toward the verified parent condition."""
    scale = float(np.clip(strength_scale, 0.0, 1.0))
    output = dict(params)
    output["condition_strength_scale"] = scale
    output["exposure_ev"] = float(params["exposure_ev"]) * scale
    for key in ("gamma", "contrast", "channel_gain_r", "channel_gain_g", "channel_gain_b"):
        output[key] = 1.0 + (float(params[key]) - 1.0) * scale
    for key in (
        "gradient_strength",
        "shadow_strength",
        "vignette_strength",
        "hotspot_strength",
        "blur_radius",
        "noise_sigma",
    ):
        output[key] = float(params[key]) * scale
    output["jpeg_quality"] = int(
        round(reference_jpeg_quality + (int(params["jpeg_quality"]) - reference_jpeg_quality) * scale)
    )
    return output


def luma_metrics(image: Image.Image, config: dict[str, Any]) -> dict[str, float]:
    array = np.array(image.convert("RGB"), dtype=np.float32)
    luma = 0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]
    black = float(config["qc"]["black_clip_value"])
    white = float(config["qc"]["white_clip_value"])
    return {
        "mean_luma": round(float(luma.mean()), 6),
        "luma_std": round(float(luma.std()), 6),
        "black_clip_fraction": round(float(np.mean(luma <= black)), 8),
        "white_clip_fraction": round(float(np.mean(luma >= white)), 8),
    }


def luma_failures(metrics: dict[str, float], config: dict[str, Any]) -> list[str]:
    qc = config["qc"]
    failures: list[str] = []
    if metrics["mean_luma"] < float(qc["min_mean_luma"]):
        failures.append("mean_luma_low")
    if metrics["mean_luma"] > float(qc["max_mean_luma"]):
        failures.append("mean_luma_high")
    if metrics["luma_std"] < float(qc["min_luma_std"]):
        failures.append("luma_std_low")
    if metrics["black_clip_fraction"] > float(qc["max_black_clip_fraction"]):
        failures.append("black_clip_high")
    if metrics["white_clip_fraction"] > float(qc["max_white_clip_fraction"]):
        failures.append("white_clip_high")
    return failures


def prepare_output(path: Path, force: bool) -> None:
    marker = path / MARKER_NAME
    if path.exists():
        if not force:
            raise FileExistsError(f"output already exists: {path}; use --force to regenerate")
        resolved = path.resolve()
        synthetic_root = (ROOT / "synthetic").resolve()
        if synthetic_root not in resolved.parents or not marker.exists():
            raise RuntimeError(f"refusing to remove unmarked or out-of-scope directory: {resolved}")
        shutil.rmtree(resolved)
    (path / "images").mkdir(parents=True)
    (path / "masks").mkdir(parents=True)
    (path / "annotations" / "contact_sheets").mkdir(parents=True)
    marker.write_text("synthetic v3 condition release\n", encoding="utf-8")


def save_contact_sheets(
    output: Path, rows: list[dict[str, str]], classes: list[str], profiles: list[str]
) -> tuple[str, dict[str, str]]:
    font = ImageFont.load_default()
    tile = 224
    label_height = 18
    overview = Image.new("RGB", (len(profiles) * tile, len(classes) * (tile + label_height)), (24, 24, 24))
    draw = ImageDraw.Draw(overview)
    for row_index, class_name in enumerate(classes):
        for column, profile in enumerate(profiles):
            row = next(
                item for item in rows
                if item["primary_class"] == class_name and item["condition_profile"] == profile
            )
            image = Image.open(ROOT / row["image_path"]).convert("RGB").resize((tile, tile), Image.Resampling.BILINEAR)
            x = column * tile
            y = row_index * (tile + label_height)
            overview.paste(image, (x, y + label_height))
            draw.text((x + 3, y + 3), f"{class_name[:8]} | {profile[:12]}", fill=(245, 245, 245), font=font)
    overview_path = output / "contact_sheet.jpg"
    overview.save(overview_path, format="JPEG", quality=92, subsampling=0)

    hashes: dict[str, str] = {}
    full_tile = 160
    columns = 12
    for class_name in classes:
        class_rows = [row for row in rows if row["primary_class"] == class_name]
        sheet_rows = math.ceil(len(class_rows) / columns)
        for kind in ("raw", "overlay"):
            sheet = Image.new("RGB", (columns * full_tile, sheet_rows * (full_tile + label_height)), (24, 24, 24))
            sheet_draw = ImageDraw.Draw(sheet)
            for index, row in enumerate(class_rows):
                image = Image.open(ROOT / row["image_path"]).convert("RGB")
                if kind == "overlay":
                    mask = Image.open(ROOT / row["mask_path"]).convert("L")
                    image = legacy.make_overlay(image, mask)
                image = image.resize((full_tile, full_tile), Image.Resampling.BILINEAR)
                column = index % columns
                sheet_row = index // columns
                x = column * full_tile
                y = sheet_row * (full_tile + label_height)
                sheet.paste(image, (x, y + label_height))
                sheet_draw.text((x + 2, y + 3), f"{index:03d} {row['condition_profile'][:9]}", fill=(245, 245, 245), font=font)
            path = output / "annotations" / "contact_sheets" / f"{class_name}_{kind}_144_at_160.jpg"
            sheet.save(path, format="JPEG", quality=90, subsampling=0)
            hashes[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    return sha256_file(overview_path), hashes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate leakage-safe v3 lighting/camera condition variants.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    config_sha = sha256_bytes(config_bytes)
    output = args.output.resolve()

    source = config["source"]
    source_manifest_path = ROOT / source["manifest_path"]
    source_config_path = ROOT / source["config_path"]
    split_path = ROOT / source["split_assignments_path"]
    pinned = {
        source_manifest_path: source["manifest_sha256"],
        source_config_path: source["config_sha256"],
        split_path: source["split_assignments_sha256"],
    }
    for path, expected in pinned.items():
        if not path.exists() or sha256_file(path) != expected:
            raise ValueError(f"pinned source mismatch: {path}")

    v2_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    parent_rows = {row["sample_id"]: row for row in read_csv(source_manifest_path)}
    split_rows = read_csv(split_path)
    selected_ids = [
        row["sample_id"] for row in split_rows
        if row["model_split"] == source["required_parent_model_split"]
    ]
    if len(selected_ids) != int(source["expected_parent_count"]) or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("unexpected gradient-train parent inventory")
    selected = [parent_rows[sample_id] for sample_id in selected_ids]
    expected_per_class = int(source["expected_parent_count_per_class"])
    if Counter(row["primary_class"] for row in selected) != Counter(
        {class_name: expected_per_class for class_name in v2_config["primary_classes"]}
    ):
        raise ValueError("unexpected gradient-train class counts")

    prepare_output(output, args.force)
    classes = list(v2_config["primary_classes"])
    profiles = list(config["profiles"])
    class_ids = {key: int(value) for key, value in v2_config["class_ids"].items()}
    max_attempts = int(config["max_condition_attempts"])
    size = int(config["image_size"])
    rows: list[dict[str, str]] = []
    instances: list[dict[str, Any]] = []

    selected.sort(key=lambda row: (classes.index(row["primary_class"]), row["sample_id"]))
    for parent_index, parent in enumerate(selected):
        class_name = parent["primary_class"]
        severity = parent["severity"]
        parent_defect, parent_clean, semantic = reconstruct_parent(parent, v2_config)
        mask_array = np.array(semantic, dtype=np.uint8)
        class_mask = mask_array == class_ids[class_name]
        bbox = legacy.mask_bbox(class_mask)
        defect_pixels = int(np.count_nonzero(class_mask))
        for variant_index, profile in enumerate(profiles):
            condition_seed = stable_condition_seed(
                int(config["global_seed"]), config["release"], parent["sample_id"], profile
            )
            last_failures: list[str] = []
            for attempt in range(max_attempts):
                effective_seed = condition_seed + attempt * 104729
                params = sample_condition(config, profile, class_name, effective_seed)
                attenuation = config["adaptive_visibility_attenuation"]
                full_attempts = int(attenuation["full_strength_attempts"])
                if attempt < full_attempts:
                    strength_scale = 1.0
                else:
                    tail_count = max(1, max_attempts - full_attempts - 1)
                    tail_index = attempt - full_attempts
                    minimum_scale = float(attenuation["minimum_strength_scale"])
                    strength_scale = 1.0 - (1.0 - minimum_scale) * tail_index / tail_count
                params = attenuate_condition(
                    params,
                    strength_scale,
                    int(attenuation["reference_jpeg_quality"]),
                )
                conditioned_defect = apply_condition(parent_defect, params)
                conditioned_clean = apply_condition(parent_clean, params)
                defect_jpeg, payload = v2.jpeg_roundtrip(conditioned_defect, int(params["jpeg_quality"]))
                clean_jpeg, _ = v2.jpeg_roundtrip(conditioned_clean, int(params["jpeg_quality"]))
                metrics, visibility_failures = v2.evaluate_visibility(
                    defect_jpeg, clean_jpeg, semantic, class_name, severity, v2_config
                )
                luminance = luma_metrics(defect_jpeg, config)
                last_failures = visibility_failures + luma_failures(luminance, config)
                if not last_failures:
                    break
            else:
                raise RuntimeError(
                    f"condition generation failed {parent['sample_id']} {profile}: {last_failures}"
                )

            parent_suffix = parent["sample_id"].removeprefix("syn-v2-700-")
            sample_id = f"{config['sample_id_prefix']}-{parent_suffix}-{profile}"
            image_path = output / "images" / class_name / f"{sample_id}.jpg"
            mask_path = output / "masks" / class_name / f"{sample_id}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(payload)
            semantic.save(mask_path, format="PNG", optimize=False)
            image_rel = image_path.relative_to(ROOT).as_posix()
            mask_rel = mask_path.relative_to(ROOT).as_posix()
            parameter_record = {
                "profile": profile,
                "condition_seed": condition_seed,
                "effective_seed": effective_seed,
                "attempt": attempt,
                "condition": params,
            }
            row = {
                "sample_id": sample_id,
                "image_path": image_rel,
                "mask_path": mask_rel,
                "domain": config["source_domain"],
                "split": config["split"],
                "model_split": config["model_split"],
                "training_use": config["training_use"],
                "evaluation_eligible": config["evaluation_eligible"],
                "primary_class": class_name,
                "visible_multilabel": class_name,
                "severity": severity,
                "base_image_id": parent["base_image_id"],
                "source_release": source["release"],
                "parent_sample_id": parent["sample_id"],
                "parent_image_path": parent["image_path"],
                "parent_mask_path": parent["mask_path"],
                "parent_image_sha256": parent["image_sha256"],
                "parent_mask_sha256": parent["mask_sha256"],
                "parent_sample_seed": parent["sample_seed"],
                "base_group_id": parent["base_group_id"],
                "source_specimen_group": parent["source_specimen_group"],
                "view": parent["view"],
                "lineage_group_id": parent["sample_id"],
                "family_split_id": parent["sample_id"],
                "defect_instance_id": parent["sample_id"],
                "augmentation_family_id": f"condition-family-{parent['sample_id']}",
                "derivation_depth": "1",
                "variant_index": str(variant_index),
                "condition_profile": profile,
                "global_seed": str(config["global_seed"]),
                "sample_seed": str(condition_seed),
                "condition_seed": str(condition_seed),
                "attempt": str(attempt),
                "generator_version": config["generator_version"],
                "qc_gate_version": config["qc_gate_version"],
                "config_sha256": config_sha,
                "image_sha256": sha256_file(image_path),
                "mask_sha256": sha256_file(mask_path),
                "width": str(size),
                "height": str(size),
                "defect_pixels": str(defect_pixels),
                "bbox_x": str(bbox[0]),
                "bbox_y": str(bbox[1]),
                "bbox_w": str(bbox[2]),
                "bbox_h": str(bbox[3]),
                "qc_512_area": str(metrics["512"]["area"]),
                "qc_512_mean_abs_delta": f"{float(metrics['512']['mean_abs_delta']):.6f}",
                "qc_512_delta_e76_p50": f"{float(metrics['512']['delta_e76_p50']):.6f}",
                "qc_512_changed_fraction": f"{float(metrics['512']['changed_fraction']):.6f}",
                "qc_224_area": str(metrics["224"]["area"]),
                "qc_224_mean_abs_delta": f"{float(metrics['224']['mean_abs_delta']):.6f}",
                "qc_224_delta_e76_p50": f"{float(metrics['224']['delta_e76_p50']):.6f}",
                "qc_224_changed_fraction": f"{float(metrics['224']['changed_fraction']):.6f}",
                "qc_status": "AUTO_PASS_CONDITION_POST_JPEG_512_224",
                "human_verified": "NO",
                "qc_metrics_json": json.dumps(metrics, sort_keys=True, separators=(",", ":")),
                "luma_metrics_json": json.dumps(luminance, sort_keys=True, separators=(",", ":")),
                "parameters_json": json.dumps(parameter_record, sort_keys=True, separators=(",", ":")),
            }
            rows.append(row)
            instances.append(
                {
                    "sample_id": sample_id,
                    "primary_class": class_name,
                    "visible_multilabel": [class_name],
                    "semantic_mask_path": mask_rel,
                    "instances": [
                        {
                            "category": class_name,
                            "category_id": class_ids[class_name],
                            "area_px": defect_pixels,
                            "bbox_xywh": list(bbox),
                        }
                    ],
                }
            )
        if (parent_index + 1) % 24 == 0:
            print(f"generated parents={parent_index + 1}/{len(selected)} samples={len(rows)}", flush=True)

    manifest_path = output / "annotations" / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    instances_path = output / "annotations" / "instances.jsonl"
    with instances_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in instances:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    summary_path = output / "annotations" / "summary.csv"
    class_counts = Counter(row["primary_class"] for row in rows)
    profile_counts = Counter(row["condition_profile"] for row in rows)
    class_profile_counts = Counter(
        (row["primary_class"], row["condition_profile"]) for row in rows
    )
    severity_counts = Counter((row["primary_class"], row["severity"]) for row in rows)
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["axis", "class", "value", "count"])
        for class_name in classes:
            writer.writerow(["primary_class", class_name, class_name, class_counts[class_name]])
            for severity in ("mild", "moderate", "severe"):
                writer.writerow(["severity", class_name, severity, severity_counts[(class_name, severity)]])
        for class_name in classes:
            for profile in profiles:
                writer.writerow(
                    ["condition_profile", class_name, profile, class_profile_counts[(class_name, profile)]]
                )

    overview_sha, sheet_hashes = save_contact_sheets(output, rows, classes, profiles)
    release_path = output / "annotations" / "release.json"
    release = {
        "release": config["release"],
        "generator_version": config["generator_version"],
        "generator_script": "scripts/generate_synthetic_v3_conditions.py",
        "generator_script_sha256": sha256_file(Path(__file__)),
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": config_sha,
        "source_release": source["release"],
        "source_manifest_sha256": source["manifest_sha256"],
        "source_config_sha256": source["config_sha256"],
        "source_split_assignments_sha256": source["split_assignments_sha256"],
        "parent_count": len(selected),
        "variants_per_parent": len(profiles),
        "sample_count": len(rows),
        "class_counts": dict(class_counts),
        "profile_counts": dict(profile_counts),
        "class_profile_counts": {
            class_name: {
                profile: class_profile_counts[(class_name, profile)] for profile in profiles
            }
            for class_name in classes
        },
        "manifest_sha256": sha256_file(manifest_path),
        "instances_sha256": sha256_file(instances_path),
        "summary_sha256": sha256_file(summary_path),
        "overview_contact_sheet_sha256": overview_sha,
        "full_contact_sheet_sha256": sheet_hashes,
        "training_use": config["training_use"],
        "evaluation_eligible": config["evaluation_eligible"],
        "limitations": [
            "All variants derive only from the existing v2 gradient-train parents.",
            "Condition variants improve camera/illumination robustness but add no new physical specimen or defect morphology.",
            "Variants and their parents must remain in the same train family and are not valid evaluation samples.",
        ],
    }
    release_path.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"PASS generated={len(rows)} parents={len(selected)} profiles={len(profiles)} "
        "train_only=YES evaluation_eligible=NO"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
