from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import generate_synthetic_v1_450 as legacy
import generate_synthetic_v2_700 as generator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "synthetic_v2_700.json"
DEFAULT_RELEASE = ROOT / "synthetic" / "v2_700"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate synthetic-v2-700 integrity and paired-clean post-JPEG 512/224 visibility."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def compare_metric_records(
    sample_id: str,
    recorded: dict[str, dict[str, float | int]],
    actual: dict[str, dict[str, float | int]],
    errors: list[str],
) -> None:
    for resolution in ("512", "224"):
        for key, actual_value in actual[resolution].items():
            try:
                recorded_value = recorded[resolution][key]
                if isinstance(actual_value, int):
                    matched = int(recorded_value) == actual_value
                else:
                    matched = math.isclose(float(recorded_value), float(actual_value), abs_tol=1e-6)
                if not matched:
                    errors.append(
                        f"QC metric mismatch {sample_id} {resolution}.{key}: "
                        f"recorded={recorded_value} actual={actual_value}"
                    )
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"invalid QC metric {sample_id} {resolution}.{key}: {error}")


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    release = args.release.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = release / "annotations" / "manifest.csv"
    instances_path = release / "annotations" / "instances.jsonl"
    release_path = release / "annotations" / "release.json"
    summary_path = release / "annotations" / "summary.csv"
    errors: list[str] = []

    if not manifest_path.exists():
        print(f"FAIL: missing manifest {manifest_path}")
        return 1
    rows = read_csv(manifest_path)
    expected_count = len(config["primary_classes"]) * int(config["samples_per_primary_class"])
    if len(rows) != expected_count:
        errors.append(f"row count {len(rows)} != {expected_count}")
    required_columns = {
        "sample_id", "image_path", "mask_path", "primary_class", "visible_multilabel",
        "severity", "sample_seed", "config_sha256", "base_sha256", "image_sha256",
        "mask_sha256", "width", "height", "defect_pixels", "bbox_x", "bbox_y",
        "bbox_w", "bbox_h", "parameters_json", "qc_metrics_json", "qc_status",
        "qc_gate_version", "training_use", "evaluation_eligible",
    }
    missing_columns = required_columns - set(rows[0] if rows else {})
    if missing_columns:
        errors.append(f"missing manifest columns: {sorted(missing_columns)}")

    ids = [row["sample_id"] for row in rows]
    seeds = [row["sample_seed"] for row in rows]
    image_strings = [row["image_path"] for row in rows]
    mask_strings = [row["mask_path"] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate sample_id")
    if len(seeds) != len(set(seeds)):
        errors.append("duplicate sample_seed")
    if len(image_strings) != len(set(image_strings)):
        errors.append("duplicate image_path")
    if len(mask_strings) != len(set(mask_strings)):
        errors.append("duplicate mask_path")

    per_class = int(config["samples_per_primary_class"])
    class_counts = Counter(row["primary_class"] for row in rows)
    expected_class_counts = {class_name: per_class for class_name in config["primary_classes"]}
    if dict(class_counts) != expected_class_counts:
        errors.append(f"class counts mismatch: {dict(class_counts)}")
    for class_name in config["primary_classes"]:
        observed = Counter(
            row["severity"] for row in rows if row["primary_class"] == class_name
        )
        expected = {key: int(value) for key, value in config["severity_quotas"].items()}
        if dict(observed) != expected:
            errors.append(f"severity quota mismatch {class_name}: {dict(observed)}")

    instance_by_id: dict[str, dict] = {}
    if not instances_path.exists():
        errors.append("missing instances.jsonl")
    else:
        with instances_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    record = json.loads(line)
                    record_id = record["sample_id"]
                    if record_id in instance_by_id:
                        errors.append(f"duplicate instance ID {record_id}")
                    instance_by_id[record_id] = record
                except (KeyError, TypeError, json.JSONDecodeError) as error:
                    errors.append(f"invalid instances line {line_number}: {error}")

    config_sha = generator.sha256_file(config_path)
    base_path = ROOT / config["base"]["path"]
    base_sha = generator.sha256_file(base_path)
    base = Image.open(base_path).convert("RGB")
    size = int(config["image_size"])
    rois = legacy.build_rois(config)
    class_ids = {key: int(value) for key, value in config["class_ids"].items()}
    allowed_mask_values = set(class_ids.values())
    expected_images: set[Path] = set()
    expected_masks: set[Path] = set()
    metrics_by_class: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for row in rows:
        sample_id = row["sample_id"]
        class_name = row["primary_class"]
        severity = row["severity"]
        image_path = (ROOT / row["image_path"]).resolve()
        mask_path = (ROOT / row["mask_path"]).resolve()
        expected_images.add(image_path)
        expected_masks.add(mask_path)
        if (release / "images").resolve() not in image_path.parents:
            errors.append(f"image path outside release {sample_id}")
        if (release / "masks").resolve() not in mask_path.parents:
            errors.append(f"mask path outside release {sample_id}")
        if not image_path.exists() or not mask_path.exists():
            errors.append(f"missing image/mask pair {sample_id}")
            continue
        try:
            with Image.open(image_path) as opened:
                if opened.format != "JPEG":
                    errors.append(f"image is not JPEG {sample_id}: {opened.format}")
                if opened.size != (size, size):
                    errors.append(f"image size mismatch {sample_id}: {opened.size}")
                defect = opened.convert("RGB")
                defect.load()
            with Image.open(mask_path) as opened:
                if opened.format != "PNG" or opened.mode != "L":
                    errors.append(f"mask format/mode mismatch {sample_id}: {opened.format}/{opened.mode}")
                if opened.size != (size, size):
                    errors.append(f"mask size mismatch {sample_id}: {opened.size}")
                semantic = opened.convert("L")
                semantic.load()
        except Exception as error:
            errors.append(f"decode failure {sample_id}: {error}")
            continue

        mask_array = np.array(semantic, dtype=np.uint8)
        values = {int(value) for value in np.unique(mask_array)}
        if not values <= allowed_mask_values:
            errors.append(f"invalid mask values {sample_id}: {sorted(values)}")
        expected_id = class_ids.get(class_name)
        if expected_id is None or values - {0} != {expected_id}:
            errors.append(f"mask class mismatch {sample_id}: {sorted(values)}")
            continue
        class_mask = mask_array == expected_id
        area = int(np.count_nonzero(class_mask))
        bbox = legacy.mask_bbox(class_mask)
        if area != int(row["defect_pixels"]):
            errors.append(f"defect area mismatch {sample_id}")
        recorded_bbox = tuple(int(row[key]) for key in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"))
        if bbox != recorded_bbox:
            errors.append(f"bbox mismatch {sample_id}")
        if row["visible_multilabel"] != class_name:
            errors.append(f"visible label mismatch {sample_id}")

        record = instance_by_id.get(sample_id)
        expected_instance = {
            "sample_id": sample_id,
            "primary_class": class_name,
            "visible_multilabel": [class_name],
            "semantic_mask_path": row["mask_path"],
            "instances": [{
                "category": class_name,
                "category_id": expected_id,
                "area_px": area,
                "bbox_xywh": list(bbox),
            }],
        }
        if record != expected_instance:
            errors.append(f"instances mismatch {sample_id}")

        if generator.sha256_file(image_path) != row["image_sha256"]:
            errors.append(f"image SHA mismatch {sample_id}")
        if generator.sha256_file(mask_path) != row["mask_sha256"]:
            errors.append(f"mask SHA mismatch {sample_id}")
        if row["config_sha256"] != config_sha:
            errors.append(f"config SHA mismatch {sample_id}")
        if row["base_sha256"] != base_sha:
            errors.append(f"base SHA mismatch {sample_id}")
        if row["generator_version"] != config["generator_version"]:
            errors.append(f"generator version mismatch {sample_id}")
        if row["qc_gate_version"] != config["qc_gate_version"]:
            errors.append(f"QC gate version mismatch {sample_id}")
        if row["qc_status"] != "AUTO_PASS_POST_JPEG_512_224":
            errors.append(f"QC status mismatch {sample_id}")
        if row["training_use"] != "TRAIN_ONLY_SYNTHETIC_NG":
            errors.append(f"training use mismatch {sample_id}")
        if row["split"] != "train" or row["evaluation_eligible"] != "NO":
            errors.append(f"synthetic split/evaluation mismatch {sample_id}")
        if float(row["roi_inside_ratio"]) < float(config["qc"]["roi_inside_ratio"]):
            errors.append(f"pre-transform ROI gate failed {sample_id}")

        try:
            parameters = json.loads(row["parameters_json"])
            geometry = parameters["geometry"]
            photometric = parameters["photometric"]
            clean, _ = generator.apply_geometry(
                base, Image.new("L", base.size, 0), geometry, size
            )
            clean = generator.apply_photometric(clean, photometric)
            clean_jpeg, _ = generator.jpeg_roundtrip(clean, int(config["jpeg_quality"]))
            actual_metrics, gate_failures = generator.evaluate_visibility(
                defect, clean_jpeg, semantic, class_name, severity, config
            )
            if gate_failures:
                errors.append(f"visibility gate failed {sample_id}: {'; '.join(gate_failures)}")
            recorded_metrics = json.loads(row["qc_metrics_json"])
            compare_metric_records(sample_id, recorded_metrics, actual_metrics, errors)
            for resolution in ("512", "224"):
                for key in ("area", "mean_abs_delta", "delta_e76_p50", "changed_fraction"):
                    metrics_by_class[class_name][f"{resolution}_{key}"].append(
                        float(actual_metrics[resolution][key])
                    )

            allowed_pre = legacy.recipe_allowed_roi(class_name, rois)
            _, transformed_allowed = generator.apply_geometry(
                Image.new("RGB", base.size),
                Image.fromarray(allowed_pre.astype(np.uint8) * 255, mode="L"),
                geometry,
                size,
            )
            allowed_array = np.array(transformed_allowed, dtype=np.uint8) > 0
            inside_ratio = float(np.count_nonzero(class_mask & allowed_array) / area)
            if inside_ratio < float(config["qc"]["roi_inside_ratio"]):
                errors.append(f"post-transform ROI gate failed {sample_id}: {inside_ratio:.6f}")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"parameters/QC replay failure {sample_id}: {error}")

    actual_images = {path.resolve() for path in (release / "images").rglob("*") if path.is_file()}
    actual_masks = {path.resolve() for path in (release / "masks").rglob("*") if path.is_file()}
    if actual_images != expected_images:
        errors.append(
            f"image orphan/missing: extra={len(actual_images - expected_images)} "
            f"missing={len(expected_images - actual_images)}"
        )
    if actual_masks != expected_masks:
        errors.append(
            f"mask orphan/missing: extra={len(actual_masks - expected_masks)} "
            f"missing={len(expected_masks - actual_masks)}"
        )
    if set(instance_by_id) != set(ids):
        errors.append("instances IDs differ from manifest")

    current_ids = set(ids)
    current_seeds = set(seeds)
    current_hashes = {row["image_sha256"] for row in rows}
    for other_manifest in (ROOT / "synthetic").glob("*/annotations/manifest.csv"):
        if other_manifest.resolve() == manifest_path.resolve():
            continue
        other_rows = read_csv(other_manifest)
        release_name = other_manifest.parents[1].name
        if current_ids & {row["sample_id"] for row in other_rows}:
            errors.append(f"cross-release sample ID duplicate with {release_name}")
        if current_seeds & {row["sample_seed"] for row in other_rows}:
            errors.append(f"cross-release seed duplicate with {release_name}")
        if current_hashes & {row["image_sha256"] for row in other_rows}:
            errors.append(f"cross-release image duplicate with {release_name}")

    if not release_path.exists():
        errors.append("missing release.json")
    else:
        metadata = json.loads(release_path.read_text(encoding="utf-8"))
        if metadata.get("release") != config["release"]:
            errors.append("release name mismatch")
        if metadata.get("sample_count") != expected_count:
            errors.append("release sample_count mismatch")
        if metadata.get("class_counts") != expected_class_counts:
            errors.append("release class_counts mismatch")
        expected_severity = {
            class_name: {key: int(value) for key, value in config["severity_quotas"].items()}
            for class_name in config["primary_classes"]
        }
        if metadata.get("severity_counts") != expected_severity:
            errors.append("release severity_counts mismatch")
        generator_path = ROOT / metadata.get("generator_script", "")
        hash_targets = {
            "generator_script_sha256": generator_path,
            "config_sha256": config_path,
            "base_sha256": base_path,
            "manifest_sha256": manifest_path,
            "instances_sha256": instances_path,
            "summary_sha256": summary_path,
        }
        for field, target in hash_targets.items():
            if not target.exists() or metadata.get(field) != generator.sha256_file(target):
                errors.append(f"release hash mismatch {field}")
        overview_path = release / "contact_sheet.jpg"
        if (
            not overview_path.exists()
            or metadata.get("overview_contact_sheet_sha256") != generator.sha256_file(overview_path)
        ):
            errors.append("release hash mismatch overview_contact_sheet_sha256")
        full_sheets = metadata.get("full_contact_sheet_sha256", {})
        if not isinstance(full_sheets, dict) or len(full_sheets) != len(config["primary_classes"]) * 2:
            errors.append("full contact sheet inventory mismatch")
        else:
            for relative_path, expected_hash in full_sheets.items():
                target = (ROOT / relative_path).resolve()
                if release not in target.parents or not target.exists():
                    errors.append(f"invalid full contact sheet path {relative_path}")
                elif generator.sha256_file(target) != expected_hash:
                    errors.append(f"full contact sheet SHA mismatch {relative_path}")

    if errors:
        print(f"FAIL: errors={len(errors)}")
        for error in errors[:120]:
            print(f"- {error}")
        if len(errors) > 120:
            print(f"- ... {len(errors) - 120} additional errors")
        return 1

    print(
        f"PASS: synthetic={len(rows)}, classes={len(class_counts)}, "
        "post_jpeg_512_224=PASS, train_only=YES"
    )
    for class_name in config["primary_classes"]:
        values = metrics_by_class[class_name]
        print(
            f"{class_name}: "
            f"A512={min(values['512_area']):.0f}..{max(values['512_area']):.0f}, "
            f"A224={min(values['224_area']):.0f}..{max(values['224_area']):.0f}, "
            f"MAD512_min={min(values['512_mean_abs_delta']):.3f}, "
            f"MAD224_min={min(values['224_mean_abs_delta']):.3f}, "
            f"dE76p50_512_min={min(values['512_delta_e76_p50']):.3f}, "
            f"dE76p50_224_min={min(values['224_delta_e76_p50']):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
