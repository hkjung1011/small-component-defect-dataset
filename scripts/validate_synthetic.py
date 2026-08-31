from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "synthetic_v1.json"
DEFAULT_RELEASE = ROOT / "synthetic" / "v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, 0, 0
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def polygon_mask(size: int, points_normalized: list[list[float]]) -> np.ndarray:
    image = Image.new("L", (size, size), 0)
    points = [(round(x * size), round(y * size)) for x, y in points_normalized]
    ImageDraw.Draw(image).polygon(points, fill=255)
    return np.array(image, dtype=np.uint8) > 0


def ellipse_mask(size: int, box_normalized: list[float]) -> np.ndarray:
    image = Image.new("L", (size, size), 0)
    x0, y0, x1, y1 = box_normalized
    ImageDraw.Draw(image).ellipse(
        (round(x0 * size), round(y0 * size), round(x1 * size), round(y1 * size)),
        fill=255,
    )
    return np.array(image, dtype=np.uint8) > 0


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = radius * 2 + 1
    return np.array(
        Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(kernel))
    ) > 0


def build_allowed_rois(config: dict) -> dict[str, np.ndarray]:
    size = int(config["image_size"])
    raw = config["rois_normalized"]
    metal = polygon_mask(size, raw["metal_tab"])
    slot = ellipse_mask(size, raw["mount_slot"])
    metal &= ~slot
    body_full = polygon_mask(size, raw["body"])
    body_visible = body_full & ~metal & ~slot
    left = polygon_mask(size, raw["left_lead"])
    right = polygon_mask(size, raw["right_lead"])
    return {
        "scratch": metal,
        "surface_spot": metal,
        "discoloration": metal,
        "contamination": metal | body_visible,
        "lead_breakage": dilate(left | right, 12),
        "body_chip": dilate(body_visible, 10),
        "body_crack": dilate(body_visible, 10),
    }


def transform_roi(
    roi: np.ndarray,
    geometry: dict[str, float],
    size: int,
) -> np.ndarray:
    pad = 64
    padded = np.pad(roi.astype(np.uint8) * 255, ((pad, pad), (pad, pad)), mode="constant")
    rotated = Image.fromarray(padded, mode="L").rotate(
        float(geometry["rotation_deg"]),
        resample=Image.Resampling.NEAREST,
        expand=False,
        fillcolor=0,
    )
    scale = float(geometry["scale"])
    crop_size = round(size / scale)
    center = (size + pad * 2) / 2
    left = round(center + float(geometry["translate_x"]) - crop_size / 2)
    top = round(center + float(geometry["translate_y"]) - crop_size / 2)
    transformed = rotated.crop(
        (left, top, left + crop_size, top + crop_size)
    ).resize((size, size), Image.Resampling.NEAREST)
    return np.array(transformed, dtype=np.uint8) > 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate synthetic-v1 image, mask, label, hash, and split invariants.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    release = args.release.resolve()
    manifest_path = release / "annotations" / "manifest.csv"
    config_sha = sha256_file(args.config.resolve())
    base_path = ROOT / config["base"]["path"]
    base_sha = sha256_file(base_path)
    errors: list[str] = []
    if not manifest_path.exists():
        print(f"FAIL: missing manifest {manifest_path}")
        return 1
    with manifest_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    instances_path = release / "annotations" / "instances.jsonl"
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
                        errors.append(f"duplicate instances.jsonl sample_id {record_id}")
                    instance_by_id[record_id] = record
                except (KeyError, TypeError, json.JSONDecodeError) as error:
                    errors.append(f"invalid instances.jsonl line {line_number}: {error}")
    expected_count = len(config["primary_classes"]) * int(config["samples_per_primary_class"])
    if len(rows) != expected_count:
        errors.append(f"expected {expected_count} rows, got {len(rows)}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("duplicate sample_id")
    image_path_strings = [row["image_path"] for row in rows]
    mask_path_strings = [row["mask_path"] for row in rows]
    sample_seeds = [row["sample_seed"] for row in rows]
    if len(image_path_strings) != len(set(image_path_strings)):
        errors.append("duplicate image_path")
    if len(mask_path_strings) != len(set(mask_path_strings)):
        errors.append("duplicate mask_path")
    if len(sample_seeds) != len(set(sample_seeds)):
        errors.append("duplicate sample_seed")
    expected_class_counts = {name: int(config["samples_per_primary_class"]) for name in config["primary_classes"]}
    actual_class_counts = dict(Counter(row["primary_class"] for row in rows))
    if actual_class_counts != expected_class_counts:
        errors.append(f"unexpected class counts: {actual_class_counts}")
    allowed_mask_values = {int(value) for value in config["class_ids"].values()}
    id_to_name = {
        int(value): key
        for key, value in config["class_ids"].items()
        if key != "background"
    }
    allowed_rois = build_allowed_rois(config)
    image_root = (release / "images").resolve()
    mask_root = (release / "masks").resolve()
    expected_image_paths: set[Path] = set()
    expected_mask_paths: set[Path] = set()
    for row in rows:
        image_path = ROOT / row["image_path"]
        mask_path = ROOT / row["mask_path"]
        expected_image_paths.add(image_path.resolve())
        expected_mask_paths.add(mask_path.resolve())
        if image_root not in image_path.resolve().parents:
            errors.append(f"image path outside release {row['sample_id']}")
        if mask_root not in mask_path.resolve().parents:
            errors.append(f"mask path outside release {row['sample_id']}")
        if not image_path.exists() or not mask_path.exists():
            errors.append(f"missing pair for {row['sample_id']}")
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as opened_image:
                image = opened_image.convert("RGB")
            with Image.open(mask_path) as opened_mask:
                if opened_mask.mode != "L":
                    errors.append(
                        f"mask mode is not L {row['sample_id']}: {opened_mask.mode}"
                    )
                mask = opened_mask.convert("L")
        except Exception as error:
            errors.append(f"decode failure {row['sample_id']}: {error}")
            continue
        if image.size != mask.size or image.size != (int(row["width"]), int(row["height"])):
            errors.append(f"size mismatch {row['sample_id']}: image={image.size} mask={mask.size}")
        mask_array = np.array(mask, dtype=np.uint8)
        values = {int(value) for value in np.unique(mask_array)}
        if not values <= allowed_mask_values:
            errors.append(f"invalid mask values {row['sample_id']}: {sorted(values)}")
        present_ids = sorted(value for value in values if value != 0)
        present_labels = sorted(id_to_name[value] for value in present_ids if value in id_to_name)
        declared_labels = sorted(
            label for label in row["visible_multilabel"].split(";") if label
        )
        if present_labels != declared_labels:
            errors.append(
                f"mask/visible_multilabel mismatch {row['sample_id']}: "
                f"mask={present_labels} declared={declared_labels}"
            )
        instance_record = instance_by_id.get(row["sample_id"])
        if instance_record is None:
            errors.append(f"missing instances.jsonl record {row['sample_id']}")
        else:
            if instance_record.get("primary_class") != row["primary_class"]:
                errors.append(f"instances primary_class mismatch {row['sample_id']}")
            if sorted(instance_record.get("visible_multilabel", [])) != present_labels:
                errors.append(f"instances visible_multilabel mismatch {row['sample_id']}")
            if instance_record.get("semantic_mask_path") != row["mask_path"]:
                errors.append(f"instances mask path mismatch {row['sample_id']}")
            expected_instances = []
            for class_id in present_ids:
                class_mask = mask_array == class_id
                expected_instances.append(
                    {
                        "category": id_to_name[class_id],
                        "category_id": class_id,
                        "area_px": int(np.count_nonzero(class_mask)),
                        "bbox_xywh": list(bbox(class_mask)),
                    }
                )
            if instance_record.get("instances") != expected_instances:
                errors.append(f"instances geometry mismatch {row['sample_id']}")
        defect_pixels = int(np.count_nonzero(mask_array))
        if defect_pixels != int(row["defect_pixels"]):
            errors.append(f"defect pixel mismatch {row['sample_id']}")
        is_normal = row["primary_class"] == "normal_proxy"
        if is_normal and defect_pixels != 0:
            errors.append(f"normal proxy has non-empty mask {row['sample_id']}")
        if not is_normal and defect_pixels == 0:
            errors.append(f"defect sample has empty mask {row['sample_id']}")
        if row["primary_class"] == "multi_defect" and len(present_labels) < 2:
            errors.append(f"multi_defect has fewer than 2 visible labels {row['sample_id']}")
        if row["primary_class"] not in {"normal_proxy", "multi_defect"} and present_labels != [row["primary_class"]]:
            errors.append(
                f"primary class mismatch {row['sample_id']}: {present_labels}"
            )
        try:
            parameters = json.loads(row["parameters_json"])
            geometry = parameters["geometry"]
            for class_id in present_ids:
                class_name = id_to_name[class_id]
                transformed_roi = transform_roi(
                    allowed_rois[class_name], geometry, int(config["image_size"])
                )
                class_mask = mask_array == class_id
                independent_inside_ratio = float(
                    np.count_nonzero(class_mask & transformed_roi)
                    / np.count_nonzero(class_mask)
                )
                if independent_inside_ratio < 0.999:
                    errors.append(
                        f"independent ROI containment below gate {row['sample_id']} "
                        f"{class_name}: {independent_inside_ratio:.10f}"
                    )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"invalid parameters_json {row['sample_id']}: {error}")
        expected_bbox = tuple(int(row[key]) for key in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"))
        if bbox(mask_array) != expected_bbox:
            errors.append(f"bbox mismatch {row['sample_id']}")
        if sha256_file(image_path) != row["image_sha256"]:
            errors.append(f"image SHA mismatch {row['sample_id']}")
        if sha256_file(mask_path) != row["mask_sha256"]:
            errors.append(f"mask SHA mismatch {row['sample_id']}")
        if row["split"] != "train" or row["evaluation_eligible"] != "NO":
            errors.append(f"non-train or evaluation-eligible synthetic row {row['sample_id']}")
        if row["domain"] != "synthetic_restored":
            errors.append(f"unexpected domain {row['sample_id']}: {row['domain']}")
        if row["config_sha256"] != config_sha:
            errors.append(f"config SHA mismatch {row['sample_id']}")
        if row["base_sha256"] != base_sha:
            errors.append(f"base SHA mismatch {row['sample_id']}")
        if row["generator_version"] != config["generator_version"]:
            errors.append(f"generator version mismatch {row['sample_id']}")
        if row["global_seed"] != str(config["global_seed"]):
            errors.append(f"global seed mismatch {row['sample_id']}")
        if float(row["roi_inside_ratio"]) < 0.999:
            errors.append(f"ROI containment below gate {row['sample_id']}: {row['roi_inside_ratio']}")
        minimum_delta = float(
            config["minimum_pre_transform_mean_abs_delta"][row["primary_class"]]
        )
        if float(row["pre_transform_mean_abs_delta"]) < minimum_delta:
            errors.append(
                f"visible delta below gate {row['sample_id']}: "
                f"{row['pre_transform_mean_abs_delta']} < {minimum_delta}"
            )

    actual_images = {path.resolve() for path in (release / "images").rglob("*.jpg")}
    actual_masks = {path.resolve() for path in (release / "masks").rglob("*.png")}
    if actual_images != expected_image_paths:
        errors.append(
            f"image orphans/missing: extra={len(actual_images-expected_image_paths)} missing={len(expected_image_paths-actual_images)}"
        )
    if actual_masks != expected_mask_paths:
        errors.append(
            f"mask orphans/missing: extra={len(actual_masks-expected_mask_paths)} missing={len(expected_mask_paths-actual_masks)}"
        )
    if set(instance_by_id) != set(sample_ids):
        errors.append(
            "instances.jsonl IDs differ from manifest: "
            f"extra={len(set(instance_by_id)-set(sample_ids))} "
            f"missing={len(set(sample_ids)-set(instance_by_id))}"
        )

    qa_relative = Path(
        config.get(
            "human_qa_path",
            "annotations/synthetic_v1_human_qa.csv",
        )
    )
    qa_path = (ROOT / qa_relative).resolve()
    if ROOT not in qa_path.parents:
        errors.append(f"human QA path outside repository: {qa_path}")
    if not qa_path.exists():
        errors.append(f"missing human QA file: {qa_relative.as_posix()}")
    else:
        with qa_path.open(encoding="utf-8", newline="") as stream:
            qa_rows = list(csv.DictReader(stream))
        qa_ids = [row["sample_id"] for row in qa_rows]
        if len(qa_ids) != len(set(qa_ids)):
            errors.append("duplicate human QA sample_id")
        manifest_class_by_id = {
            row["sample_id"]: row["primary_class"] for row in rows
        }
        qa_class_counts = Counter(
            manifest_class_by_id.get(row["sample_id"], "__unknown__")
            for row in qa_rows
        )
        missing_qa_coverage = {
            class_name: qa_class_counts.get(class_name, 0)
            for class_name in config["primary_classes"]
            if qa_class_counts.get(class_name, 0) < 3
        }
        if missing_qa_coverage:
            errors.append(
                "human QA requires at least 3 samples per primary class: "
                f"{missing_qa_coverage}"
            )
        unknown_qa_ids = set(qa_ids) - set(sample_ids)
        if unknown_qa_ids:
            errors.append(f"human QA references unknown samples: {sorted(unknown_qa_ids)}")
        if any(row["reviewer"] != "OpenAI Codex" for row in qa_rows):
            errors.append("unexpected human QA reviewer")
        if any(row["status"] != "PASS_POC_VISUAL" for row in qa_rows):
            errors.append("unexpected human QA status")

    release_json = release / "annotations" / "release.json"
    if not release_json.exists():
        errors.append("missing release.json")
    else:
        metadata = json.loads(release_json.read_text(encoding="utf-8"))
        if metadata.get("release") != config["release"]:
            errors.append("release.json release name mismatch")
        if int(metadata.get("sample_count", -1)) != len(rows):
            errors.append("release.json sample_count mismatch")
        if metadata.get("source_domain") != "synthetic_restored":
            errors.append("release.json source_domain mismatch")
        if metadata.get("split") != "train":
            errors.append("release.json split mismatch")
        if metadata.get("generator_version") != config["generator_version"]:
            errors.append("release.json generator version mismatch")
        if metadata.get("class_counts") != expected_class_counts:
            errors.append("release.json class_counts mismatch")
        expected_config_path = args.config.resolve().relative_to(ROOT).as_posix()
        if metadata.get("config_path") != expected_config_path:
            errors.append("release.json config path mismatch")
        expected_base_path = base_path.resolve().relative_to(ROOT).as_posix()
        if metadata.get("base_path") != expected_base_path:
            errors.append("release.json base path mismatch")
        generator_value = metadata.get("generator_script")
        generator_target: Path | None = None
        if not isinstance(generator_value, str):
            errors.append("release.json missing generator_script")
        else:
            generator_target = (ROOT / generator_value).resolve()
            if ROOT not in generator_target.parents:
                errors.append("release.json generator_script outside repository")
                generator_target = None
        hash_targets = {
            "config_sha256": args.config.resolve(),
            "base_sha256": base_path,
            "manifest_sha256": manifest_path,
            "instances_sha256": instances_path,
            "summary_sha256": release / "annotations" / "summary.csv",
        }
        if generator_target is not None:
            hash_targets["generator_script_sha256"] = generator_target
        for field, target in hash_targets.items():
            if not target.exists() or metadata.get(field) != sha256_file(target):
                errors.append(f"release hash mismatch: {field}")

    current_ids = set(sample_ids)
    current_seeds = set(sample_seeds)
    current_image_hashes = {row["image_sha256"] for row in rows}
    for other_manifest in (ROOT / "synthetic").glob("*/annotations/manifest.csv"):
        if other_manifest.resolve() == manifest_path.resolve():
            continue
        with other_manifest.open(encoding="utf-8", newline="") as stream:
            other_rows = list(csv.DictReader(stream))
        other_fields = set(other_rows[0]) if other_rows else set()
        duplicate_ids = (
            current_ids & {row["sample_id"] for row in other_rows}
            if "sample_id" in other_fields
            else set()
        )
        duplicate_seeds = (
            current_seeds & {row["sample_seed"] for row in other_rows}
            if "sample_seed" in other_fields
            else set()
        )
        duplicate_images = (
            current_image_hashes & {row["image_sha256"] for row in other_rows}
            if "image_sha256" in other_fields
            else set()
        )
        other_release = other_manifest.parents[1].name
        if duplicate_ids:
            errors.append(
                f"cross-release sample_id duplicates with {other_release}: "
                f"{len(duplicate_ids)}"
            )
        if duplicate_seeds:
            errors.append(
                f"cross-release sample_seed duplicates with {other_release}: "
                f"{len(duplicate_seeds)}"
            )
        if duplicate_images:
            errors.append(
                f"cross-release image SHA duplicates with {other_release}: "
                f"{len(duplicate_images)}"
            )

    if errors:
        print("FAIL")
        for error in errors[:100]:
            print(f"- {error}")
        if len(errors) > 100:
            print(f"- ... {len(errors)-100} additional errors")
        return 1
    print(
        f"PASS: synthetic={len(rows)}, classes={len(actual_class_counts)}, "
        f"images={len(actual_images)}, masks={len(actual_masks)}, train_only=YES"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
