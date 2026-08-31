from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

import generate_synthetic_v1_450 as legacy
import generate_synthetic_v2_700 as v2_generator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "synthetic_v3_conditions.json"
DEFAULT_RELEASE = ROOT / "synthetic" / "v3_conditions"
EXPECTED_QC_STATUS = "AUTO_PASS_CONDITION_POST_JPEG_512_224"
FLOAT_TOLERANCE = 1e-6


class ValidationSetupError(RuntimeError):
    """Raised when pinned source material cannot be trusted or loaded."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independently validate the synthetic-v3 condition augmentation release, "
            "including pinned lineage, deterministic replay, and post-JPEG QC."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return v2_generator.sha256_file(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_repository_path(value: str, field: str) -> Path:
    if not value or Path(value).is_absolute():
        raise ValidationSetupError(f"{field} must be a non-empty repository-relative path")
    candidate = (ROOT / Path(value)).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValidationSetupError(f"{field} escapes repository root: {value}") from error
    return candidate


def require_pinned_file(source: dict[str, Any], path_key: str, sha_key: str) -> Path:
    raw_path = source.get(path_key)
    expected_sha = source.get(sha_key)
    if not isinstance(raw_path, str):
        raise ValidationSetupError(f"source.{path_key} must be a string")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValidationSetupError(f"source.{sha_key} must be a SHA-256 hex string")
    path = resolve_repository_path(raw_path, f"source.{path_key}")
    if not path.is_file():
        raise ValidationSetupError(f"missing pinned source file: {path}")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha.lower():
        raise ValidationSetupError(
            f"source pin mismatch {path_key}: expected={expected_sha.lower()} actual={actual_sha}"
        )
    return path


def integer(value: Any, field: str) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} is not an integer: {value!r}") from error


def finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} is not numeric: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} is not finite: {value!r}")
    return result


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    return legacy.mask_bbox(mask)


def compare_nested_metrics(
    sample_id: str,
    recorded: dict[str, Any],
    actual: dict[str, dict[str, float | int]],
    errors: list[str],
) -> None:
    if "visibility" in recorded and isinstance(recorded["visibility"], dict):
        recorded = recorded["visibility"]
    for resolution in ("512", "224"):
        block = recorded.get(resolution)
        if not isinstance(block, dict):
            errors.append(f"missing QC metrics {sample_id} resolution={resolution}")
            continue
        for key, actual_value in actual[resolution].items():
            if key not in block:
                errors.append(f"missing QC metric {sample_id} {resolution}.{key}")
                continue
            try:
                if isinstance(actual_value, int):
                    matched = integer(block[key], key) == actual_value
                else:
                    matched = math.isclose(
                        finite_float(block[key], key),
                        float(actual_value),
                        rel_tol=0.0,
                        abs_tol=FLOAT_TOLERANCE,
                    )
            except ValueError as error:
                errors.append(f"invalid QC metric {sample_id} {resolution}.{key}: {error}")
                continue
            if not matched:
                errors.append(
                    f"QC metric mismatch {sample_id} {resolution}.{key}: "
                    f"recorded={block[key]} actual={actual_value}"
                )


def calculate_luma_metrics(image: Image.Image, qc: dict[str, Any]) -> dict[str, float]:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    luma = (
        array[..., 0] * np.float32(0.2126)
        + array[..., 1] * np.float32(0.7152)
        + array[..., 2] * np.float32(0.0722)
    )
    black_value = finite_float(qc["black_clip_value"], "qc.black_clip_value")
    white_value = finite_float(qc["white_clip_value"], "qc.white_clip_value")
    return {
        "mean_luma": round(float(np.mean(luma)), 6),
        "luma_std": round(float(np.std(luma)), 6),
        "black_clip_fraction": round(float(np.mean(luma <= black_value)), 6),
        "white_clip_fraction": round(float(np.mean(luma >= white_value)), 6),
    }


def validate_luma(
    sample_id: str,
    image: Image.Image,
    raw_record: str,
    qc: dict[str, Any],
    errors: list[str],
) -> dict[str, float]:
    actual = calculate_luma_metrics(image, qc)
    try:
        recorded = json.loads(raw_record)
        if not isinstance(recorded, dict):
            raise TypeError("luma_metrics_json must decode to an object")
    except (json.JSONDecodeError, TypeError) as error:
        errors.append(f"invalid luma_metrics_json {sample_id}: {error}")
        recorded = {}
    for key, actual_value in actual.items():
        if key not in recorded:
            errors.append(f"missing luma metric {sample_id}: {key}")
            continue
        try:
            recorded_value = finite_float(recorded[key], key)
        except ValueError as error:
            errors.append(f"invalid luma metric {sample_id} {key}: {error}")
            continue
        if not math.isclose(recorded_value, actual_value, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE):
            errors.append(
                f"luma metric mismatch {sample_id} {key}: "
                f"recorded={recorded_value} actual={actual_value}"
            )

    gates = {
        "mean_luma": ("min_mean_luma", ">="),
        "luma_std": ("min_luma_std", ">="),
        "black_clip_fraction": ("max_black_clip_fraction", "<="),
        "white_clip_fraction": ("max_white_clip_fraction", "<="),
    }
    for metric, (gate_name, direction) in gates.items():
        threshold = finite_float(qc[gate_name], f"qc.{gate_name}")
        value = actual[metric]
        passed = value >= threshold if direction == ">=" else value <= threshold
        if not passed:
            errors.append(
                f"luma gate failed {sample_id}: {metric}={value:.6f} "
                f"must be {direction}{threshold:.6f}"
            )
    max_mean = finite_float(qc["max_mean_luma"], "qc.max_mean_luma")
    if actual["mean_luma"] > max_mean:
        errors.append(
            f"luma gate failed {sample_id}: mean_luma={actual['mean_luma']:.6f} "
            f"must be <={max_mean:.6f}"
        )
    return actual


def validate_condition_parameters(
    row: dict[str, str],
    config: dict[str, Any],
    errors: list[str],
) -> dict[str, Any] | None:
    sample_id = row["sample_id"]
    try:
        record = json.loads(row["parameters_json"])
        if not isinstance(record, dict):
            raise TypeError("parameters_json must decode to an object")
    except (json.JSONDecodeError, TypeError) as error:
        errors.append(f"invalid parameters_json {sample_id}: {error}")
        return None

    values = record
    for key in ("condition", "parameters", "sampled"):
        if isinstance(record.get(key), dict):
            values = record[key]
            break

    profile = row["condition_profile"]
    recorded_profile = record.get("condition_profile", record.get("profile"))
    if recorded_profile != profile:
        errors.append(
            f"parameter profile mismatch {sample_id}: row={profile!r} json={recorded_profile!r}"
        )
    try:
        if integer(record.get("condition_seed"), "condition_seed") != integer(
            row["condition_seed"], "condition_seed"
        ):
            errors.append(f"parameter condition_seed mismatch {sample_id}")
        if integer(record.get("attempt"), "attempt") != integer(row["attempt"], "attempt"):
            errors.append(f"parameter attempt mismatch {sample_id}")
    except ValueError as error:
        errors.append(f"invalid parameter lineage {sample_id}: {error}")

    ranges = config["profile_ranges"].get(profile)
    if not isinstance(ranges, dict):
        errors.append(f"profile has no configured parameter ranges {sample_id}: {profile}")
        return record
    override = config.get("class_overrides", {}).get(row["primary_class"], {})
    if not isinstance(override, dict):
        override = {}
    for name, bounds in ranges.items():
        if name not in values:
            errors.append(f"missing condition parameter {sample_id}: {name}")
            continue
        if not isinstance(bounds, list) or len(bounds) != 2:
            errors.append(f"invalid configured range {profile}.{name}")
            continue
        try:
            actual = finite_float(values[name], f"{profile}.{name}")
            lower = finite_float(bounds[0], f"{profile}.{name}.min")
            upper = finite_float(bounds[1], f"{profile}.{name}.max")
        except ValueError as error:
            errors.append(f"invalid condition parameter {sample_id}: {error}")
            continue
        effective_lower, effective_upper = lower, upper
        if name in {"channel_gain_r", "channel_gain_g", "channel_gain_b"}:
            floor = float(override.get("channel_gain_floor", -math.inf))
            ceiling = float(override.get("channel_gain_ceiling", math.inf))
            effective_lower = min(ceiling, max(floor, lower))
            effective_upper = min(ceiling, max(floor, upper))
        elif name == "hotspot_strength" and "hotspot_strength_ceiling" in override:
            ceiling = float(override["hotspot_strength_ceiling"])
            effective_lower = min(ceiling, lower)
            effective_upper = min(ceiling, upper)
        elif name == "exposure_ev" and "exposure_ev_floor" in override:
            floor = float(override["exposure_ev_floor"])
            effective_lower = max(floor, lower)
            effective_upper = max(floor, upper)
        elif name == "gamma" and "gamma_ceiling" in override:
            ceiling = float(override["gamma_ceiling"])
            effective_lower = min(ceiling, lower)
            effective_upper = min(ceiling, upper)
        elif name == "contrast" and "contrast_floor" in override:
            floor = float(override["contrast_floor"])
            effective_lower = max(floor, lower)
            effective_upper = max(floor, upper)
        elif name == "blur_radius" and "blur_radius_ceiling" in override:
            ceiling = float(override["blur_radius_ceiling"])
            effective_lower = min(ceiling, lower)
            effective_upper = min(ceiling, upper)
        elif name == "noise_sigma" and "noise_sigma_ceiling" in override:
            ceiling = float(override["noise_sigma_ceiling"])
            effective_lower = min(ceiling, lower)
            effective_upper = min(ceiling, upper)
        elif name == "jpeg_quality" and "jpeg_quality_floor" in override:
            floor = float(override["jpeg_quality_floor"])
            effective_lower = max(floor, lower)
            effective_upper = max(floor, upper)
        # The generator may deterministically attenuate a hard condition toward
        # the verified v2 parent after failed visibility attempts.  Validate the
        # stored, post-attenuation value against the hull from the sampled range
        # to its neutral/reference value; exact replay below still proves the
        # particular value and scale.
        if "condition_strength_scale" in values:
            neutral_by_name = {
                "exposure_ev": 0.0,
                "gamma": 1.0,
                "contrast": 1.0,
                "channel_gain_r": 1.0,
                "channel_gain_g": 1.0,
                "channel_gain_b": 1.0,
                "gradient_strength": 0.0,
                "shadow_strength": 0.0,
                "vignette_strength": 0.0,
                "hotspot_strength": 0.0,
                "blur_radius": 0.0,
                "noise_sigma": 0.0,
                "jpeg_quality": float(
                    config.get("adaptive_visibility_attenuation", {}).get(
                        "reference_jpeg_quality", 95
                    )
                ),
            }
            if name in neutral_by_name:
                neutral = neutral_by_name[name]
                effective_lower = min(effective_lower, neutral)
                effective_upper = max(effective_upper, neutral)
        if not min(effective_lower, effective_upper) <= actual <= max(
            effective_lower, effective_upper
        ):
            errors.append(
                f"condition parameter outside range {sample_id} {name}: "
                f"{actual} not in effective [{effective_lower}, {effective_upper}]"
            )
        if name == "jpeg_quality" and not float(actual).is_integer():
            errors.append(f"jpeg_quality must be an integer {sample_id}: {actual}")

    if override:
        gain_floor = override.get("channel_gain_floor")
        gain_ceiling = override.get("channel_gain_ceiling")
        for channel in ("channel_gain_r", "channel_gain_g", "channel_gain_b"):
            if channel not in values:
                continue
            try:
                gain = finite_float(values[channel], channel)
                if gain_floor is not None and gain < float(gain_floor):
                    errors.append(f"class override failed {sample_id}: {channel}={gain}")
                if gain_ceiling is not None and gain > float(gain_ceiling):
                    errors.append(f"class override failed {sample_id}: {channel}={gain}")
            except ValueError as error:
                errors.append(f"invalid class override parameter {sample_id}: {error}")
        if "hotspot_strength" in values and "hotspot_strength_ceiling" in override:
            try:
                hotspot = finite_float(values["hotspot_strength"], "hotspot_strength")
                if hotspot > float(override["hotspot_strength_ceiling"]):
                    errors.append(f"class override failed {sample_id}: hotspot_strength={hotspot}")
            except ValueError as error:
                errors.append(f"invalid class override parameter {sample_id}: {error}")
    return record


def extract_condition_values(record: dict[str, Any]) -> dict[str, Any]:
    for key in ("condition", "parameters", "sampled"):
        value = record.get(key)
        if isinstance(value, dict):
            return value
    return record


def normalize_condition_result(result: Any) -> tuple[Image.Image, bytes | None]:
    if isinstance(result, Image.Image):
        return result.convert("RGB"), None
    if isinstance(result, (bytes, bytearray)):
        payload = bytes(result)
        with Image.open(__import__("io").BytesIO(payload)) as opened:
            image = opened.convert("RGB")
            image.load()
        return image, payload
    if isinstance(result, dict):
        payload = result.get("jpeg_bytes") or result.get("payload")
        for key in ("image", "conditioned", "output"):
            if isinstance(result.get(key), Image.Image):
                return result[key].convert("RGB"), bytes(payload) if payload is not None else None
    if isinstance(result, tuple):
        image: Image.Image | None = None
        payload: bytes | None = None
        for item in result:
            if image is None and isinstance(item, Image.Image):
                image = item.convert("RGB")
            elif payload is None and isinstance(item, (bytes, bytearray)):
                payload = bytes(item)
        if image is not None:
            return image, payload
    raise TypeError(f"unsupported apply_condition result type: {type(result).__name__}")


def jpeg_after_condition(
    generator: Any,
    source: Image.Image,
    parameter_record: dict[str, Any],
) -> tuple[Image.Image, bytes]:
    values = extract_condition_values(parameter_record)
    conditioned, direct_payload = normalize_condition_result(
        generator.apply_condition(source.copy(), values)
    )
    if direct_payload is not None:
        return conditioned, direct_payload
    quality = integer(values.get("jpeg_quality"), "jpeg_quality")
    decoded, payload = v2_generator.jpeg_roundtrip(conditioned, quality)
    return decoded, payload


def fallback_reconstruct_parent(
    parent: dict[str, str], v2_config: dict[str, Any]
) -> tuple[Image.Image, Image.Image, Image.Image]:
    base_path = resolve_repository_path(v2_config["base"]["path"], "v2.base.path")
    base = Image.open(base_path).convert("RGB")
    parameters = json.loads(parent["parameters_json"])
    rng = __import__("random").Random(int(parameters["effective_seed"]))
    rois = legacy.build_rois(v2_config)
    class_ids = {key: int(value) for key, value in v2_config["class_ids"].items()}
    defect, semantic, replay_params, _, _ = v2_generator.apply_single_recipe(
        base,
        parent["primary_class"],
        parent["severity"],
        rois,
        class_ids,
        rng,
    )
    if replay_params != parameters["instance_pre_transform"]:
        raise ValueError(f"parent recipe replay mismatch: {parent['sample_id']}")
    geometry = parameters["geometry"]
    photometric = parameters["photometric"]
    size = int(v2_config["image_size"])
    defect, semantic = v2_generator.apply_geometry(defect, semantic, geometry, size)
    clean, _ = v2_generator.apply_geometry(
        base, Image.new("L", base.size, 0), geometry, size
    )
    defect = v2_generator.apply_photometric(defect, photometric)
    clean = v2_generator.apply_photometric(clean, photometric)
    _, parent_payload = v2_generator.jpeg_roundtrip(
        defect, int(v2_config["jpeg_quality"])
    )
    if sha256_file(resolve_repository_path(parent["image_path"], "parent image")) != parent[
        "image_sha256"
    ]:
        raise ValueError(f"published parent image SHA mismatch: {parent['sample_id']}")
    if __import__("hashlib").sha256(parent_payload).hexdigest() != parent["image_sha256"]:
        raise ValueError(f"parent image replay mismatch: {parent['sample_id']}")
    return defect, clean, semantic


def reconstructed_parent_from_generator(
    generator: Any,
    parent: dict[str, str],
    v2_config: dict[str, Any],
    expected_defect: Image.Image,
    expected_mask: Image.Image,
    errors: list[str],
) -> tuple[Image.Image, Image.Image, Image.Image]:
    fallback_defect, fallback_clean, fallback_mask = fallback_reconstruct_parent(
        parent, v2_config
    )
    helper = getattr(generator, "reconstruct_parent", None)
    if not callable(helper):
        errors.append("generator does not export reconstruct_parent(parent_row, v2_config)")
        return fallback_defect, fallback_clean, fallback_mask
    try:
        result = helper(parent, v2_config)
    except Exception as error:
        errors.append(f"parent reconstruction helper failed {parent['sample_id']}: {error}")
        return fallback_defect, fallback_clean, fallback_mask

    defect: Image.Image | None = None
    clean: Image.Image | None = None
    mask: Image.Image | None = None
    if isinstance(result, dict):
        defect = result.get("defect") or result.get("parent_defect")
        clean = result.get("clean") or result.get("parent_clean")
        mask = result.get("mask") or result.get("parent_mask")
    elif isinstance(result, tuple):
        images = [item for item in result if isinstance(item, Image.Image)]
        if len(images) >= 3:
            defect, clean, mask = images[:3]
        elif len(images) == 2:
            defect, clean = images
        elif len(images) == 1:
            clean = images[0]
    elif isinstance(result, Image.Image):
        clean = result

    sample_id = parent["sample_id"]
    if isinstance(defect, Image.Image):
        defect = defect.convert("RGB")
        if not np.array_equal(np.asarray(defect), np.asarray(fallback_defect)):
            errors.append(f"reconstructed parent defect mismatch {sample_id}")
    else:
        errors.append(f"reconstruct_parent did not return a defect image {sample_id}")
        defect = fallback_defect
    if isinstance(mask, Image.Image):
        if not np.array_equal(np.asarray(mask.convert("L")), np.asarray(expected_mask)):
            errors.append(f"reconstructed parent mask mismatch {sample_id}")
    if not isinstance(clean, Image.Image):
        errors.append(f"reconstruct_parent did not return a clean image {sample_id}")
        return defect, fallback_clean, fallback_mask
    clean = clean.convert("RGB")
    if not np.array_equal(np.asarray(clean), np.asarray(fallback_clean)):
        errors.append(f"reconstructed parent clean mismatch {sample_id}")
        clean = fallback_clean
    # The helper returns the pre-JPEG parent.  Its published JPEG is checked by
    # both the helper and the independent fallback above; comparing it directly
    # with the decoded JPEG would incorrectly treat compression as a mismatch.
    del expected_defect
    return defect, clean, fallback_mask


def load_pinned_source(config: dict[str, Any]) -> dict[str, Any]:
    source = config.get("source")
    if not isinstance(source, dict):
        raise ValidationSetupError("config.source must be an object")
    source_config_path = require_pinned_file(source, "config_path", "config_sha256")
    source_manifest_path = require_pinned_file(source, "manifest_path", "manifest_sha256")
    split_path = require_pinned_file(
        source, "split_assignments_path", "split_assignments_sha256"
    )
    source_config = load_json(source_config_path)
    if source_config.get("release") != source.get("release"):
        raise ValidationSetupError("pinned source release name does not match source config")
    source_rows = read_csv(source_manifest_path)
    source_by_id: dict[str, dict[str, str]] = {}
    for row in source_rows:
        sample_id = row.get("sample_id", "")
        if not sample_id or sample_id in source_by_id:
            raise ValidationSetupError(f"invalid/duplicate source sample_id: {sample_id!r}")
        source_by_id[sample_id] = row

    assignments = read_csv(split_path)
    assignment_ids: set[str] = set()
    allow_ids: set[str] = set()
    required_split = str(source["required_parent_model_split"])
    for row in assignments:
        sample_id = row.get("sample_id", "")
        if not sample_id or sample_id in assignment_ids:
            raise ValidationSetupError(f"invalid/duplicate split assignment ID: {sample_id!r}")
        assignment_ids.add(sample_id)
        parent = source_by_id.get(sample_id)
        if parent is None:
            raise ValidationSetupError(f"split assignment has unknown source ID: {sample_id}")
        for field in ("image_path", "primary_class", "severity", "sample_seed", "image_sha256"):
            if row.get(field) != parent.get(field):
                raise ValidationSetupError(
                    f"split/source mismatch {sample_id} field={field}: "
                    f"split={row.get(field)!r} source={parent.get(field)!r}"
                )
        if row.get("model_split") == required_split:
            allow_ids.add(sample_id)

    if assignment_ids != set(source_by_id):
        raise ValidationSetupError(
            "source manifest/split assignment ID inventory mismatch: "
            f"extra={len(assignment_ids - set(source_by_id))} "
            f"missing={len(set(source_by_id) - assignment_ids)}"
        )

    expected_parent_count = int(source["expected_parent_count"])
    if len(allow_ids) != expected_parent_count:
        raise ValidationSetupError(
            f"gradient-train parent allowlist count mismatch: "
            f"expected={expected_parent_count} actual={len(allow_ids)}"
        )
    classes = list(source_config["primary_classes"])
    expected_per_class = int(source["expected_parent_count_per_class"])
    allow_counts = Counter(source_by_id[item]["primary_class"] for item in allow_ids)
    if {name: allow_counts[name] for name in classes} != {
        name: expected_per_class for name in classes
    }:
        raise ValidationSetupError(f"parent allowlist class counts mismatch: {dict(allow_counts)}")
    return {
        "config": source_config,
        "config_path": source_config_path,
        "manifest_path": source_manifest_path,
        "split_path": split_path,
        "rows": source_rows,
        "by_id": source_by_id,
        "allow_ids": allow_ids,
        "classes": classes,
    }


def validate_summary(
    summary_path: Path,
    rows: list[dict[str, str]],
    classes: list[str],
    profiles: list[str],
    severities: list[str],
    errors: list[str],
) -> None:
    if not summary_path.is_file():
        errors.append("missing summary.csv")
        return
    summary = read_csv(summary_path)
    if not summary or set(summary[0]) != {"axis", "class", "value", "count"}:
        errors.append("summary.csv header must be axis,class,value,count")
        return
    indexed: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for line_number, item in enumerate(summary, start=2):
        try:
            indexed[(item["axis"], item["class"], item["value"])].append(
                integer(item["count"], f"summary line {line_number} count")
            )
        except ValueError as error:
            errors.append(str(error))

    class_counts = Counter(row["primary_class"] for row in rows)
    class_severity = Counter((row["primary_class"], row["severity"]) for row in rows)
    class_profile = Counter(
        (row["primary_class"], row["condition_profile"]) for row in rows
    )
    expected: dict[tuple[str, str, str], int] = {}
    for class_name in classes:
        expected[("primary_class", class_name, class_name)] = class_counts[class_name]
        for severity in severities:
            expected[("severity", class_name, severity)] = class_severity[(class_name, severity)]
        for profile in profiles:
            expected[("condition_profile", class_name, profile)] = class_profile[
                (class_name, profile)
            ]
    for key, count in expected.items():
        values = indexed.get(key, [])
        if values != [count]:
            errors.append(f"summary mismatch {key}: expected={[count]} actual={values}")
    extra = sorted(set(indexed) - set(expected))
    if extra:
        errors.append(f"summary contains unexpected rows: {extra[:12]}")


def validate_release_metadata(
    release_path: Path,
    release_root: Path,
    config_path: Path,
    config: dict[str, Any],
    source_context: dict[str, Any],
    manifest_path: Path,
    instances_path: Path,
    summary_path: Path,
    rows: list[dict[str, str]],
    errors: list[str],
) -> None:
    if not release_path.is_file():
        errors.append("missing release.json")
        return
    try:
        metadata = load_json(release_path)
        if not isinstance(metadata, dict):
            raise TypeError("release.json must contain an object")
    except (json.JSONDecodeError, TypeError) as error:
        errors.append(f"invalid release.json: {error}")
        return

    profiles = list(config["profiles"])
    classes = source_context["classes"]
    parent_count = int(config["source"]["expected_parent_count"])
    expected_total = parent_count * len(profiles)
    expected_class = int(config["source"]["expected_parent_count_per_class"]) * len(profiles)
    class_counts = Counter(row["primary_class"] for row in rows)
    profile_counts = Counter(row["condition_profile"] for row in rows)
    class_profile = Counter(
        (row["primary_class"], row["condition_profile"]) for row in rows
    )
    expected_class_counts = {name: expected_class for name in classes}
    expected_profile_counts = {name: parent_count for name in profiles}
    expected_class_profile = {
        class_name: {
            profile: int(config["source"]["expected_parent_count_per_class"])
            for profile in profiles
        }
        for class_name in classes
    }
    scalar_expectations = {
        "release": config["release"],
        "generator_version": config["generator_version"],
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(config_path),
        "sample_count": expected_total,
        "parent_count": parent_count,
        "variants_per_parent": len(profiles),
        "source_release": config["source"]["release"],
        "source_manifest_sha256": config["source"]["manifest_sha256"],
        "source_config_sha256": config["source"]["config_sha256"],
        "source_split_assignments_sha256": config["source"]["split_assignments_sha256"],
        "training_use": config["training_use"],
        "evaluation_eligible": config["evaluation_eligible"],
    }
    for field, expected in scalar_expectations.items():
        if metadata.get(field) != expected:
            errors.append(
                f"release metadata mismatch {field}: expected={expected!r} "
                f"actual={metadata.get(field)!r}"
            )
    if "profiles" in metadata and metadata.get("profiles") != profiles:
        errors.append("release profiles mismatch")
    if metadata.get("class_counts") != expected_class_counts or dict(class_counts) != expected_class_counts:
        errors.append("release class_counts mismatch")
    if metadata.get("profile_counts") != expected_profile_counts or dict(profile_counts) != expected_profile_counts:
        errors.append("release profile_counts mismatch")
    if metadata.get("class_profile_counts") != expected_class_profile:
        errors.append("release class_profile_counts mismatch")
    if {
        class_name: {profile: class_profile[(class_name, profile)] for profile in profiles}
        for class_name in classes
    } != expected_class_profile:
        errors.append("computed class_profile_counts mismatch")

    generator_value = metadata.get("generator_script")
    try:
        generator_path = resolve_repository_path(str(generator_value), "generator_script")
    except ValidationSetupError as error:
        errors.append(str(error))
        generator_path = ROOT
    hash_targets = {
        "generator_script_sha256": generator_path,
        "config_sha256": config_path,
        "manifest_sha256": manifest_path,
        "instances_sha256": instances_path,
        "summary_sha256": summary_path,
    }
    for field, target in hash_targets.items():
        if not target.is_file():
            errors.append(f"release hash target missing {field}: {target}")
        elif metadata.get(field) != sha256_file(target):
            errors.append(f"release hash mismatch {field}")

    overview = release_root / "contact_sheet.jpg"
    overview_hash = metadata.get("overview_contact_sheet_sha256")
    if overview.exists() or overview_hash is not None:
        if not overview.is_file() or overview_hash != sha256_file(overview):
            errors.append("release hash mismatch overview_contact_sheet_sha256")

    sheet_root = release_root / "annotations" / "contact_sheets"
    actual_sheets = {
        path.resolve()
        for path in sheet_root.rglob("*")
        if path.is_file()
    } if sheet_root.exists() else set()
    inventory = metadata.get("full_contact_sheet_sha256", {})
    if not isinstance(inventory, dict):
        errors.append("full_contact_sheet_sha256 must be an object")
    else:
        inventory_paths: set[Path] = set()
        for relative, expected_hash in inventory.items():
            try:
                target = resolve_repository_path(relative, "full_contact_sheet_sha256 path")
            except ValidationSetupError as error:
                errors.append(str(error))
                continue
            inventory_paths.add(target)
            if release_root not in target.parents:
                errors.append(f"contact sheet outside release: {relative}")
            elif not target.is_file() or sha256_file(target) != expected_hash:
                errors.append(f"contact sheet hash mismatch: {relative}")
        if inventory_paths != actual_sheets:
            errors.append(
                "contact sheet inventory mismatch: "
                f"extra={len(actual_sheets - inventory_paths)} "
                f"missing={len(inventory_paths - actual_sheets)}"
            )


def print_failure(errors: Iterable[str]) -> int:
    items = list(errors)
    print(f"FAIL: errors={len(items)}")
    for error in items[:160]:
        print(f"- {error}")
    if len(items) > 160:
        print(f"- ... {len(items) - 160} additional errors")
    return 1


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    release_root = args.release.resolve()
    try:
        config = load_json(config_path)
        if not isinstance(config, dict):
            raise ValidationSetupError("config must contain a JSON object")
        profiles_value = config.get("profiles")
        if (
            not isinstance(profiles_value, list)
            or len(profiles_value) != 6
            or len(set(profiles_value)) != 6
            or any(not isinstance(item, str) or not item for item in profiles_value)
        ):
            raise ValidationSetupError("config must define exactly six unique condition profiles")
        hard_requirements = {
            "image_size": 512,
            "model_input_size": 224,
            "split": "train",
            "model_split": "gradient_train_auxiliary",
            "training_use": "TRAIN_ONLY_CONDITION_SYNTHETIC",
            "evaluation_eligible": "NO",
        }
        for field, expected in hard_requirements.items():
            if config.get(field) != expected:
                raise ValidationSetupError(
                    f"config hard requirement mismatch {field}: "
                    f"expected={expected!r} actual={config.get(field)!r}"
                )
        source_value = config.get("source")
        if not isinstance(source_value, dict):
            raise ValidationSetupError("config.source must be an object")
        source_requirements = {
            "required_parent_model_split": "gradient_train",
            "expected_parent_count": 168,
            "expected_parent_count_per_class": 24,
        }
        for field, expected in source_requirements.items():
            if source_value.get(field) != expected:
                raise ValidationSetupError(
                    f"config source requirement mismatch {field}: "
                    f"expected={expected!r} actual={source_value.get(field)!r}"
                )
        source_context = load_pinned_source(config)
        if len(source_context["classes"]) != 7:
            raise ValidationSetupError("pinned source must contain exactly seven classes")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, ValidationSetupError) as error:
        return print_failure([f"setup/source pin failure: {error}"])

    try:
        release_root.relative_to((ROOT / "synthetic").resolve())
    except ValueError:
        return print_failure([f"release must be under repository synthetic/: {release_root}"])

    errors: list[str] = []
    try:
        generator = importlib.import_module("generate_synthetic_v3_conditions")
    except Exception as error:
        generator = None
        errors.append(f"cannot import condition generator for deterministic replay: {error}")
    if generator is not None and not callable(getattr(generator, "apply_condition", None)):
        errors.append("generator does not export apply_condition(image, params)")

    manifest_path = release_root / "annotations" / "manifest.csv"
    instances_path = release_root / "annotations" / "instances.jsonl"
    summary_path = release_root / "annotations" / "summary.csv"
    metadata_path = release_root / "annotations" / "release.json"
    if not manifest_path.is_file():
        return print_failure(errors + [f"missing manifest: {manifest_path}"])
    rows = read_csv(manifest_path)

    required_columns = {
        "sample_id", "image_path", "mask_path", "domain", "split", "model_split",
        "base_group_id", "source_specimen_group", "primary_class", "visible_multilabel",
        "severity", "source_release", "global_seed",
        "generator_version", "qc_gate_version", "config_sha256", "image_sha256",
        "mask_sha256", "width", "height", "defect_pixels", "bbox_x", "bbox_y",
        "bbox_w", "bbox_h", "training_use",
        "evaluation_eligible", "qc_status", "human_verified", "parent_sample_id",
        "parent_image_path", "parent_mask_path", "parent_image_sha256",
        "parent_mask_sha256", "parent_sample_seed", "lineage_group_id", "family_split_id",
        "defect_instance_id", "augmentation_family_id", "derivation_depth", "variant_index",
        "condition_profile", "condition_seed", "attempt", "parameters_json",
        "qc_metrics_json", "luma_metrics_json",
    }
    if not rows:
        return print_failure(errors + ["manifest is empty"])
    missing = sorted(required_columns - set(rows[0]))
    if missing:
        return print_failure(errors + [f"manifest missing columns: {missing}"])

    classes = source_context["classes"]
    class_ids = {
        key: int(value) for key, value in source_context["config"]["class_ids"].items()
    }
    profiles = list(config["profiles"])
    profile_index = {name: index for index, name in enumerate(profiles)}
    expected_parent_count = int(config["source"]["expected_parent_count"])
    expected_per_class_parent = int(config["source"]["expected_parent_count_per_class"])
    expected_total = expected_parent_count * len(profiles)
    if len(rows) != expected_total:
        errors.append(f"manifest row count mismatch: expected={expected_total} actual={len(rows)}")

    ids: set[str] = set()
    image_paths: set[str] = set()
    mask_paths: set[str] = set()
    image_hashes: set[str] = set()
    condition_seeds: set[str] = set()
    expected_images: set[Path] = set()
    expected_masks: set[Path] = set()
    rows_by_parent: dict[str, list[dict[str, str]]] = defaultdict(list)
    instance_rows: dict[str, dict[str, Any]] = {}
    parent_cache: dict[str, tuple[Image.Image, Image.Image, Image.Image]] = {}
    config_sha = sha256_file(config_path)

    if not instances_path.is_file():
        errors.append("missing instances.jsonl")
    else:
        with instances_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise TypeError("instance line must decode to an object")
                    record_id = record["sample_id"]
                    if record_id in instance_rows:
                        errors.append(f"duplicate instance record {record_id}")
                    instance_rows[record_id] = record
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    errors.append(f"invalid instances.jsonl line {line_number}: {error}")

    for row_number, row in enumerate(rows, start=2):
        sample_id = row["sample_id"].strip()
        parent_id = row["parent_sample_id"].strip()
        profile = row["condition_profile"].strip()
        prefix = f"row {row_number} ({sample_id or 'UNKNOWN'})"
        if not sample_id or sample_id in ids:
            errors.append(f"{prefix}: empty/duplicate sample_id")
        else:
            ids.add(sample_id)
        if parent_id not in source_context["allow_ids"]:
            errors.append(f"{prefix}: parent is not in pinned gradient_train allowlist: {parent_id}")
            parent = None
        else:
            parent = source_context["by_id"][parent_id]
            rows_by_parent[parent_id].append(row)
        if profile not in profile_index:
            errors.append(f"{prefix}: unexpected condition_profile {profile!r}")

        for field, collection in (
            ("image_path", image_paths),
            ("mask_path", mask_paths),
            ("condition_seed", condition_seeds),
        ):
            value = row[field].strip()
            if not value or value in collection:
                errors.append(f"{prefix}: empty/duplicate {field}: {value!r}")
            collection.add(value)
        image_sha = row["image_sha256"].lower()
        if image_sha in image_hashes:
            errors.append(f"{prefix}: duplicate image_sha256 {image_sha}")
        image_hashes.add(image_sha)

        try:
            image_path = resolve_repository_path(row["image_path"], "image_path")
            mask_path = resolve_repository_path(row["mask_path"], "mask_path")
        except ValidationSetupError as error:
            errors.append(f"{prefix}: {error}")
            continue
        expected_images.add(image_path)
        expected_masks.add(mask_path)
        if (release_root / "images").resolve() not in image_path.parents:
            errors.append(f"{prefix}: image path outside release images/")
        if (release_root / "masks").resolve() not in mask_path.parents:
            errors.append(f"{prefix}: mask path outside release masks/")
        if not image_path.is_file() or not mask_path.is_file():
            errors.append(f"{prefix}: missing image/mask pair")
            continue

        try:
            with Image.open(image_path) as opened:
                if opened.format != "JPEG" or opened.size != (int(config["image_size"]),) * 2:
                    errors.append(
                        f"{prefix}: image format/size mismatch {opened.format}/{opened.size}"
                    )
                defect = opened.convert("RGB")
                defect.load()
            with Image.open(mask_path) as opened:
                if opened.format != "PNG" or opened.mode != "L" or opened.size != defect.size:
                    errors.append(
                        f"{prefix}: mask format/mode/size mismatch "
                        f"{opened.format}/{opened.mode}/{opened.size}"
                    )
                semantic = opened.convert("L")
                semantic.load()
        except Exception as error:
            errors.append(f"{prefix}: decode failure: {error}")
            continue

        if sha256_file(image_path) != image_sha:
            errors.append(f"{prefix}: image SHA mismatch")
        if sha256_file(mask_path) != row["mask_sha256"].lower():
            errors.append(f"{prefix}: mask SHA mismatch")
        fixed_values = {
            "domain": config["source_domain"],
            "source_release": config["source"]["release"],
            "split": config["split"],
            "model_split": config["model_split"],
            "training_use": config["training_use"],
            "evaluation_eligible": config["evaluation_eligible"],
            "generator_version": config["generator_version"],
            "qc_gate_version": config["qc_gate_version"],
            "config_sha256": config_sha,
            "qc_status": EXPECTED_QC_STATUS,
            "human_verified": "NO",
            "global_seed": str(config["global_seed"]),
            "derivation_depth": "1",
        }
        for field, expected in fixed_values.items():
            if row[field] != expected:
                errors.append(
                    f"{prefix}: fixed field mismatch {field}: "
                    f"expected={expected!r} actual={row[field]!r}"
                )
        try:
            expected_size = int(config["image_size"])
            if integer(row["width"], "width") != expected_size or integer(
                row["height"], "height"
            ) != expected_size:
                errors.append(
                    f"{prefix}: manifest dimensions must be {expected_size}x{expected_size}"
                )
            attempt = integer(row["attempt"], "attempt")
            if not 0 <= attempt < int(config["max_condition_attempts"]):
                errors.append(f"{prefix}: attempt out of range {attempt}")
            variant_index = integer(row["variant_index"], "variant_index")
            if profile in profile_index and variant_index != profile_index[profile]:
                errors.append(
                    f"{prefix}: variant_index/profile mismatch "
                    f"{variant_index}!={profile_index[profile]}"
                )
        except ValueError as error:
            errors.append(f"{prefix}: {error}")

        parameter_record = validate_condition_parameters(row, config, errors)
        if generator is not None and parameter_record is not None:
            stable_seed = getattr(generator, "stable_condition_seed", None)
            sampler = getattr(generator, "sample_condition", None)
            if not callable(stable_seed) or not callable(sampler):
                errors.append(
                    f"{prefix}: generator must export stable_condition_seed and sample_condition"
                )
            elif profile in profile_index:
                try:
                    expected_seed = int(
                        stable_seed(
                            int(config["global_seed"]),
                            str(config["release"]),
                            parent_id,
                            profile,
                        )
                    )
                    recorded_seed = integer(row["condition_seed"], "condition_seed")
                    if recorded_seed != expected_seed:
                        errors.append(
                            f"{prefix}: stable condition seed mismatch "
                            f"expected={expected_seed} actual={recorded_seed}"
                        )
                    attempt_value = integer(row["attempt"], "attempt")
                    expected_effective_seed = expected_seed + attempt_value * 104729
                    if integer(
                        parameter_record.get("effective_seed"), "effective_seed"
                    ) != expected_effective_seed:
                        errors.append(f"{prefix}: effective_seed mismatch")
                    expected_parameters = sampler(
                        config,
                        profile,
                        row["primary_class"],
                        expected_effective_seed,
                    )
                    attenuator = getattr(generator, "attenuate_condition", None)
                    attenuation = config.get("adaptive_visibility_attenuation")
                    if isinstance(attenuation, dict):
                        if not callable(attenuator):
                            raise TypeError(
                                "generator config enables adaptive attenuation but "
                                "attenuate_condition is not exported"
                            )
                        full_attempts = int(attenuation["full_strength_attempts"])
                        if attempt_value < full_attempts:
                            strength_scale = 1.0
                        else:
                            maximum_attempts = int(config["max_condition_attempts"])
                            tail_count = max(1, maximum_attempts - full_attempts - 1)
                            tail_index = attempt_value - full_attempts
                            minimum_scale = float(attenuation["minimum_strength_scale"])
                            strength_scale = 1.0 - (
                                1.0 - minimum_scale
                            ) * tail_index / tail_count
                        expected_parameters = attenuator(
                            expected_parameters,
                            strength_scale,
                            int(attenuation["reference_jpeg_quality"]),
                        )
                    if extract_condition_values(parameter_record) != expected_parameters:
                        errors.append(f"{prefix}: deterministic condition parameter replay mismatch")
                except (TypeError, ValueError, KeyError) as error:
                    errors.append(f"{prefix}: condition seed/parameter replay failure: {error}")
        luma = validate_luma(sample_id, defect, row["luma_metrics_json"], config["qc"], errors)
        del luma

        if parent is None:
            continue
        parent_match_fields = (
            "base_group_id", "source_specimen_group", "primary_class",
            "visible_multilabel", "severity",
        )
        for field in parent_match_fields:
            if row[field] != parent[field]:
                errors.append(
                    f"{prefix}: parent lineage mismatch {field}: "
                    f"row={row[field]!r} parent={parent[field]!r}"
                )
        explicit_parent = {
            "parent_image_path": parent["image_path"],
            "parent_mask_path": parent["mask_path"],
            "parent_image_sha256": parent["image_sha256"],
            "parent_mask_sha256": parent["mask_sha256"],
            "parent_sample_seed": parent["sample_seed"],
        }
        for field, expected in explicit_parent.items():
            if row[field] != expected:
                errors.append(
                    f"{prefix}: explicit parent field mismatch {field}: "
                    f"expected={expected!r} actual={row[field]!r}"
                )
        if row["mask_sha256"] != parent["mask_sha256"]:
            errors.append(f"{prefix}: output mask bytes differ from pinned parent mask")
        for field in (
            "lineage_group_id", "family_split_id", "defect_instance_id",
            "augmentation_family_id",
        ):
            if not row[field].strip():
                errors.append(f"{prefix}: empty lineage field {field}")

        try:
            parent_image_path = resolve_repository_path(parent["image_path"], "parent image")
            parent_mask_path = resolve_repository_path(parent["mask_path"], "parent mask")
            if sha256_file(parent_image_path) != parent["image_sha256"]:
                errors.append(f"{prefix}: pinned parent image changed")
            if sha256_file(parent_mask_path) != parent["mask_sha256"]:
                errors.append(f"{prefix}: pinned parent mask changed")
            if parent_id not in parent_cache:
                with Image.open(parent_image_path) as opened:
                    parent_defect = opened.convert("RGB")
                    parent_defect.load()
                with Image.open(parent_mask_path) as opened:
                    parent_mask = opened.convert("L")
                    parent_mask.load()
                if generator is not None:
                    replay_defect, parent_clean, replay_mask = reconstructed_parent_from_generator(
                        generator,
                        parent,
                        source_context["config"],
                        parent_defect,
                        parent_mask,
                        errors,
                    )
                else:
                    replay_defect, parent_clean, replay_mask = fallback_reconstruct_parent(
                        parent, source_context["config"]
                    )
                if not np.array_equal(np.asarray(replay_mask.convert("L")), np.asarray(parent_mask)):
                    errors.append(f"{prefix}: replayed parent mask differs from published parent mask")
                parent_cache[parent_id] = (replay_defect, parent_clean, parent_mask)
            parent_defect, parent_clean, parent_mask = parent_cache[parent_id]
        except Exception as error:
            errors.append(f"{prefix}: parent load/reconstruction failure: {error}")
            continue

        output_mask_array = np.asarray(semantic, dtype=np.uint8)
        parent_mask_array = np.asarray(parent_mask, dtype=np.uint8)
        if not np.array_equal(output_mask_array, parent_mask_array):
            errors.append(f"{prefix}: output mask differs from parent mask")
        expected_class_id = class_ids.get(row["primary_class"])
        mask_values = {int(value) for value in np.unique(output_mask_array)}
        if expected_class_id is None or mask_values - {0} != {expected_class_id}:
            errors.append(f"{prefix}: mask class values mismatch {sorted(mask_values)}")
            continue
        class_mask = output_mask_array == expected_class_id
        area = int(np.count_nonzero(class_mask))
        bbox = mask_bbox(class_mask)
        try:
            recorded_bbox = tuple(integer(row[name], name) for name in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"))
            if integer(row["defect_pixels"], "defect_pixels") != area:
                errors.append(f"{prefix}: defect_pixels mismatch")
            if recorded_bbox != bbox:
                errors.append(f"{prefix}: bbox mismatch recorded={recorded_bbox} actual={bbox}")
            if area != integer(parent["defect_pixels"], "parent.defect_pixels"):
                errors.append(f"{prefix}: parent defect area mismatch")
        except ValueError as error:
            errors.append(f"{prefix}: {error}")

        instance = instance_rows.get(sample_id)
        if instance is None:
            errors.append(f"{prefix}: missing instance record")
        else:
            expected_instance_core = {
                "sample_id": sample_id,
                "primary_class": row["primary_class"],
                "visible_multilabel": [row["primary_class"]],
                "semantic_mask_path": row["mask_path"],
            }
            for field, expected in expected_instance_core.items():
                if instance.get(field) != expected:
                    errors.append(f"{prefix}: instance field mismatch {field}")
            items = instance.get("instances")
            if not isinstance(items, list) or len(items) != 1:
                errors.append(f"{prefix}: instances must contain exactly one object")
            else:
                expected_item = {
                    "category": row["primary_class"],
                    "category_id": expected_class_id,
                    "area_px": area,
                    "bbox_xywh": list(bbox),
                }
                for field, expected in expected_item.items():
                    if items[0].get(field) != expected:
                        errors.append(f"{prefix}: instance object mismatch {field}")

        if parameter_record is None or generator is None or not callable(
            getattr(generator, "apply_condition", None)
        ):
            continue
        try:
            replay_defect, replay_payload = jpeg_after_condition(
                generator, parent_defect, parameter_record
            )
            if replay_payload != image_path.read_bytes():
                errors.append(f"{prefix}: deterministic JPEG replay mismatch")
            if not np.array_equal(np.asarray(replay_defect), np.asarray(defect)):
                errors.append(f"{prefix}: deterministic decoded replay mismatch")
            replay_clean, _ = jpeg_after_condition(generator, parent_clean, parameter_record)
            metrics, gate_failures = v2_generator.evaluate_visibility(
                defect,
                replay_clean,
                semantic,
                row["primary_class"],
                row["severity"],
                source_context["config"],
            )
            if bool(config["qc"].get("require_parent_visibility_gate", True)) and gate_failures:
                errors.append(
                    f"{prefix}: post-condition visibility gate failed: "
                    + "; ".join(gate_failures)
                )
            try:
                recorded_metrics = json.loads(row["qc_metrics_json"])
                if not isinstance(recorded_metrics, dict):
                    raise TypeError("qc_metrics_json must decode to an object")
                compare_nested_metrics(sample_id, recorded_metrics, metrics, errors)
            except (json.JSONDecodeError, TypeError) as error:
                errors.append(f"{prefix}: invalid qc_metrics_json: {error}")
        except Exception as error:
            errors.append(f"{prefix}: condition/QC replay failure: {error}")

    if set(instance_rows) != ids:
        errors.append(
            f"instances ID inventory mismatch: extra={len(set(instance_rows) - ids)} "
            f"missing={len(ids - set(instance_rows))}"
        )
    actual_images = {
        path.resolve() for path in (release_root / "images").rglob("*") if path.is_file()
    } if (release_root / "images").exists() else set()
    actual_masks = {
        path.resolve() for path in (release_root / "masks").rglob("*") if path.is_file()
    } if (release_root / "masks").exists() else set()
    if actual_images != expected_images:
        errors.append(
            f"image orphan/missing inventory: extra={len(actual_images - expected_images)} "
            f"missing={len(expected_images - actual_images)}"
        )
    if actual_masks != expected_masks:
        errors.append(
            f"mask orphan/missing inventory: extra={len(actual_masks - expected_masks)} "
            f"missing={len(expected_masks - actual_masks)}"
        )

    if set(rows_by_parent) != source_context["allow_ids"]:
        errors.append(
            f"parent inventory mismatch: extra={len(set(rows_by_parent) - source_context['allow_ids'])} "
            f"missing={len(source_context['allow_ids'] - set(rows_by_parent))}"
        )
    lineage_owner: dict[tuple[str, str], str] = {}
    for parent_id, items in rows_by_parent.items():
        if len(items) != len(profiles):
            errors.append(
                f"parent variant count mismatch {parent_id}: "
                f"expected={len(profiles)} actual={len(items)}"
            )
        observed_profiles = Counter(item["condition_profile"] for item in items)
        if observed_profiles != Counter({name: 1 for name in profiles}):
            errors.append(f"parent profile inventory mismatch {parent_id}: {dict(observed_profiles)}")
        for field in (
            "lineage_group_id", "family_split_id", "defect_instance_id",
            "augmentation_family_id",
        ):
            values = {item[field] for item in items}
            if len(values) != 1:
                errors.append(f"parent lineage field not stable {parent_id} {field}: {sorted(values)}")
                continue
            value = next(iter(values))
            owner_key = (field, value)
            previous = lineage_owner.get(owner_key)
            if previous is not None and previous != parent_id:
                errors.append(
                    f"lineage identifier shared by parents {field}={value}: "
                    f"{previous}, {parent_id}"
                )
            lineage_owner[owner_key] = parent_id

    class_counts = Counter(row["primary_class"] for row in rows)
    expected_class_count = expected_per_class_parent * len(profiles)
    if {name: class_counts[name] for name in classes} != {
        name: expected_class_count for name in classes
    }:
        errors.append(f"class counts mismatch: {dict(class_counts)}")
    profile_counts = Counter(row["condition_profile"] for row in rows)
    if {name: profile_counts[name] for name in profiles} != {
        name: expected_parent_count for name in profiles
    }:
        errors.append(f"profile counts mismatch: {dict(profile_counts)}")
    class_profile = Counter(
        (row["primary_class"], row["condition_profile"]) for row in rows
    )
    for class_name in classes:
        for profile in profiles:
            if class_profile[(class_name, profile)] != expected_per_class_parent:
                errors.append(
                    f"class/profile count mismatch {class_name}/{profile}: "
                    f"{class_profile[(class_name, profile)]}"
                )

    parent_severity = Counter(
        (
            source_context["by_id"][parent_id]["primary_class"],
            source_context["by_id"][parent_id]["severity"],
        )
        for parent_id in source_context["allow_ids"]
    )
    output_severity = Counter((row["primary_class"], row["severity"]) for row in rows)
    for key, count in parent_severity.items():
        if output_severity[key] != count * len(profiles):
            errors.append(
                f"class/severity lineage count mismatch {key}: "
                f"expected={count * len(profiles)} actual={output_severity[key]}"
            )

    validate_summary(
        summary_path,
        rows,
        classes,
        profiles,
        list(source_context["config"]["severity_quotas"]),
        errors,
    )
    validate_release_metadata(
        metadata_path,
        release_root,
        config_path,
        config,
        source_context,
        manifest_path,
        instances_path,
        summary_path,
        rows,
        errors,
    )

    current_ids = ids
    current_hashes = image_hashes
    for other_manifest in (ROOT / "synthetic").glob("*/annotations/manifest.csv"):
        if other_manifest.resolve() == manifest_path.resolve():
            continue
        try:
            other_rows = read_csv(other_manifest)
        except OSError as error:
            errors.append(f"cannot audit cross-release manifest {other_manifest}: {error}")
            continue
        other_ids = {item.get("sample_id", "") for item in other_rows}
        other_hashes = {item.get("image_sha256", "") for item in other_rows}
        if current_ids & other_ids:
            errors.append(f"cross-release sample ID duplicate with {other_manifest.parents[1].name}")
        if current_hashes & other_hashes:
            errors.append(f"cross-release image duplicate with {other_manifest.parents[1].name}")

    if errors:
        return print_failure(errors)
    print(
        f"PASS: synthetic={len(rows)}, parents={len(rows_by_parent)}, "
        f"profiles={len(profiles)}, classes={len(classes)}, "
        "gradient_train_only=YES, deterministic_replay=PASS, "
        "post_jpeg_512_224=PASS, evaluation_eligible=NO"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
