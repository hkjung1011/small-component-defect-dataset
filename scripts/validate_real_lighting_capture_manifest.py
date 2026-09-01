#!/usr/bin/env python3
"""Validate the real-lighting capture manifest and its leakage controls.

The default input is the public, header-only template.  Use ``--schema-only``
to validate that template before real measurements have been entered.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "annotations" / "real_lighting_capture_template.csv"
SCHEMA_VERSION = "real-lighting-capture-v2"

COLUMNS = (
    "schema_version",
    "source_domain",
    "illuminance_value_kind",
    "capture_id",
    "filename",
    "file_sha256",
    "captured_at_utc",
    "measurement_capture_offset_limit_seconds",
    "site_id",
    "station_id",
    "operator_id",
    "session_id",
    "session_group_id",
    "condition_id",
    "lighting_profile_id",
    "camera_profile_id",
    "specimen_id",
    "specimen_group_id",
    "production_lot_id",
    "view_id",
    "specimen_status",
    "visible_defect_status",
    "visible_defect_classes_json",
    "split",
    "split_group_id",
    "evaluation_eligible",
    "background_id",
    "background_color",
    "background_material",
    "background_surface_state",
    "camera_make",
    "camera_model",
    "camera_serial",
    "lens_model",
    "focal_length_mm",
    "camera_azimuth_deg",
    "camera_elevation_deg",
    "camera_distance_mm",
    "image_width_px",
    "image_height_px",
    "bit_depth",
    "image_format",
    "exposure_mode",
    "exposure_time_us",
    "iso",
    "analog_gain_db",
    "aperture_f_number",
    "exposure_compensation_ev",
    "white_balance_mode",
    "white_balance_kelvin",
    "wb_red_gain",
    "wb_blue_gain",
    "focus_mode",
    "lux_meter_id",
    "lux_meter_serial",
    "lux_meter_calibration_date",
    "lux_meter_calibration_due_date",
    "lux_meter_calibration_certificate",
    "lux_measurement_evidence_id",
    "lux_measurement_evidence_sha256",
    "meter_measurement_geometry",
    "meter_location_frame",
    "meter_x_mm",
    "meter_y_mm",
    "meter_z_mm",
    "meter_sensor_normal_azimuth_deg",
    "meter_sensor_normal_elevation_deg",
    "lux_measurement_timestamp_utc",
    "illuminance_lux_measured",
    "illuminance_repeat_count",
    "illuminance_std_lux",
    "cct_meter_id",
    "cct_meter_serial",
    "cct_meter_calibration_date",
    "cct_meter_calibration_due_date",
    "cct_meter_calibration_certificate",
    "cct_measurement_evidence_id",
    "cct_measurement_evidence_sha256",
    "cct_measurement_timestamp_utc",
    "cct_measured_k",
    "light_count",
    "multi_light_mode",
    "minimum_active_light_angular_separation_deg",
    "lights_json",
    "component_only_illumination",
    "direct_belt_illumination",
    "stray_light_control",
    "shadow_present",
    "shadow_type",
    "shadows_json",
    "component_clipped_high_pct",
    "component_clipped_low_pct",
    "focus_metric_name",
    "focus_metric_value",
    "capture_qc_status",
    "capture_qc_notes",
    "label_review_status",
    "label_reviewer",
    "label_review_date",
)

LIGHT_KEYS = (
    "light_id",
    "role",
    "source_type",
    "azimuth_deg",
    "elevation_deg",
    "distance_mm",
    "nominal_cct_k",
    "power_setting_pct",
    "diffuser_id",
    "polarizer_angle_deg",
)

SHADOW_KEYS = (
    "shadow_id",
    "source_light_id",
    "shadow_type",
    "direction_azimuth_deg",
    "length_mm",
    "contrast_ratio",
    "measurement_method",
)

DEFECT_CLASSES = {
    "scratch",
    "surface_spot",
    "discoloration",
    "contamination",
    "lead_breakage",
    "body_chip",
    "body_crack",
}

ID_FIELDS = (
    "capture_id",
    "site_id",
    "station_id",
    "operator_id",
    "session_id",
    "session_group_id",
    "condition_id",
    "lighting_profile_id",
    "camera_profile_id",
    "specimen_id",
    "specimen_group_id",
    "production_lot_id",
    "view_id",
    "split_group_id",
    "background_id",
    "camera_serial",
    "lux_meter_id",
    "lux_meter_serial",
    "lux_measurement_evidence_id",
    "cct_meter_id",
    "cct_meter_serial",
    "cct_measurement_evidence_id",
    "label_reviewer",
)

OPTIONAL_FIELDS = {
    "iso",
    "analog_gain_db",
    "white_balance_kelvin",
    "wb_red_gain",
    "wb_blue_gain",
    "focus_metric_name",
    "focus_metric_value",
    "capture_qc_notes",
}

TEXT_FIELDS = (
    "background_material",
    "background_surface_state",
    "camera_make",
    "camera_model",
    "lens_model",
    "lux_meter_calibration_certificate",
    "meter_location_frame",
    "cct_meter_calibration_certificate",
    "stray_light_control",
    "focus_metric_name",
    "capture_qc_notes",
)

ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"CSV manifest (default: {DEFAULT_MANIFEST})",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--schema-only",
        action="store_true",
        help="check the exact header-only template; populated manifests fail",
    )
    mode.add_argument(
        "--self-test",
        action="store_true",
        help="run built-in positive and negative semantic tests",
    )
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="also require every filename and SHA-256 under --images-root",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        help="root directory used only with --check-files",
    )
    return parser.parse_args()


def row_error(errors: list[str], line: int, field: str, message: str) -> None:
    errors.append(f"line {line}, {field}: {message}")


def parse_number(
    row: dict[str, str],
    line: int,
    field: str,
    errors: list[str],
    *,
    integer: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
    maximum_exclusive: bool = False,
    optional: bool = False,
) -> float | int | None:
    raw = row[field].strip()
    if not raw:
        if not optional:
            row_error(errors, line, field, "required numeric value is blank")
        return None
    try:
        value: float | int
        if integer:
            if not re.fullmatch(r"[+-]?\d+", raw):
                raise ValueError
            value = int(raw)
        else:
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError
    except ValueError:
        row_error(errors, line, field, f"not a finite {'integer' if integer else 'number'}")
        return None
    if minimum is not None and value < minimum:
        row_error(errors, line, field, f"must be >= {minimum}")
    if maximum is not None:
        if maximum_exclusive and value >= maximum:
            row_error(errors, line, field, f"must be < {maximum}")
        elif not maximum_exclusive and value > maximum:
            row_error(errors, line, field, f"must be <= {maximum}")
    return value


def parse_date_value(
    row: dict[str, str], line: int, field: str, errors: list[str]
) -> date | None:
    try:
        return date.fromisoformat(row[field].strip())
    except ValueError:
        row_error(errors, line, field, "must be YYYY-MM-DD")
        return None


def parse_utc_timestamp(
    row: dict[str, str], line: int, field: str, errors: list[str]
) -> datetime | None:
    raw = row[field].strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        value = datetime.fromisoformat(raw)
    except ValueError:
        row_error(errors, line, field, "must be an ISO-8601 timestamp with UTC offset")
        return None
    if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
        row_error(errors, line, field, "must be UTC (Z or +00:00)")
        return None
    return value


def reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def json_number(
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    maximum_exclusive: bool = False,
) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    if not math.isfinite(number):
        return False
    if minimum is not None and number < minimum:
        return False
    if maximum is not None:
        if maximum_exclusive and number >= maximum:
            return False
        if not maximum_exclusive and number > maximum:
            return False
    return True


def parse_lights(
    row: dict[str, str], line: int, errors: list[str]
) -> tuple[list[dict[str, Any]], str] | None:
    try:
        value = json.loads(
            row["lights_json"], object_pairs_hook=reject_duplicate_json_keys
        )
    except (json.JSONDecodeError, ValueError) as exc:
        row_error(errors, line, "lights_json", f"invalid strict JSON: {exc}")
        return None
    if not isinstance(value, list) or not value:
        row_error(errors, line, "lights_json", "must be a non-empty JSON array")
        return None

    seen_ids: set[str] = set()
    for index, light in enumerate(value, start=1):
        prefix = f"lights_json[{index}]"
        if not isinstance(light, dict):
            row_error(errors, line, prefix, "must be a JSON object")
            continue
        if set(light.keys()) != set(LIGHT_KEYS):
            row_error(
                errors,
                line,
                prefix,
                f"keys must be exactly {list(LIGHT_KEYS)}",
            )
            continue
        light_id = light["light_id"]
        if not isinstance(light_id, str) or not ID_RE.fullmatch(light_id):
            row_error(errors, line, f"{prefix}.light_id", "invalid identifier")
        elif light_id in seen_ids:
            row_error(errors, line, f"{prefix}.light_id", "duplicate active light")
        else:
            seen_ids.add(light_id)

        if light["role"] not in {"key", "fill", "rim", "auxiliary"}:
            row_error(errors, line, f"{prefix}.role", "invalid role")
        for field in ("source_type", "diffuser_id"):
            text = light[field]
            if not isinstance(text, str) or not ID_RE.fullmatch(text):
                row_error(errors, line, f"{prefix}.{field}", "invalid identifier")
        numeric_rules = {
            "azimuth_deg": (0.0, 360.0, True),
            "elevation_deg": (0.0, 90.0, False),
            "distance_mm": (0.000001, None, False),
            "nominal_cct_k": (0.000001, None, False),
            "power_setting_pct": (0.0, 100.0, False),
        }
        for field, (minimum, maximum, exclusive) in numeric_rules.items():
            if not json_number(
                light[field],
                minimum=minimum,
                maximum=maximum,
                maximum_exclusive=exclusive,
            ):
                row_error(errors, line, f"{prefix}.{field}", "invalid numeric range")
        polarizer = light["polarizer_angle_deg"]
        if polarizer is not None and not json_number(
            polarizer, minimum=0.0, maximum=180.0, maximum_exclusive=True
        ):
            row_error(
                errors,
                line,
                f"{prefix}.polarizer_angle_deg",
                "must be null or in [0, 180)",
            )

    normalized_lights: list[dict[str, Any]] = []
    for light in value:
        if not isinstance(light, dict) or set(light) != set(LIGHT_KEYS):
            normalized_lights = value
            break
        normalized = dict(light)
        for field in (
            "azimuth_deg",
            "elevation_deg",
            "distance_mm",
            "nominal_cct_k",
            "power_setting_pct",
        ):
            if json_number(normalized[field]):
                normalized[field] = float(normalized[field])
        if normalized["polarizer_angle_deg"] is not None and json_number(
            normalized["polarizer_angle_deg"]
        ):
            normalized["polarizer_angle_deg"] = float(
                normalized["polarizer_angle_deg"]
            )
        normalized_lights.append(normalized)
    if all(isinstance(light, dict) for light in normalized_lights):
        normalized_lights = sorted(
            normalized_lights, key=lambda light: str(light.get("light_id", ""))
        )
    canonical = json.dumps(
        normalized_lights,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return value, canonical


def light_unit_vector(light: dict[str, Any]) -> tuple[float, float, float] | None:
    azimuth = light.get("azimuth_deg")
    elevation = light.get("elevation_deg")
    if not json_number(
        azimuth, minimum=0.0, maximum=360.0, maximum_exclusive=True
    ) or not json_number(elevation, minimum=0.0, maximum=90.0):
        return None
    azimuth_rad = math.radians(float(azimuth))
    elevation_rad = math.radians(float(elevation))
    horizontal = math.cos(elevation_rad)
    return (
        horizontal * math.cos(azimuth_rad),
        horizontal * math.sin(azimuth_rad),
        math.sin(elevation_rad),
    )


def maximum_light_angular_separation(
    lights: list[dict[str, Any]],
) -> float | None:
    vectors = [light_unit_vector(light) for light in lights]
    maximum: float | None = None
    for left_index, left in enumerate(vectors):
        if left is None:
            continue
        for right in vectors[left_index + 1 :]:
            if right is None:
                continue
            dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right))))
            separation = math.degrees(math.acos(dot))
            maximum = separation if maximum is None else max(maximum, separation)
    return maximum


def parse_shadows(
    row: dict[str, str],
    line: int,
    active_light_ids: set[str],
    errors: list[str],
) -> tuple[list[dict[str, Any]], str] | None:
    try:
        value = json.loads(
            row["shadows_json"], object_pairs_hook=reject_duplicate_json_keys
        )
    except (json.JSONDecodeError, ValueError) as exc:
        row_error(errors, line, "shadows_json", f"invalid strict JSON: {exc}")
        return None
    if not isinstance(value, list):
        row_error(errors, line, "shadows_json", "must be a JSON array")
        return None

    seen_shadow_ids: set[str] = set()
    seen_directional_sources: set[str] = set()
    normalized_shadows: list[dict[str, Any]] = []
    for index, shadow in enumerate(value, start=1):
        prefix = f"shadows_json[{index}]"
        if not isinstance(shadow, dict):
            row_error(errors, line, prefix, "must be a JSON object")
            continue
        if set(shadow) != set(SHADOW_KEYS):
            row_error(
                errors,
                line,
                prefix,
                f"keys must be exactly {list(SHADOW_KEYS)}",
            )
            continue

        shadow_id = shadow["shadow_id"]
        if not isinstance(shadow_id, str) or not ID_RE.fullmatch(shadow_id):
            row_error(errors, line, f"{prefix}.shadow_id", "invalid identifier")
        elif shadow_id in seen_shadow_ids:
            row_error(errors, line, f"{prefix}.shadow_id", "duplicate shadow")
        else:
            seen_shadow_ids.add(shadow_id)

        shadow_type = shadow["shadow_type"]
        if shadow_type not in {"contact", "directional", "uncontrolled"}:
            row_error(errors, line, f"{prefix}.shadow_type", "invalid shadow type")

        source_light_id = shadow["source_light_id"]
        direction = shadow["direction_azimuth_deg"]
        if shadow_type == "directional":
            if (
                not isinstance(source_light_id, str)
                or source_light_id not in active_light_ids
            ):
                row_error(
                    errors,
                    line,
                    f"{prefix}.source_light_id",
                    "directional shadow must reference an active light_id",
                )
            elif source_light_id in seen_directional_sources:
                row_error(
                    errors,
                    line,
                    f"{prefix}.source_light_id",
                    "duplicate directional shadow for active light",
                )
            else:
                seen_directional_sources.add(source_light_id)
            if not json_number(
                direction,
                minimum=0.0,
                maximum=360.0,
                maximum_exclusive=True,
            ):
                row_error(
                    errors,
                    line,
                    f"{prefix}.direction_azimuth_deg",
                    "directional shadow requires a value in [0, 360)",
                )
        else:
            if source_light_id is not None:
                row_error(
                    errors,
                    line,
                    f"{prefix}.source_light_id",
                    "contact/uncontrolled shadow requires null",
                )
            if direction is not None:
                row_error(
                    errors,
                    line,
                    f"{prefix}.direction_azimuth_deg",
                    "contact/uncontrolled shadow requires null",
                )

        if not json_number(shadow["length_mm"], minimum=0.0):
            row_error(errors, line, f"{prefix}.length_mm", "must be >= 0")
        if not json_number(
            shadow["contrast_ratio"], minimum=0.0, maximum=1.0
        ):
            row_error(
                errors,
                line,
                f"{prefix}.contrast_ratio",
                "must be in [0, 1]",
            )
        method = shadow["measurement_method"]
        if not isinstance(method, str) or not method.strip():
            row_error(
                errors,
                line,
                f"{prefix}.measurement_method",
                "required non-empty string",
            )
        elif any(ord(character) < 32 for character in method):
            row_error(
                errors,
                line,
                f"{prefix}.measurement_method",
                "control characters are not allowed",
            )
        elif method.lstrip().startswith(("=", "+", "-", "@")):
            row_error(
                errors,
                line,
                f"{prefix}.measurement_method",
                "spreadsheet formula prefix is not allowed",
            )

        normalized = dict(shadow)
        for field in ("direction_azimuth_deg", "length_mm", "contrast_ratio"):
            if normalized[field] is not None and json_number(normalized[field]):
                normalized[field] = float(normalized[field])
        normalized_shadows.append(normalized)

    canonical = json.dumps(
        sorted(normalized_shadows, key=lambda shadow: str(shadow.get("shadow_id", ""))),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return value, canonical


def parse_defect_classes(
    row: dict[str, str], line: int, errors: list[str]
) -> list[str] | None:
    try:
        value = json.loads(
            row["visible_defect_classes_json"],
            object_pairs_hook=reject_duplicate_json_keys,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        row_error(
            errors,
            line,
            "visible_defect_classes_json",
            f"invalid strict JSON: {exc}",
        )
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        row_error(
            errors,
            line,
            "visible_defect_classes_json",
            "must be a JSON array of strings",
        )
        return None
    if len(set(value)) != len(value):
        row_error(errors, line, "visible_defect_classes_json", "duplicate class")
    unknown = sorted(set(value) - DEFECT_CLASSES)
    if unknown:
        row_error(
            errors,
            line,
            "visible_defect_classes_json",
            f"unknown classes: {unknown}",
        )
    return value


def validate_path_and_format(
    row: dict[str, str], line: int, errors: list[str]
) -> None:
    raw = row["filename"].strip()
    path = PurePosixPath(raw)
    if (
        not raw
        or "\\" in raw
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or ":" in path.parts[0]
    ):
        row_error(
            errors,
            line,
            "filename",
            "must be a safe relative POSIX path without dot segments",
        )
        return
    extensions = {
        "jpg": {".jpg"},
        "jpeg": {".jpeg", ".jpg"},
        "png": {".png"},
        "tif": {".tif"},
        "tiff": {".tiff", ".tif"},
        "bmp": {".bmp"},
        "dng": {".dng"},
        "raw": {".raw"},
    }
    image_format = row["image_format"].strip().lower()
    if image_format not in extensions:
        row_error(errors, line, "image_format", "unsupported image format")
    elif path.suffix.lower() not in extensions[image_format]:
        row_error(errors, line, "filename", "extension does not match image_format")


def validate_calibration_window(
    row: dict[str, str],
    line: int,
    prefix: str,
    captured: datetime | None,
    measured: datetime | None,
    errors: list[str],
) -> None:
    calibrated = parse_date_value(row, line, f"{prefix}_calibration_date", errors)
    due = parse_date_value(row, line, f"{prefix}_calibration_due_date", errors)
    if calibrated is not None and due is not None and due < calibrated:
        row_error(errors, line, f"{prefix}_calibration_due_date", "precedes calibration")
    if captured is not None and calibrated is not None and captured.date() < calibrated:
        row_error(errors, line, f"{prefix}_calibration_date", "is after capture")
    if captured is not None and due is not None and captured.date() > due:
        row_error(errors, line, f"{prefix}_calibration_due_date", "expired at capture")
    if measured is not None and calibrated is not None and measured.date() < calibrated:
        row_error(
            errors,
            line,
            f"{prefix}_calibration_date",
            "is after measurement timestamp",
        )
    if measured is not None and due is not None and measured.date() > due:
        row_error(
            errors,
            line,
            f"{prefix}_calibration_due_date",
            "expired at measurement timestamp",
        )


def validate_row(
    row: dict[str, str], line: int, errors: list[str]
) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for field in COLUMNS:
        if field not in OPTIONAL_FIELDS and not row[field].strip():
            row_error(errors, line, field, "required value is blank")

    if row["schema_version"].strip() != SCHEMA_VERSION:
        row_error(errors, line, "schema_version", f"must be {SCHEMA_VERSION}")
    for field in ID_FIELDS:
        if not ID_RE.fullmatch(row[field].strip()):
            row_error(errors, line, field, "invalid identifier")

    for field in TEXT_FIELDS:
        value = row[field]
        if not value and field not in OPTIONAL_FIELDS:
            continue
        if any(ord(character) < 32 for character in value):
            row_error(errors, line, field, "control characters are not allowed")
        if value.lstrip().startswith(("=", "+", "-", "@")):
            row_error(errors, line, field, "spreadsheet formula prefix is not allowed")

    validate_path_and_format(row, line, errors)
    if not SHA256_RE.fullmatch(row["file_sha256"].strip()):
        row_error(errors, line, "file_sha256", "must be 64 lowercase hex characters")
    for field in (
        "lux_measurement_evidence_sha256",
        "cct_measurement_evidence_sha256",
    ):
        if not SHA256_RE.fullmatch(row[field].strip()):
            row_error(errors, line, field, "must be 64 lowercase hex characters")

    captured = parse_utc_timestamp(row, line, "captured_at_utc", errors)
    parsed["captured"] = captured
    lux_timestamp = parse_utc_timestamp(
        row, line, "lux_measurement_timestamp_utc", errors
    )
    cct_timestamp = parse_utc_timestamp(
        row, line, "cct_measurement_timestamp_utc", errors
    )
    parsed["lux_timestamp"] = lux_timestamp
    parsed["cct_timestamp"] = cct_timestamp

    enum_rules = {
        "source_domain": {"REAL_CAPTURE"},
        "illuminance_value_kind": {"MEASURED"},
        "specimen_status": {"OK_confirmed", "NG_confirmed", "HOLD_unverified"},
        "visible_defect_status": {
            "confirmed",
            "review",
            "none_visible",
            "unobservable",
        },
        "split": {"train", "validation", "test", "hold"},
        "evaluation_eligible": {"YES", "NO"},
        "background_color": {"black"},
        "exposure_mode": {"manual", "auto", "semi_auto"},
        "white_balance_mode": {"manual", "custom", "preset", "auto"},
        "focus_mode": {"manual", "fixed", "auto"},
        "meter_measurement_geometry": {
            "specimen_removed_at_origin",
            "adjacent_in_plane",
            "in_situ_probe",
        },
        "multi_light_mode": {"single", "simultaneous"},
        "component_only_illumination": {"YES"},
        "direct_belt_illumination": {"NO"},
        "shadow_present": {"YES", "NO"},
        "shadow_type": {
            "none",
            "contact",
            "directional",
            "contact_and_directional",
            "uncontrolled",
        },
        "capture_qc_status": {"PASS", "FAIL", "HOLD"},
        "label_review_status": {"confirmed", "provisional", "hold"},
    }
    for field, allowed in enum_rules.items():
        if row[field].strip() not in allowed:
            row_error(errors, line, field, f"must be one of {sorted(allowed)}")

    defects = parse_defect_classes(row, line, errors)
    if defects is not None:
        visible_status = row["visible_defect_status"].strip()
        if visible_status == "none_visible" and defects:
            row_error(
                errors,
                line,
                "visible_defect_classes_json",
                "must be [] when visible_defect_status=none_visible",
            )
        if visible_status == "unobservable" and defects:
            row_error(
                errors,
                line,
                "visible_defect_classes_json",
                "must be [] when visible_defect_status=unobservable",
            )
        if visible_status == "confirmed" and not defects:
            row_error(
                errors,
                line,
                "visible_defect_classes_json",
                "confirmed visible defect requires at least one class",
            )
        if row["specimen_status"].strip() == "OK_confirmed" and defects:
            row_error(
                errors,
                line,
                "visible_defect_classes_json",
                "OK_confirmed cannot contain a visible defect class",
            )

    split = row["split"].strip()
    eligible = row["evaluation_eligible"].strip()
    if eligible == "YES" and split not in {"validation", "test"}:
        row_error(
            errors,
            line,
            "evaluation_eligible",
            "YES is permitted only for validation/test",
        )
    if split in {"train", "hold"} and eligible != "NO":
        row_error(errors, line, "evaluation_eligible", "must be NO for train/hold")
    if eligible == "YES":
        if row["specimen_status"].strip() == "HOLD_unverified":
            row_error(errors, line, "specimen_status", "HOLD cannot be evaluation eligible")
        if row["label_review_status"].strip() != "confirmed":
            row_error(
                errors,
                line,
                "label_review_status",
                "evaluation-eligible row must be confirmed",
            )
        if row["capture_qc_status"].strip() != "PASS":
            row_error(
                errors,
                line,
                "capture_qc_status",
                "evaluation-eligible row must pass capture QC",
            )
        visible_status = row["visible_defect_status"].strip()
        specimen_status = row["specimen_status"].strip()
        if visible_status in {"review", "unobservable"}:
            row_error(
                errors,
                line,
                "visible_defect_status",
                "evaluation-eligible row must have a confirmed visible state",
            )
        if specimen_status == "OK_confirmed" and visible_status != "none_visible":
            row_error(
                errors,
                line,
                "visible_defect_status",
                "evaluation-eligible OK requires none_visible",
            )
        if specimen_status == "NG_confirmed" and visible_status != "confirmed":
            row_error(
                errors,
                line,
                "visible_defect_status",
                "evaluation-eligible NG requires a confirmed visible defect",
            )

    numeric_specs = {
        "measurement_capture_offset_limit_seconds": (False, 0.0, None, False),
        "focal_length_mm": (False, 0.000001, None, False),
        "camera_azimuth_deg": (False, 0.0, 360.0, True),
        "camera_elevation_deg": (False, 0.0, 90.0, False),
        "camera_distance_mm": (False, 0.000001, None, False),
        "image_width_px": (True, 1.0, None, False),
        "image_height_px": (True, 1.0, None, False),
        "bit_depth": (True, 1.0, 32.0, False),
        "exposure_time_us": (False, 0.000001, None, False),
        "aperture_f_number": (False, 0.000001, None, False),
        "exposure_compensation_ev": (False, None, None, False),
        "meter_x_mm": (False, None, None, False),
        "meter_y_mm": (False, None, None, False),
        "meter_z_mm": (False, None, None, False),
        "meter_sensor_normal_azimuth_deg": (False, 0.0, 360.0, True),
        "meter_sensor_normal_elevation_deg": (False, -90.0, 90.0, False),
        "illuminance_lux_measured": (False, 0.0, None, False),
        "illuminance_repeat_count": (True, 1.0, None, False),
        "illuminance_std_lux": (False, 0.0, None, False),
        "cct_measured_k": (False, 0.000001, None, False),
        "light_count": (True, 1.0, 8.0, False),
        "minimum_active_light_angular_separation_deg": (
            False,
            0.0,
            180.0,
            False,
        ),
        "component_clipped_high_pct": (False, 0.0, 100.0, False),
        "component_clipped_low_pct": (False, 0.0, 100.0, False),
    }
    for field, (integer, minimum, maximum, exclusive) in numeric_specs.items():
        parsed[field] = parse_number(
            row,
            line,
            field,
            errors,
            integer=integer,
            minimum=minimum,
            maximum=maximum,
            maximum_exclusive=exclusive,
        )
    angular_threshold = parsed.get("minimum_active_light_angular_separation_deg")
    if angular_threshold is not None and angular_threshold <= 0:
        row_error(
            errors,
            line,
            "minimum_active_light_angular_separation_deg",
            "must be > 0; the project requires genuinely different light angles",
        )

    offset_limit = parsed.get("measurement_capture_offset_limit_seconds")
    if captured is not None and offset_limit is not None:
        for field, measured in (
            ("lux_measurement_timestamp_utc", lux_timestamp),
            ("cct_measurement_timestamp_utc", cct_timestamp),
        ):
            if measured is not None:
                offset = abs((measured - captured).total_seconds())
                if offset > float(offset_limit):
                    row_error(
                        errors,
                        line,
                        field,
                        f"capture offset {offset:.6f}s exceeds declared limit "
                        f"{float(offset_limit):.6f}s",
                    )

    iso = parse_number(row, line, "iso", errors, minimum=0.000001, optional=True)
    gain = parse_number(row, line, "analog_gain_db", errors, optional=True)
    if iso is None and gain is None:
        row_error(errors, line, "iso/analog_gain_db", "at least one must be recorded")
    wb_kelvin = parse_number(
        row, line, "white_balance_kelvin", errors, minimum=0.000001, optional=True
    )
    wb_red = parse_number(
        row, line, "wb_red_gain", errors, minimum=0.000001, optional=True
    )
    wb_blue = parse_number(
        row, line, "wb_blue_gain", errors, minimum=0.000001, optional=True
    )
    if (wb_red is None) != (wb_blue is None):
        row_error(errors, line, "wb_red_gain/wb_blue_gain", "record both or neither")
    if row["white_balance_mode"].strip() in {"manual", "custom", "preset"}:
        if wb_kelvin is None:
            row_error(
                errors,
                line,
                "white_balance_kelvin",
                "required for manual/custom/preset WB",
            )

    repeat_count = parsed.get("illuminance_repeat_count")
    std_lux = parsed.get("illuminance_std_lux")
    if repeat_count == 1 and std_lux not in {None, 0, 0.0}:
        row_error(errors, line, "illuminance_std_lux", "must be 0 for one reading")

    lights_result = parse_lights(row, line, errors)
    active_light_ids: set[str] = set()
    if lights_result is not None:
        lights, canonical = lights_result
        parsed["lights_canonical"] = canonical
        parsed["active_light_count"] = len(lights)
        active_light_ids = {
            light["light_id"]
            for light in lights
            if isinstance(light, dict)
            and isinstance(light.get("light_id"), str)
            and ID_RE.fullmatch(light["light_id"])
        }
        if parsed.get("light_count") is not None and parsed["light_count"] != len(lights):
            row_error(errors, line, "light_count", "does not match lights_json length")
        mode = row["multi_light_mode"].strip()
        if len(lights) == 1 and mode != "single":
            row_error(errors, line, "multi_light_mode", "one light requires single")
        if len(lights) > 1 and mode != "simultaneous":
            row_error(
                errors,
                line,
                "multi_light_mode",
                "multiple lights require simultaneous",
            )
        if len(lights) > 1 and angular_threshold is not None:
            maximum_separation = maximum_light_angular_separation(lights)
            if (
                maximum_separation is None
                or maximum_separation + 1e-9 < float(angular_threshold)
            ):
                actual = "unavailable" if maximum_separation is None else f"{maximum_separation:.6f}"
                row_error(
                    errors,
                    line,
                    "lights_json",
                    f"maximum active-light angular separation {actual}deg is below "
                    f"declared minimum {float(angular_threshold):.6f}deg",
                )

    if (
        row["component_only_illumination"].strip() == "YES"
        and row["direct_belt_illumination"].strip() != "NO"
    ):
        row_error(
            errors,
            line,
            "direct_belt_illumination",
            "must be NO when component_only_illumination=YES",
        )

    shadow_present = row["shadow_present"].strip()
    shadow_type = row["shadow_type"].strip()
    shadows_result = parse_shadows(row, line, active_light_ids, errors)
    if shadows_result is not None:
        shadows, shadows_canonical = shadows_result
        parsed["shadows_canonical"] = shadows_canonical
        parsed["shadow_count"] = len(shadows)
        if shadow_present == "NO":
            if shadow_type != "none":
                row_error(errors, line, "shadow_type", "NO requires none")
            if shadows:
                row_error(errors, line, "shadows_json", "must be [] when no shadow")
        elif shadow_present == "YES":
            if not shadows:
                row_error(errors, line, "shadows_json", "present shadow requires entries")
            shadow_types = {
                shadow.get("shadow_type")
                for shadow in shadows
                if isinstance(shadow, dict)
            }
            if "uncontrolled" in shadow_types and len(shadow_types) > 1:
                row_error(
                    errors,
                    line,
                    "shadows_json",
                    "uncontrolled cannot be combined with contact/directional entries",
                )
            if shadow_types == {"contact"}:
                expected_summary = "contact"
            elif shadow_types == {"directional"}:
                expected_summary = "directional"
            elif shadow_types == {"contact", "directional"}:
                expected_summary = "contact_and_directional"
            elif shadow_types == {"uncontrolled"}:
                expected_summary = "uncontrolled"
            else:
                expected_summary = None
            if expected_summary is None or shadow_type != expected_summary:
                row_error(
                    errors,
                    line,
                    "shadow_type",
                    f"must summarize shadows_json as {expected_summary}",
                )

    focus_name = row["focus_metric_name"].strip()
    focus_value = parse_number(
        row, line, "focus_metric_value", errors, minimum=0.0, optional=True
    )
    if bool(focus_name) != (focus_value is not None):
        row_error(errors, line, "focus metric", "record both metric name and value or neither")

    validate_calibration_window(
        row, line, "lux_meter", captured, lux_timestamp, errors
    )
    validate_calibration_window(
        row, line, "cct_meter", captured, cct_timestamp, errors
    )
    label_date = parse_date_value(row, line, "label_review_date", errors)
    if captured is not None and label_date is not None and label_date < captured.date():
        row_error(errors, line, "label_review_date", "precedes capture")

    parsed["condition_signature"] = (
        row["lighting_profile_id"].strip(),
        row["camera_profile_id"].strip(),
        row["background_id"].strip(),
    )
    parsed["camera_signature"] = tuple(
        row[field].strip()
        for field in (
            "camera_make",
            "camera_model",
            "camera_serial",
            "lens_model",
            "focal_length_mm",
            "camera_azimuth_deg",
            "camera_elevation_deg",
            "camera_distance_mm",
            "image_width_px",
            "image_height_px",
            "bit_depth",
            "image_format",
            "exposure_mode",
            "exposure_time_us",
            "iso",
            "analog_gain_db",
            "aperture_f_number",
            "exposure_compensation_ev",
            "white_balance_mode",
            "white_balance_kelvin",
            "wb_red_gain",
            "wb_blue_gain",
            "focus_mode",
        )
    )
    parsed["lighting_signature"] = (
        parsed.get("lights_canonical"),
        row["multi_light_mode"].strip(),
        row["component_only_illumination"].strip(),
        row["direct_belt_illumination"].strip(),
        row["stray_light_control"].strip(),
        row["station_id"].strip(),
        row["meter_measurement_geometry"].strip(),
        row["meter_location_frame"].strip(),
        row["meter_x_mm"].strip(),
        row["meter_y_mm"].strip(),
        row["meter_z_mm"].strip(),
        row["meter_sensor_normal_azimuth_deg"].strip(),
        row["meter_sensor_normal_elevation_deg"].strip(),
        float(angular_threshold) if angular_threshold is not None else None,
    )
    parsed["session_policy_signature"] = (
        float(offset_limit) if offset_limit is not None else None,
        float(angular_threshold) if angular_threshold is not None else None,
    )
    parsed["lux_evidence_signature"] = (
        row["lux_measurement_evidence_sha256"].strip(),
        row["lux_meter_id"].strip(),
        row["lux_meter_serial"].strip(),
        lux_timestamp.isoformat() if lux_timestamp is not None else None,
        parsed.get("illuminance_lux_measured"),
        parsed.get("illuminance_repeat_count"),
        parsed.get("illuminance_std_lux"),
        row["meter_measurement_geometry"].strip(),
        row["meter_location_frame"].strip(),
        parsed.get("meter_x_mm"),
        parsed.get("meter_y_mm"),
        parsed.get("meter_z_mm"),
        parsed.get("meter_sensor_normal_azimuth_deg"),
        parsed.get("meter_sensor_normal_elevation_deg"),
    )
    parsed["cct_evidence_signature"] = (
        row["cct_measurement_evidence_sha256"].strip(),
        row["cct_meter_id"].strip(),
        row["cct_meter_serial"].strip(),
        cct_timestamp.isoformat() if cct_timestamp is not None else None,
        parsed.get("cct_measured_k"),
        row["meter_measurement_geometry"].strip(),
        row["meter_location_frame"].strip(),
        parsed.get("meter_x_mm"),
        parsed.get("meter_y_mm"),
        parsed.get("meter_z_mm"),
        parsed.get("meter_sensor_normal_azimuth_deg"),
        parsed.get("meter_sensor_normal_elevation_deg"),
    )
    return parsed


def check_consistent_mapping(
    rows: list[tuple[int, dict[str, str], dict[str, Any]]],
    key_field: str,
    value_fields: tuple[str, ...],
    errors: list[str],
) -> None:
    seen: dict[str, tuple[tuple[str, ...], int]] = {}
    for line, row, _ in rows:
        key = row[key_field].strip()
        value = tuple(row[field].strip() for field in value_fields)
        prior = seen.get(key)
        if prior is None:
            seen[key] = (value, line)
        elif prior[0] != value:
            row_error(
                errors,
                line,
                key_field,
                f"conflicts with line {prior[1]} for fields {list(value_fields)}",
            )


def validate_dataset_rows(
    source_rows: list[tuple[int, dict[str, str]]], errors: list[str]
) -> list[tuple[int, dict[str, str], dict[str, Any]]]:
    parsed_rows = [
        (line, row, validate_row(row, line, errors)) for line, row in source_rows
    ]

    for unique_field in ("capture_id", "filename", "file_sha256"):
        seen: dict[str, int] = {}
        for line, row, _ in parsed_rows:
            value = row[unique_field].strip()
            if value in seen:
                row_error(
                    errors,
                    line,
                    unique_field,
                    f"duplicate of line {seen[value]}",
                )
            else:
                seen[value] = line

    check_consistent_mapping(
        parsed_rows,
        "specimen_id",
        (
            "specimen_group_id",
            "production_lot_id",
            "specimen_status",
            "split_group_id",
            "split",
        ),
        errors,
    )
    check_consistent_mapping(
        parsed_rows, "specimen_group_id", ("split_group_id", "split"), errors
    )
    check_consistent_mapping(parsed_rows, "split_group_id", ("split",), errors)
    check_consistent_mapping(parsed_rows, "session_group_id", ("split",), errors)
    check_consistent_mapping(
        parsed_rows,
        "session_id",
        ("session_group_id", "site_id", "station_id", "split"),
        errors,
    )

    if not any(
        row["multi_light_mode"].strip() == "simultaneous"
        and int(parsed.get("active_light_count", 0)) >= 2
        for _, row, parsed in parsed_rows
    ):
        errors.append(
            "manifest coverage: at least one simultaneous multi-light row is required"
        )
    if not any(
        row["shadow_present"].strip() == "YES"
        and int(parsed.get("shadow_count", 0)) >= 1
        for _, row, parsed in parsed_rows
    ):
        errors.append("manifest coverage: at least one measured shadow row is required")

    signature_maps: dict[str, dict[str, tuple[Any, int]]] = {
        "camera_profile_id": {},
        "lighting_profile_id": {},
        "condition_id": {},
        "session_id": {},
        "session_group_id": {},
        "lux_measurement_evidence_id": {},
        "cct_measurement_evidence_id": {},
    }
    signature_names = {
        "camera_profile_id": "camera_signature",
        "lighting_profile_id": "lighting_signature",
        "condition_id": "condition_signature",
        "session_id": "session_policy_signature",
        "session_group_id": "session_policy_signature",
        "lux_measurement_evidence_id": "lux_evidence_signature",
        "cct_measurement_evidence_id": "cct_evidence_signature",
    }
    for line, row, parsed in parsed_rows:
        for id_field, signature_name in signature_names.items():
            identifier = row[id_field].strip()
            signature = parsed.get(signature_name)
            prior = signature_maps[id_field].get(identifier)
            if prior is None:
                signature_maps[id_field][identifier] = (signature, line)
            elif prior[0] != signature:
                row_error(
                    errors,
                    line,
                    id_field,
                    f"reused with a different setup than line {prior[1]}",
                )
    return parsed_rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_files(
    rows: list[tuple[int, dict[str, str], dict[str, Any]]],
    images_root: Path,
    errors: list[str],
) -> None:
    root = images_root.resolve()
    if not root.is_dir():
        errors.append(f"images root is not a directory: {root}")
        return
    for line, row, _ in rows:
        relative = Path(*PurePosixPath(row["filename"].strip()).parts)
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            row_error(errors, line, "filename", "resolves outside images root")
            continue
        if not candidate.is_file():
            row_error(errors, line, "filename", f"file not found: {candidate}")
            continue
        actual = sha256(candidate)
        if actual != row["file_sha256"].strip():
            row_error(errors, line, "file_sha256", f"mismatch for {candidate}")


def read_manifest(
    path: Path,
) -> tuple[list[str], list[tuple[int, dict[str, str]]], list[str]]:
    errors: list[str] = []
    if not path.is_file():
        return [], [], [f"manifest not found: {path}"]
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            raw_rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        return [], [], [f"cannot read manifest: {exc}"]
    if not raw_rows:
        return [], [], ["manifest is empty; exact header is required"]

    header = raw_rows[0]
    if tuple(header) != COLUMNS:
        errors.append("manifest header/order is not the exact public schema")
        expected = list(COLUMNS)
        errors.append(f"expected columns={expected}")
        errors.append(f"actual columns={header}")
        return header, [], errors

    rows: list[tuple[int, dict[str, str]]] = []
    for line, values in enumerate(raw_rows[1:], start=2):
        if len(values) != len(COLUMNS):
            errors.append(
                f"line {line}: expected {len(COLUMNS)} fields, got {len(values)}"
            )
            continue
        rows.append((line, dict(zip(COLUMNS, values, strict=True))))
    return header, rows, errors


def make_self_test_row(
    capture_index: int, *, simultaneous: bool, with_shadow: bool
) -> dict[str, str]:
    row = {column: "X" for column in COLUMNS}
    for field in (
        "focal_length_mm",
        "camera_azimuth_deg",
        "camera_elevation_deg",
        "camera_distance_mm",
        "image_width_px",
        "image_height_px",
        "bit_depth",
        "exposure_time_us",
        "aperture_f_number",
        "exposure_compensation_ev",
        "meter_x_mm",
        "meter_y_mm",
        "meter_z_mm",
        "meter_sensor_normal_azimuth_deg",
        "meter_sensor_normal_elevation_deg",
        "component_clipped_high_pct",
        "component_clipped_low_pct",
    ):
        row[field] = "1"

    suffix = str(capture_index)
    row.update(
        {
            "schema_version": SCHEMA_VERSION,
            "source_domain": "REAL_CAPTURE",
            "illuminance_value_kind": "MEASURED",
            "capture_id": f"CAP-{suffix}",
            "filename": f"train/CAP-{suffix}.jpg",
            "file_sha256": ("a" if capture_index == 1 else "b") * 64,
            "captured_at_utc": "2026-09-01T00:10:00Z",
            "measurement_capture_offset_limit_seconds": "900",
            "site_id": "SITE-1",
            "station_id": "STATION-1",
            "operator_id": "OP-1",
            "session_id": "SESSION-1",
            "session_group_id": "SESSION-G1",
            "condition_id": f"COND-{suffix}",
            "lighting_profile_id": f"LIGHT-P{suffix}",
            "camera_profile_id": "CAM-P1",
            "specimen_id": f"SPEC-{suffix}",
            "specimen_group_id": f"SPEC-G{suffix}",
            "production_lot_id": "LOT-1",
            "view_id": "FRONT",
            "specimen_status": "OK_confirmed",
            "visible_defect_status": "none_visible",
            "visible_defect_classes_json": "[]",
            "split": "train",
            "split_group_id": f"SPLIT-G{suffix}",
            "evaluation_eligible": "NO",
            "background_id": "BELT-1",
            "background_color": "black",
            "background_material": "matte rubber",
            "background_surface_state": "clean",
            "camera_make": "Example",
            "camera_model": "CAM1",
            "camera_serial": "CAM-S1",
            "lens_model": "LENS1",
            "focal_length_mm": "25",
            "camera_azimuth_deg": "0",
            "camera_elevation_deg": "90",
            "camera_distance_mm": "300",
            "image_width_px": "1920",
            "image_height_px": "1080",
            "bit_depth": "8",
            "image_format": "jpg",
            "exposure_mode": "manual",
            "exposure_time_us": "4000",
            "iso": "100",
            "analog_gain_db": "",
            "aperture_f_number": "4",
            "exposure_compensation_ev": "0",
            "white_balance_mode": "manual",
            "white_balance_kelvin": "5000",
            "wb_red_gain": "",
            "wb_blue_gain": "",
            "focus_mode": "manual",
            "lux_meter_id": "LUX-1",
            "lux_meter_serial": "LUX-S1",
            "lux_meter_calibration_date": "2026-01-01",
            "lux_meter_calibration_due_date": "2027-01-01",
            "lux_meter_calibration_certificate": "CERT-LUX-1",
            "lux_measurement_evidence_id": f"LUX-E{suffix}",
            "lux_measurement_evidence_sha256": (
                "c" if capture_index == 1 else "d"
            )
            * 64,
            "meter_measurement_geometry": "specimen_removed_at_origin",
            "meter_location_frame": "STATION-1",
            "meter_x_mm": "0",
            "meter_y_mm": "0",
            "meter_z_mm": "0",
            "meter_sensor_normal_azimuth_deg": "0",
            "meter_sensor_normal_elevation_deg": "90",
            "lux_measurement_timestamp_utc": "2026-09-01T00:00:00Z",
            "illuminance_lux_measured": "500",
            "illuminance_repeat_count": "3",
            "illuminance_std_lux": "2",
            "cct_meter_id": "CCT-1",
            "cct_meter_serial": "CCT-S1",
            "cct_meter_calibration_date": "2026-01-01",
            "cct_meter_calibration_due_date": "2027-01-01",
            "cct_meter_calibration_certificate": "CERT-CCT-1",
            "cct_measurement_evidence_id": f"CCT-E{suffix}",
            "cct_measurement_evidence_sha256": (
                "e" if capture_index == 1 else "f"
            )
            * 64,
            "cct_measurement_timestamp_utc": "2026-09-01T00:01:00Z",
            "cct_measured_k": "5000",
            "minimum_active_light_angular_separation_deg": "30",
            "component_only_illumination": "YES",
            "direct_belt_illumination": "NO",
            "stray_light_control": "hood and baffle",
            "component_clipped_high_pct": "0",
            "component_clipped_low_pct": "0",
            "focus_metric_name": "",
            "focus_metric_value": "",
            "capture_qc_status": "PASS",
            "capture_qc_notes": "",
            "label_review_status": "confirmed",
            "label_reviewer": "REVIEWER-1",
            "label_review_date": "2026-09-01",
        }
    )

    lights = [
        {
            "light_id": "L1",
            "role": "key",
            "source_type": "LED",
            "azimuth_deg": 0.0,
            "elevation_deg": 30.0,
            "distance_mm": 300.0,
            "nominal_cct_k": 5000.0,
            "power_setting_pct": 70.0,
            "diffuser_id": "DIFF-A",
            "polarizer_angle_deg": None,
        }
    ]
    if simultaneous:
        lights.append(
            {
                "light_id": "L2",
                "role": "fill",
                "source_type": "LED",
                "azimuth_deg": 90.0,
                "elevation_deg": 30.0,
                "distance_mm": 320.0,
                "nominal_cct_k": 5000.0,
                "power_setting_pct": 30.0,
                "diffuser_id": "DIFF-B",
                "polarizer_angle_deg": None,
            }
        )
    row["light_count"] = str(len(lights))
    row["multi_light_mode"] = "simultaneous" if simultaneous else "single"
    row["lights_json"] = json.dumps(lights, separators=(",", ":"))

    if with_shadow:
        row["shadow_present"] = "YES"
        row["shadow_type"] = "directional"
        row["shadows_json"] = json.dumps(
            [
                {
                    "shadow_id": "SHADOW-1",
                    "source_light_id": "L1",
                    "shadow_type": "directional",
                    "direction_azimuth_deg": 180.0,
                    "length_mm": 5.0,
                    "contrast_ratio": 0.4,
                    "measurement_method": "image_roi_linear_luma",
                }
            ],
            separators=(",", ":"),
        )
    else:
        row["shadow_present"] = "NO"
        row["shadow_type"] = "none"
        row["shadows_json"] = "[]"
    return row


def run_self_tests() -> int:
    failures: list[str] = []

    def validate(rows: list[dict[str, str]]) -> list[str]:
        errors: list[str] = []
        validate_dataset_rows(
            [(index + 2, row) for index, row in enumerate(rows)], errors
        )
        return errors

    valid_rows = [
        make_self_test_row(1, simultaneous=False, with_shadow=False),
        make_self_test_row(2, simultaneous=True, with_shadow=True),
    ]
    positive_errors = validate(valid_rows)
    if positive_errors:
        failures.append(f"positive fixture failed: {positive_errors}")

    def expect_error(name: str, rows: list[dict[str, str]], fragment: str) -> None:
        errors = validate(rows)
        if not any(fragment in error for error in errors):
            failures.append(f"{name}: missing error containing {fragment!r}; got {errors}")

    component_gate = [dict(row) for row in valid_rows]
    component_gate[0]["component_only_illumination"] = "NO"
    expect_error(
        "component-only hard gate",
        component_gate,
        "component_only_illumination",
    )

    belt_gate = [dict(row) for row in valid_rows]
    belt_gate[0]["direct_belt_illumination"] = "YES"
    expect_error("direct-belt hard gate", belt_gate, "direct_belt_illumination")

    source_gate = [dict(row) for row in valid_rows]
    source_gate[0]["source_domain"] = "SYNTHETIC_PROXY"
    expect_error("real source-domain hard gate", source_gate, "source_domain")

    value_kind_gate = [dict(row) for row in valid_rows]
    value_kind_gate[0]["illuminance_value_kind"] = "PROXY"
    expect_error(
        "measured illuminance-kind hard gate",
        value_kind_gate,
        "illuminance_value_kind",
    )

    offset_gate = [dict(row) for row in valid_rows]
    offset_gate[0]["lux_measurement_timestamp_utc"] = "2026-09-01T01:00:00Z"
    expect_error("capture offset gate", offset_gate, "exceeds declared limit")

    calibration_gate = [dict(row) for row in valid_rows]
    calibration_gate[0]["lux_measurement_timestamp_utc"] = "2025-12-31T23:59:00Z"
    calibration_gate[0]["measurement_capture_offset_limit_seconds"] = "99999999"
    calibration_gate[1]["measurement_capture_offset_limit_seconds"] = "99999999"
    expect_error(
        "measurement calibration gate",
        calibration_gate,
        "is after measurement timestamp",
    )

    angular_gate = [dict(row) for row in valid_rows]
    angular_lights = json.loads(angular_gate[1]["lights_json"])
    angular_lights[1]["azimuth_deg"] = angular_lights[0]["azimuth_deg"]
    angular_lights[1]["elevation_deg"] = angular_lights[0]["elevation_deg"]
    angular_gate[1]["lights_json"] = json.dumps(
        angular_lights, separators=(",", ":")
    )
    expect_error("angular separation gate", angular_gate, "below declared minimum")

    expect_error(
        "multi-light coverage gate",
        [make_self_test_row(1, simultaneous=False, with_shadow=True)],
        "simultaneous multi-light row",
    )
    expect_error(
        "shadow coverage gate",
        [make_self_test_row(1, simultaneous=True, with_shadow=False)],
        "measured shadow row",
    )

    shadow_source_gate = [dict(row) for row in valid_rows]
    shadows = json.loads(shadow_source_gate[1]["shadows_json"])
    shadows[0]["source_light_id"] = "UNKNOWN"
    shadow_source_gate[1]["shadows_json"] = json.dumps(
        shadows, separators=(",", ":")
    )
    expect_error(
        "shadow source genealogy gate",
        shadow_source_gate,
        "must reference an active light_id",
    )

    session_gate = [dict(row) for row in valid_rows]
    session_gate[1]["measurement_capture_offset_limit_seconds"] = "901"
    expect_error("session policy gate", session_gate, "session_id")

    evidence_gate = [dict(row) for row in valid_rows]
    evidence_gate[1]["lux_measurement_evidence_id"] = evidence_gate[0][
        "lux_measurement_evidence_id"
    ]
    expect_error(
        "measurement evidence identity gate",
        evidence_gate,
        "lux_measurement_evidence_id",
    )

    if failures:
        print("FAIL: self-test")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("PASS: self-test checks=13")
    return 0


def main() -> int:
    args = parse_args()
    if args.self_test:
        if args.check_files or args.images_root is not None:
            print("FAIL")
            print("- --self-test cannot be combined with file-check options")
            return 1
        return run_self_tests()
    if args.check_files and args.images_root is None:
        print("FAIL")
        print("- --check-files requires --images-root")
        return 1
    if args.schema_only and args.check_files:
        print("FAIL")
        print("- --schema-only cannot be combined with --check-files")
        return 1

    _, rows, errors = read_manifest(args.manifest)
    if args.schema_only:
        if rows:
            errors.append(
                "--schema-only requires a header-only template with zero data rows"
            )
        if errors:
            print("FAIL")
            for error in errors:
                print(f"- {error}")
            return 1
        print(
            "PASS: schema-only "
            f"schema={SCHEMA_VERSION}, columns={len(COLUMNS)}, rows={len(rows)}"
        )
        return 0

    parsed_rows: list[tuple[int, dict[str, str], dict[str, Any]]] = []
    if not errors:
        if not rows:
            errors.append("no data rows; use --schema-only for the empty template")
        else:
            parsed_rows = validate_dataset_rows(rows, errors)
    if args.check_files and parsed_rows:
        validate_files(parsed_rows, args.images_root, errors)

    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    split_counts = Counter(row["split"] for _, row, _ in parsed_rows)
    status_counts = Counter(row["specimen_status"] for _, row, _ in parsed_rows)
    print(
        "PASS: "
        f"schema={SCHEMA_VERSION}, captures={len(parsed_rows)}, "
        f"splits={dict(sorted(split_counts.items()))}, "
        f"specimen_status={dict(sorted(status_counts.items()))}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
