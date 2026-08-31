from __future__ import annotations

import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "annotations" / "image_labels_v4.csv"
SOURCE_MANIFEST = ROOT / "annotations" / "source_image_manifest.csv"

STATUS_DIRS = {
    "OK": ROOT / "data" / "by_specimen_status" / "OK_confirmed",
    "NG": ROOT
    / "data"
    / "by_specimen_status"
    / "NOT_OK_or_unverified"
    / "NG_confirmed",
    "HOLD": ROOT
    / "data"
    / "by_specimen_status"
    / "NOT_OK_or_unverified"
    / "HOLD_unverified",
}

VISIBLE_RULES = {
    "scratch_confirmed": lambda row: row["scratch_state"] == "positive",
    "surface_spot_unknown": lambda row: row["surface_spot_state"] == "positive",
    "lead_deformation_review": lambda row: row["lead_deformation_state"] == "review",
    "no_visible_defect_on_view": lambda row: row["visible_primary"]
    == "no_visible_defect_on_view",
    "quality_hold": lambda row: row["visible_primary"] == "quality_hold",
    "multi_part_hold": lambda row: row["visible_primary"] == "multi_part_hold",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jpg_names(path: Path) -> set[str]:
    return {item.name for item in path.glob("*.jpg") if item.is_file()}


def main() -> int:
    errors: list[str] = []

    with LABELS.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 17:
        errors.append(f"expected 17 label rows, got {len(rows)}")

    ids = [row["image_id"] for row in rows]
    filenames = [row["filename"] for row in rows]
    if len(set(ids)) != len(ids):
        errors.append("duplicate image_id")
    if len(set(filenames)) != len(filenames):
        errors.append("duplicate filename")

    counts = Counter(row["specimen_status"] for row in rows)
    expected_counts = {"OK": 0, "NG": 13, "HOLD": 4}
    if dict(counts) != {"NG": 13, "HOLD": 4}:
        errors.append(f"unexpected specimen counts: {dict(counts)}")

    if any(row["normal_training_eligible"] != "NO" for row in rows):
        errors.append("normal_training_eligible must be NO for all current rows")

    if any(row["manual_reviewer"] != "OpenAI Codex" for row in rows):
        errors.append("manual_reviewer must be OpenAI Codex for all v4 rows")
    if any(row["manual_review_date"] != "2026-08-31" for row in rows):
        errors.append("unexpected manual_review_date in v4 labels")

    manual_counts = Counter(row["manual_image_label"] for row in rows)
    expected_manual_counts = {
        "HOLD_QUALITY": 4,
        "VIEW_OK_NOT_SPECIMEN_OK": 4,
        "NG_SCRATCH": 4,
        "NG_SCRATCH_SPOT": 1,
        "HOLD_MULTI_PART": 3,
        "NG_SCRATCH_SPOT_LEAD_REVIEW": 1,
    }
    if dict(manual_counts) != expected_manual_counts:
        errors.append(f"unexpected manual label counts: {dict(manual_counts)}")

    with SOURCE_MANIFEST.open(encoding="utf-8-sig", newline="") as stream:
        source_hashes = {
            row["filename"]: row["normalized_sha256"].lower()
            for row in csv.DictReader(stream)
        }

    for status, folder in STATUS_DIRS.items():
        actual = jpg_names(folder)
        expected = {
            row["filename"] for row in rows if row["specimen_status"] == status
        }
        if actual != expected:
            errors.append(
                f"{status} files differ: missing={sorted(expected-actual)}, "
                f"extra={sorted(actual-expected)}"
            )
        for filename in actual:
            expected_hash = source_hashes.get(filename)
            if expected_hash is None:
                errors.append(f"no source hash for {filename}")
            elif sha256(folder / filename) != expected_hash:
                errors.append(f"SHA-256 mismatch: {folder / filename}")

    visible_root = ROOT / "data" / "by_visible_class"
    for class_name, predicate in VISIBLE_RULES.items():
        actual = jpg_names(visible_root / class_name)
        expected = {row["filename"] for row in rows if predicate(row)}
        if actual != expected:
            errors.append(
                f"{class_name} files differ: missing={sorted(expected-actual)}, "
                f"extra={sorted(actual-expected)}"
            )

    for empty_class in (
        "breakage_confirmed",
        "discoloration_confirmed",
        "contamination_confirmed",
    ):
        if jpg_names(visible_root / empty_class):
            errors.append(f"{empty_class} must contain no JPG files")

    crop_root = ROOT / "data" / "crops_by_specimen_status"
    if len(jpg_names(crop_root / "NG_confirmed")) != 13:
        errors.append("expected 13 NG crop files")
    if len(jpg_names(crop_root / "HOLD_unverified")) != 1:
        errors.append("expected 1 HOLD crop file")

    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "PASS: "
        f"images={len(rows)}, OK={expected_counts['OK']}, "
        f"NG={expected_counts['NG']}, HOLD={expected_counts['HOLD']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
