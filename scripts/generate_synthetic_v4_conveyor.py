"""Generate a train-only black-conveyor multi-instance detection release.

The release keeps the verified v2 defect morphology and parent split fixed.  It
replays only the 168 v2 ``gradient_train`` parents, extracts each component with
a verified nominal alpha, composites five non-overlapping components onto a
dark conveyor, and applies illumination only inside the component alpha.  A
paired-clean scene is rendered with identical placement, lighting, sensor noise,
and JPEG settings so every published defect is checked again at detector input
size.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, features

import generate_synthetic_v2_700 as v2
import generate_synthetic_v3_conditions as v3


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "synthetic_v4_conveyor.json"
DEFAULT_RELEASE = ROOT / "synthetic" / "v4_conveyor"
MARKER = ".synthetic_v4_conveyor_marker"


@dataclass
class ParentAsset:
    row: dict[str, str]
    defect: Image.Image
    clean: Image.Image
    semantic: Image.Image
    nominal_alpha: Image.Image


@dataclass
class Context:
    config: dict[str, Any]
    config_path: Path
    config_sha256: str
    release_root: Path
    v2_config: dict[str, Any]
    source_rows: dict[str, dict[str, str]]
    source_rows_by_class: dict[str, list[dict[str, str]]]
    source_split_rows: dict[str, dict[str, str]]
    background: Image.Image
    nominal_alpha_base: Image.Image
    plans: list[dict[str, Any]]
    parent_cache: dict[str, ParentAsset]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    material = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & 0xFFFFFFFF


def current_runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "libjpeg": str(features.version("jpg")),
        "zlib": str(features.version("zlib")),
    }


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def resolve_repository_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"path escapes repository root: {value}") from error
    return path


def verify_file(value: str, expected_sha256: str) -> Path:
    path = resolve_repository_path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch {value}: {actual} != {expected_sha256}")
    return path


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    return x0, y0, x1 - x0, y1 - y0


def bbox_xyxy(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    bbox = mask_bbox(mask)
    if bbox is None:
        return None
    x, y, width, height = bbox
    return x, y, x + width, y + height


def iou_xyxy(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    if not intersection:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / float(area_a + area_b - intersection)


def bbox_gap_xyxy(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> float:
    horizontal = max(a[0] - b[2], b[0] - a[2], 0)
    vertical = max(a[1] - b[3], b[1] - a[3], 0)
    return math.hypot(horizontal, vertical)


def coco_uncompressed_rle(mask: np.ndarray) -> dict[str, Any]:
    """Encode a full-image binary mask as standard COCO uncompressed RLE."""

    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2:
        raise ValueError("COCO RLE mask must be two-dimensional")
    flat = binary.T.reshape(-1)
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    boundaries = np.concatenate(([0], changes, [flat.size]))
    counts = np.diff(boundaries).astype(np.int64).tolist()
    if int(flat[0]) == 1:
        counts.insert(0, 0)
    return {"size": [int(binary.shape[0]), int(binary.shape[1])], "counts": counts}


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    binary = np.asarray(mask, dtype=np.uint8)
    window = radius * 2 + 1
    horizontal_padded = np.pad(binary, ((0, 0), (radius, radius)))
    horizontal_sum = np.pad(
        np.cumsum(horizontal_padded, axis=1, dtype=np.int32),
        ((0, 0), (1, 0)),
    )
    horizontal = (
        horizontal_sum[:, window:] - horizontal_sum[:, :-window]
    ) > 0
    vertical_padded = np.pad(
        horizontal.astype(np.uint8), ((radius, radius), (0, 0))
    )
    vertical_sum = np.pad(
        np.cumsum(vertical_padded, axis=0, dtype=np.int32),
        ((1, 0), (0, 0)),
    )
    return (vertical_sum[window:, :] - vertical_sum[:-window, :]) > 0


def erode_image(mask: Image.Image, radius: int) -> Image.Image:
    if radius <= 0:
        return mask.copy()
    return mask.filter(ImageFilter.MinFilter(radius * 2 + 1))


def safe_prepare_release(path: Path, force: bool) -> None:
    path = path.resolve()
    synthetic_root = (ROOT / "synthetic").resolve()
    try:
        path.relative_to(synthetic_root)
    except ValueError as error:
        raise ValueError(f"release path must be inside synthetic/: {path}") from error
    if path.parent != synthetic_root or not path.name.startswith("v4_conveyor"):
        raise ValueError(
            "release path must be a direct synthetic/v4_conveyor* child: "
            f"{path}"
        )
    if path.exists():
        marker = path / MARKER
        if not force:
            raise FileExistsError(f"release already exists; pass --force: {path}")
        if not marker.is_file():
            raise ValueError(f"refusing to replace unmarked directory: {path}")
        if marker.read_text(encoding="ascii") != "synthetic-v4-conveyor\n":
            raise ValueError(f"refusing to replace directory with invalid marker: {path}")
        shutil.rmtree(path)
    for relative in (
        "images/train",
        "labels/yolo_component_status/train",
        "labels/yolo_defects/train",
        "masks/component_visible_instances/train",
        "masks/defect_semantic/train",
        "annotations/coco",
        "annotations/contact_sheets",
    ):
        (path / relative).mkdir(parents=True, exist_ok=True)
    (path / MARKER).write_text("synthetic-v4-conveyor\n", encoding="ascii", newline="\n")


def pair_counts(groups: Iterable[Iterable[int]]) -> Counter[tuple[int, int]]:
    return Counter(
        pair for group in groups for pair in combinations(sorted(group), 2)
    )


def balanced_class_blocks(seed: int) -> list[list[int]]:
    """Return 96 five-class blocks with exact class/slot balance and near-BIBD pairs."""

    groups = [[(index * 5 + slot) % 8 for slot in range(5)] for index in range(96)]
    rng = random.Random(seed)
    target = 960.0 / 28.0
    counts = pair_counts(groups)
    score = sum(
        (counts[pair] - target) ** 2 for pair in combinations(range(8), 2)
    )
    theoretical_minimum = 5.714285714285714
    for iteration in range(500_000):
        first, second = rng.sample(range(len(groups)), 2)
        slot = rng.randrange(5)
        left = groups[first][slot]
        right = groups[second][slot]
        if left == right or right in groups[first] or left in groups[second]:
            continue
        old_first = list(combinations(sorted(groups[first]), 2))
        old_second = list(combinations(sorted(groups[second]), 2))
        candidate_first = list(groups[first])
        candidate_second = list(groups[second])
        candidate_first[slot] = right
        candidate_second[slot] = left
        new_first = list(combinations(sorted(candidate_first), 2))
        new_second = list(combinations(sorted(candidate_second), 2))
        affected = set(old_first + old_second + new_first + new_second)
        old_local = sum((counts[pair] - target) ** 2 for pair in affected)
        updated = counts.copy()
        for pair in old_first + old_second:
            updated[pair] -= 1
        for pair in new_first + new_second:
            updated[pair] += 1
        new_local = sum((updated[pair] - target) ** 2 for pair in affected)
        temperature = max(0.001, 2.0 * (1.0 - iteration / 500_000.0))
        if new_local <= old_local or rng.random() < math.exp(
            (old_local - new_local) / temperature
        ):
            groups[first] = candidate_first
            groups[second] = candidate_second
            counts = updated
            score += new_local - old_local
            if score <= theoretical_minimum + 1e-8:
                break
    class_counts = Counter(value for group in groups for value in group)
    slot_counts = {
        class_id: [
            sum(group[slot] == class_id for group in groups) for slot in range(5)
        ]
        for class_id in range(8)
    }
    counts = pair_counts(groups)
    if set(class_counts.values()) != {60}:
        raise RuntimeError(f"class block balance failed: {class_counts}")
    if any(values != [12] * 5 for values in slot_counts.values()):
        raise RuntimeError(f"class slot balance failed: {slot_counts}")
    if min(counts.values()) != 34 or max(counts.values()) != 35:
        raise RuntimeError(
            f"class pair balance failed: {min(counts.values())}..{max(counts.values())}"
        )
    return groups


def balanced_grid_assignments(
    class_groups: list[list[int]], seed: int
) -> list[list[int]]:
    """Assign five distinct cells per scene with exact 7/8 class-cell balance."""

    if len(class_groups) != 96 or any(len(group) != 5 for group in class_groups):
        raise ValueError("grid assignment expects 96 five-class scene groups")
    rng = random.Random(seed)
    cell_groups = balanced_class_blocks(stable_seed(seed, "cell-subsets"))
    rng.shuffle(cell_groups)
    assignments: list[list[int]] = []
    counts: Counter[tuple[int, int]] = Counter()
    for classes, cells in zip(class_groups, cell_groups, strict=True):
        assigned = list(cells)
        rng.shuffle(assigned)
        assignments.append(assigned)
        counts.update(zip(classes, assigned, strict=True))

    target = 7.5
    score = sum(
        (counts[(class_id, cell)] - target) ** 2
        for class_id in range(8)
        for cell in range(8)
    )
    theoretical_minimum = 16.0
    for iteration in range(500_000):
        scene_index = rng.randrange(len(class_groups))
        first, second = rng.sample(range(5), 2)
        first_class = class_groups[scene_index][first]
        second_class = class_groups[scene_index][second]
        first_cell = assignments[scene_index][first]
        second_cell = assignments[scene_index][second]
        affected = {
            (first_class, first_cell),
            (first_class, second_cell),
            (second_class, first_cell),
            (second_class, second_cell),
        }
        old_local = sum((counts[key] - target) ** 2 for key in affected)
        counts[(first_class, first_cell)] -= 1
        counts[(second_class, second_cell)] -= 1
        counts[(first_class, second_cell)] += 1
        counts[(second_class, first_cell)] += 1
        new_local = sum((counts[key] - target) ** 2 for key in affected)
        temperature = max(0.001, 1.5 * (1.0 - iteration / 500_000.0))
        if new_local <= old_local or rng.random() < math.exp(
            (old_local - new_local) / temperature
        ):
            assignments[scene_index][first] = second_cell
            assignments[scene_index][second] = first_cell
            score += new_local - old_local
            if score <= theoretical_minimum + 1e-8:
                break
        else:
            counts[(first_class, first_cell)] += 1
            counts[(second_class, second_cell)] += 1
            counts[(first_class, second_cell)] -= 1
            counts[(second_class, first_cell)] -= 1

    if any(len(set(cells)) != 5 for cells in assignments):
        raise RuntimeError("grid cell uniqueness failed")
    if set(counts.values()) != {7, 8} or len(counts) != 64:
        raise RuntimeError(f"class x grid balance failed: {Counter(counts.values())}")
    cell_counts = Counter(cell for cells in assignments for cell in cells)
    if set(cell_counts.values()) != {60}:
        raise RuntimeError(f"grid cell total balance failed: {cell_counts}")
    return assignments


def build_scene_plans(
    config: dict[str, Any], selected_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    classes = list(config["classes"])
    profiles = list(config["lighting_profiles"])
    rows_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected_rows:
        rows_by_class[row["primary_class"]].append(row)
    for class_name in classes[1:]:
        rows_by_class[class_name].sort(key=lambda item: item["sample_id"])
        if len(rows_by_class[class_name]) != 24:
            raise ValueError(f"expected 24 gradient parents for {class_name}")

    base_blocks = balanced_class_blocks(
        stable_seed(config["global_seed"], config["release"], "class-blocks")
    )
    blocks_by_profile: list[list[list[int]]] = []
    for profile_index, profile in enumerate(profiles):
        rng = random.Random(
            stable_seed(config["global_seed"], config["release"], profile)
        )
        class_permutation = list(range(8))
        rng.shuffle(class_permutation)
        order = list(range(96))
        rng.shuffle(order)
        blocks_by_profile.append(
            [
                [class_permutation[value] for value in base_blocks[index]]
                for index in order
            ]
        )

    grids_by_profile = [
        balanced_grid_assignments(
            blocks_by_profile[profile_index],
            stable_seed(config["global_seed"], config["release"], profile, "grid"),
        )
        for profile_index, profile in enumerate(profiles)
    ]

    # For every defect class, a parent appears exactly twice in each of the five
    # placement slots.  Its two profile assignments per slot rotate cyclically,
    # yielding exactly 2/3 appearances in every lighting profile and 10 total.
    defect_tokens: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    for class_name in classes[1:]:
        parents = rows_by_class[class_name]
        for parent_index, parent in enumerate(parents):
            for placement_slot in range(5):
                first_profile = (parent_index + placement_slot) % len(profiles)
                second_profile = (first_profile + 1) % len(profiles)
                defect_tokens[(class_name, first_profile, placement_slot)].append(
                    parent["sample_id"]
                )
                defect_tokens[(class_name, second_profile, placement_slot)].append(
                    parent["sample_id"]
                )
    for key, tokens in defect_tokens.items():
        if len(tokens) != 12:
            raise RuntimeError(f"unexpected defect token count for {key}: {len(tokens)}")
        token_rng = random.Random(
            stable_seed(config["global_seed"], config["release"], *key, "parents")
        )
        token_rng.shuffle(tokens)

    defect_token_counters: Counter[tuple[str, int, int]] = Counter()
    plans: list[dict[str, Any]] = []
    normal_occurrences: list[tuple[int, int]] = []
    for local_index in range(96):
        for profile_index, profile in enumerate(profiles):
            scene_index = local_index * len(profiles) + profile_index
            scene_id = f"{config['sample_id_prefix']}-{scene_index:04d}"
            class_ids = blocks_by_profile[profile_index][local_index]
            instances: list[dict[str, Any]] = []
            for placement_slot, class_id in enumerate(class_ids):
                class_name = classes[class_id]
                grid_cell = grids_by_profile[profile_index][local_index][placement_slot]
                instance_plan = {
                    "instance_index": placement_slot + 1,
                    "placement_slot": placement_slot,
                    "grid_cell": grid_cell,
                    "class_name": class_name,
                    "class_index": class_id,
                    "source_parent_sample_id": "",
                    "normal_proxy_from_paired_clean": class_name == "normal_proxy",
                }
                if class_name == "normal_proxy":
                    normal_occurrences.append((scene_index, placement_slot))
                else:
                    counter_key = (class_name, profile_index, placement_slot)
                    occurrence = defect_token_counters[counter_key]
                    defect_token_counters[counter_key] += 1
                    instance_plan["source_parent_sample_id"] = defect_tokens[
                        counter_key
                    ][occurrence]
                instances.append(instance_plan)
            plans.append(
                {
                    "scene_index": scene_index,
                    "scene_id": scene_id,
                    "image_id": scene_index + 1,
                    "lighting_profile": profile,
                    "profile_index": profile_index,
                    "profile_scene_index": local_index,
                    "scene_seed": stable_seed(
                        config["global_seed"], config["release"], scene_id
                    ),
                    "instances": instances,
                }
            )

    # Every gradient parent contributes a paired-clean normal once; 72 SHA-ranked
    # parents contribute a second time.  Assignment avoids putting a parent and
    # its own paired-clean proxy in the same scene.
    ranked_rows = sorted(
        selected_rows,
        key=lambda row: hashlib.sha256(row["sample_id"].encode("utf-8")).hexdigest(),
    )
    normal_tokens = [row["sample_id"] for row in ranked_rows]
    normal_tokens.extend(row["sample_id"] for row in ranked_rows[:72])
    token_rng = random.Random(
        stable_seed(config["global_seed"], config["release"], "normal-tokens")
    )
    token_rng.shuffle(normal_tokens)
    for scene_index, placement_slot in normal_occurrences:
        conflict_ids = {
            item["source_parent_sample_id"]
            for item in plans[scene_index]["instances"]
            if item["source_parent_sample_id"]
        }
        chosen_index = next(
            (
                index
                for index, parent_id in enumerate(normal_tokens)
                if parent_id not in conflict_ids
            ),
            None,
        )
        if chosen_index is None:
            raise RuntimeError("cannot assign non-conflicting normal proxy parent")
        parent_id = normal_tokens.pop(chosen_index)
        plans[scene_index]["instances"][placement_slot][
            "source_parent_sample_id"
        ] = parent_id
    if normal_tokens:
        raise RuntimeError("normal proxy source token inventory was not exhausted")

    defect_usage = Counter()
    defect_profile_usage = Counter()
    defect_slot_usage = Counter()
    normal_usage = Counter()
    for plan in plans:
        if len({item["class_name"] for item in plan["instances"]}) != 5:
            raise RuntimeError(f"scene classes are not distinct: {plan['scene_id']}")
        for item in plan["instances"]:
            if item["normal_proxy_from_paired_clean"]:
                normal_usage[item["source_parent_sample_id"]] += 1
            else:
                defect_usage[item["source_parent_sample_id"]] += 1
                defect_profile_usage[
                    (item["source_parent_sample_id"], plan["lighting_profile"])
                ] += 1
                defect_slot_usage[
                    (item["source_parent_sample_id"], item["placement_slot"])
                ] += 1
    if set(defect_usage.values()) != {10} or len(defect_usage) != 168:
        raise RuntimeError("defect parent reuse must be exactly 10 for all 168 parents")
    if set(defect_profile_usage.values()) != {2, 3} or len(defect_profile_usage) != 672:
        raise RuntimeError("defect parent x profile reuse must be exactly two or three")
    if set(defect_slot_usage.values()) != {2} or len(defect_slot_usage) != 840:
        raise RuntimeError("defect parent x placement slot reuse must be exactly two")
    if max(normal_usage.values()) != 2 or min(normal_usage.values()) != 1:
        raise RuntimeError("normal parent reuse must be one or two")
    return plans


def load_context(config_path: Path, release_root: Path) -> Context:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_sha256 = sha256_file(config_path)
    runtime_contract = config["runtime_contract"]
    actual_runtime = current_runtime_versions()
    expected_runtime = {
        key: str(runtime_contract[key])
        for key in ("python", "numpy", "pillow", "libjpeg", "zlib")
    }
    if actual_runtime != expected_runtime:
        raise RuntimeError(
            f"runtime contract mismatch: actual={actual_runtime} "
            f"expected={expected_runtime}"
        )
    verify_file(
        runtime_contract["requirements_path"],
        runtime_contract["requirements_sha256"],
    )
    for helper in runtime_contract["helper_scripts"]:
        verify_file(helper["path"], helper["sha256"])
    source = config["source"]
    background_config = config["background"]
    alpha_config = config["component_alpha"]
    source_manifest_path = verify_file(
        source["manifest_path"], source["manifest_sha256"]
    )
    source_config_path = verify_file(source["config_path"], source["config_sha256"])
    split_path = verify_file(
        source["split_assignments_path"], source["split_assignments_sha256"]
    )
    verify_file(source["clean_base_path"], source["clean_base_sha256"])
    background_path = verify_file(background_config["path"], background_config["sha256"])
    verify_file(background_config["prompt_path"], background_config["prompt_sha256"])
    alpha_path = verify_file(
        alpha_config["nominal_asset_path"], alpha_config["nominal_asset_sha256"]
    )
    verify_file(
        alpha_config["verification_overlay_path"],
        alpha_config["verification_overlay_sha256"],
    )
    v2_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    manifest_rows = read_csv(source_manifest_path)
    rows = {row["sample_id"]: row for row in manifest_rows}
    split_rows = {
        row["sample_id"]: row for row in read_csv(split_path)
    }
    selected_ids = sorted(
        sample_id
        for sample_id, row in split_rows.items()
        if row["model_split"] == source["required_parent_model_split"]
    )
    if len(selected_ids) != int(source["expected_parent_count"]):
        raise ValueError(f"expected 168 gradient parents, got {len(selected_ids)}")
    selected_rows = [rows[sample_id] for sample_id in selected_ids]
    class_counts = Counter(row["primary_class"] for row in selected_rows)
    if set(class_counts.values()) != {
        int(source["expected_parent_count_per_defect_class"])
    }:
        raise ValueError(f"unexpected gradient parent class counts: {class_counts}")
    source_rows_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected_rows:
        source_rows_by_class[row["primary_class"]].append(row)
    background = Image.open(background_path).convert("RGB")
    background.load()
    nominal_alpha = Image.open(alpha_path).convert("L")
    nominal_alpha.load()
    if nominal_alpha.size != (512, 512):
        raise ValueError("nominal component alpha must be 512x512")
    if set(np.unique(np.asarray(nominal_alpha)).tolist()) - {0, 255}:
        raise ValueError("nominal component alpha must be binary")
    plans = build_scene_plans(config, selected_rows)
    if len(plans) != int(config["scene_count"]):
        raise ValueError("scene plan count mismatch")
    return Context(
        config=config,
        config_path=config_path,
        config_sha256=config_sha256,
        release_root=release_root.resolve(),
        v2_config=v2_config,
        source_rows={row["sample_id"]: row for row in selected_rows},
        source_rows_by_class=dict(source_rows_by_class),
        source_split_rows={sample_id: split_rows[sample_id] for sample_id in selected_ids},
        background=background,
        nominal_alpha_base=nominal_alpha,
        plans=plans,
        parent_cache={},
    )


def load_parent_asset(context: Context, parent_id: str) -> ParentAsset:
    cached = context.parent_cache.get(parent_id)
    if cached is not None:
        return cached
    row = context.source_rows[parent_id]
    defect, clean, semantic = v3.reconstruct_parent(row, context.v2_config)
    parameters = json.loads(row["parameters_json"])
    dummy = Image.new("RGB", context.nominal_alpha_base.size, (0, 0, 0))
    _, nominal_alpha = v2.apply_geometry(
        dummy,
        context.nominal_alpha_base,
        parameters["geometry"],
        int(context.v2_config["image_size"]),
    )
    nominal_alpha = nominal_alpha.point(lambda value: 255 if value >= 128 else 0)
    cached = ParentAsset(
        row=row,
        defect=defect.convert("RGB"),
        clean=clean.convert("RGB"),
        semantic=semantic.convert("L"),
        nominal_alpha=nominal_alpha.convert("L"),
    )
    context.parent_cache[parent_id] = cached
    return cached


def uniform_pair(rng: random.Random, bounds: list[float]) -> float:
    return rng.uniform(float(bounds[0]), float(bounds[1]))


def make_background(
    context: Context, plan: dict[str, Any], attempt: int
) -> tuple[Image.Image, dict[str, Any]]:
    config = context.config
    background_config = config["background"]
    rng = random.Random(
        stable_seed(plan["scene_seed"], attempt, "background")
    )
    orientation = rng.choice(background_config["orientation_degrees"])
    transpose_by_angle = {
        0: None,
        90: Image.Transpose.ROTATE_90,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_270,
    }
    background = context.background
    if transpose_by_angle[int(orientation)] is not None:
        background = background.transpose(transpose_by_angle[int(orientation)])
    width = int(config["scene_width"])
    height = int(config["scene_height"])
    cover_scale = max(width / background.width, height / background.height)
    resized = background.resize(
        (
            max(width, int(math.ceil(background.width * cover_scale))),
            max(height, int(math.ceil(background.height * cover_scale))),
        ),
        Image.Resampling.LANCZOS,
    )
    crop_x = rng.randint(0, max(0, resized.width - width))
    crop_y = rng.randint(0, max(0, resized.height - height))
    background = resized.crop((crop_x, crop_y, crop_x + width, crop_y + height))
    brightness = uniform_pair(rng, background_config["brightness"])
    contrast = uniform_pair(rng, background_config["contrast"])
    background = ImageEnhance.Brightness(background).enhance(brightness)
    background = ImageEnhance.Contrast(background).enhance(contrast)
    noise_sigma = uniform_pair(rng, background_config["noise_sigma"])
    noise_seed = rng.randrange(0, 2**32)
    array = np.asarray(background, dtype=np.float32)
    if noise_sigma:
        noise_rng = np.random.default_rng(noise_seed)
        array += noise_rng.normal(0.0, noise_sigma, array.shape)
    background = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")
    return background, {
        "orientation_degrees": orientation,
        "crop_x": crop_x,
        "crop_y": crop_y,
        "resized_width": resized.width,
        "resized_height": resized.height,
        "brightness": round(brightness, 8),
        "contrast": round(contrast, 8),
        "noise_sigma": round(noise_sigma, 8),
        "noise_seed": noise_seed,
    }


def sample_component_light(
    context: Context,
    plan: dict[str, Any],
    instance_plan: dict[str, Any],
    attempt: int,
) -> dict[str, Any]:
    profile = plan["lighting_profile"]
    ranges = context.config["lighting_profile_ranges"][profile]
    rng = random.Random(
        stable_seed(
            plan["scene_seed"],
            attempt,
            "component-light",
            instance_plan["placement_slot"],
        )
    )
    params: dict[str, Any] = {
        "profile": profile,
        "component_only": True,
        "exposure_ev": uniform_pair(rng, ranges["exposure_ev"]),
        "gamma": 1.0,
        "contrast": uniform_pair(rng, ranges["contrast"]),
        "channel_gain_r": uniform_pair(rng, ranges["channel_gain_r"]),
        "channel_gain_g": uniform_pair(rng, ranges["channel_gain_g"]),
        "channel_gain_b": uniform_pair(rng, ranges["channel_gain_b"]),
        "gradient_strength": uniform_pair(rng, ranges["gradient_strength"]),
        "gradient_angle_deg": rng.uniform(-180.0, 180.0),
        "shadow_strength": 0.0,
        "shadow_width": 0.25,
        "shadow_angle_deg": 0.0,
        "shadow_offset": 0.0,
        "vignette_strength": 0.0,
        "hotspot_strength": uniform_pair(rng, ranges["hotspot_strength"]),
        "hotspot_sigma": rng.uniform(0.45, 0.75),
        "hotspot_center_x": rng.uniform(0.35, 0.65),
        "hotspot_center_y": rng.uniform(0.35, 0.65),
        "blur_radius": 0.0,
        "noise_sigma": 0.0,
        "noise_seed": 0,
        "jpeg_quality": 95,
    }
    return params


def transform_instance(
    context: Context,
    plan: dict[str, Any],
    instance_plan: dict[str, Any],
    attempt: int,
) -> dict[str, Any]:
    config = context.config
    layout = config["layout"]
    alpha_config = config["component_alpha"]
    parent = load_parent_asset(context, instance_plan["source_parent_sample_id"])
    class_name = instance_plan["class_name"]
    is_normal = class_name == "normal_proxy"
    defect = parent.clean if is_normal else parent.defect
    clean = parent.clean
    semantic_array = (
        np.zeros((512, 512), dtype=np.uint8)
        if is_normal
        else np.asarray(parent.semantic, dtype=np.uint8)
    )
    nominal_hard = np.asarray(parent.nominal_alpha, dtype=np.uint8) >= 128
    visible_hard = nominal_hard.copy()
    if class_name in alpha_config["missing_material_classes"]:
        visible_hard &= semantic_array == 0
    edge_erosion = int(alpha_config["edge_erosion_px"])
    nominal_paste = erode_image(parent.nominal_alpha, edge_erosion)
    visible_paste_array = np.asarray(nominal_paste, dtype=np.uint8).copy()
    if class_name in alpha_config["missing_material_classes"]:
        visible_paste_array[semantic_array > 0] = 0
    visible_paste = Image.fromarray(visible_paste_array, mode="L")

    light_params = sample_component_light(
        context, plan, instance_plan, attempt
    )
    lit_defect = v3.apply_condition(defect, light_params)
    lit_clean = v3.apply_condition(clean, light_params)
    source_bbox = bbox_xyxy(nominal_hard)
    if source_bbox is None:
        raise RuntimeError("empty parent nominal alpha")
    margin = 6
    x0 = max(0, source_bbox[0] - margin)
    y0 = max(0, source_bbox[1] - margin)
    x1 = min(512, source_bbox[2] + margin)
    y1 = min(512, source_bbox[3] + margin)
    box = (x0, y0, x1, y1)
    crops: dict[str, Image.Image] = {
        "defect": lit_defect.crop(box),
        "clean": lit_clean.crop(box),
        "nominal_hard": Image.fromarray(
            nominal_hard.astype(np.uint8) * 255, mode="L"
        ).crop(box),
        "visible_hard": Image.fromarray(
            visible_hard.astype(np.uint8) * 255, mode="L"
        ).crop(box),
        "nominal_paste": nominal_paste.crop(box),
        "visible_paste": visible_paste.crop(box),
        "semantic": Image.fromarray(semantic_array, mode="L").crop(box),
    }
    rng = random.Random(
        stable_seed(
            plan["scene_seed"],
            attempt,
            "placement",
            instance_plan["placement_slot"],
        )
    )
    rotation = rng.uniform(*layout["rotation_degrees"])
    rotated_reference = crops["visible_paste"].rotate(
        rotation,
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=0,
    )
    soft_blur = float(alpha_config["soft_edge_blur_px"])
    if soft_blur > 0:
        rotated_reference = rotated_reference.filter(
            ImageFilter.GaussianBlur(soft_blur)
        )
    rotated_reference_bbox = bbox_xyxy(
        np.asarray(rotated_reference, dtype=np.uint8) >= 128
    )
    if rotated_reference_bbox is None:
        raise RuntimeError("empty rotated visible component reference")
    rotated_native_long_side = max(
        rotated_reference_bbox[2] - rotated_reference_bbox[0],
        rotated_reference_bbox[3] - rotated_reference_bbox[1],
    )
    scale_floor, scale_ceiling = layout["effective_scale_guard"]
    target_floor = max(
        float(layout["component_final_visible_long_side_px"][0]),
        rotated_native_long_side * float(scale_floor),
    )
    target_ceiling = min(
        float(layout["component_final_visible_long_side_px"][1]),
        rotated_native_long_side * float(scale_ceiling),
    )
    if target_floor > target_ceiling:
        raise RuntimeError(f"no valid target size for {parent.row['sample_id']}")
    target_long_side = rng.uniform(target_floor, target_ceiling)
    scale = target_long_side / float(rotated_native_long_side)
    target_size = (
        max(1, int(round(crops["defect"].width * scale))),
        max(1, int(round(crops["defect"].height * scale))),
    )
    for name in ("defect", "clean"):
        crops[name] = crops[name].resize(target_size, Image.Resampling.LANCZOS)
    for name in ("nominal_hard", "visible_hard", "semantic"):
        crops[name] = crops[name].resize(target_size, Image.Resampling.NEAREST)
    for name in ("nominal_paste", "visible_paste"):
        crops[name] = crops[name].resize(target_size, Image.Resampling.LANCZOS)
    for name in ("defect", "clean"):
        crops[name] = crops[name].rotate(
            rotation,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=(0, 0, 0),
        )
    for name in ("nominal_hard", "visible_hard", "semantic"):
        crops[name] = crops[name].rotate(
            rotation,
            resample=Image.Resampling.NEAREST,
            expand=True,
            fillcolor=0,
        )
    for name in ("nominal_paste", "visible_paste"):
        crops[name] = crops[name].rotate(
            rotation,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=0,
        )
        if soft_blur > 0:
            crops[name] = crops[name].filter(ImageFilter.GaussianBlur(soft_blur))
    crops["nominal_rendered"] = crops["nominal_paste"].point(
        lambda value: 255 if value >= 128 else 0
    )
    crops["visible_rendered"] = crops["visible_paste"].point(
        lambda value: 255 if value >= 128 else 0
    )
    return {
        **instance_plan,
        "source_parent": parent.row,
        "light_params": light_params,
        "scale": scale,
        "target_long_side_px": target_long_side,
        "rotation_degrees": rotation,
        "jitter_x": rng.randint(*layout["jitter_x_px"]),
        "jitter_y": rng.randint(*layout["jitter_y_px"]),
        "defect_image": crops["defect"],
        "clean_image": crops["clean"],
        "nominal_hard": crops["nominal_hard"],
        "visible_hard": crops["visible_hard"],
        "nominal_paste": crops["nominal_paste"],
        "visible_paste": crops["visible_paste"],
        "nominal_rendered": crops["nominal_rendered"],
        "visible_rendered": crops["visible_rendered"],
        "semantic": crops["semantic"],
    }


def paste_mask_array(
    canvas: np.ndarray, mask: np.ndarray, left: int, top: int, value: int
) -> None:
    height, width = mask.shape
    region = canvas[top : top + height, left : left + width]
    region[mask] = value


def apply_scene_sensor(
    image: Image.Image, sensor_params: dict[str, Any]
) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    sigma = float(sensor_params["noise_sigma"])
    if sigma:
        rng = np.random.default_rng(int(sensor_params["noise_seed"]))
        array += rng.normal(0.0, sigma, array.shape)
    output = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")
    blur = float(sensor_params["blur_radius_px"])
    if blur > 0.04:
        output = output.filter(ImageFilter.GaussianBlur(blur))
    return output


def detector_visibility_metrics(
    defect: Image.Image,
    clean: Image.Image,
    defect_mask: np.ndarray,
    detector_input_size: int,
    pixel_threshold: float,
) -> dict[str, float | int]:
    scale = detector_input_size / float(defect.width)
    resized_size = (
        detector_input_size,
        max(1, int(round(defect.height * scale))),
    )
    defect_array = np.asarray(
        defect.resize(resized_size, Image.Resampling.BILINEAR), dtype=np.uint8
    )
    clean_array = np.asarray(
        clean.resize(resized_size, Image.Resampling.BILINEAR), dtype=np.uint8
    )
    mask = np.asarray(
        Image.fromarray(defect_mask.astype(np.uint8) * 255, mode="L").resize(
            resized_size, Image.Resampling.NEAREST
        ),
        dtype=np.uint8,
    ) > 0
    bbox = mask_bbox(mask)
    if bbox is None:
        return {
            "area": 0,
            "bbox_x": 0,
            "bbox_y": 0,
            "bbox_w": 0,
            "bbox_h": 0,
            "major": 0,
            "minor": 0,
            "diag": 0.0,
            "mean_abs_delta": 0.0,
            "delta_e76_p50": 0.0,
            "changed_fraction": 0.0,
        }
    x, y, width, height = bbox
    differences = np.abs(
        defect_array.astype(np.float32) - clean_array.astype(np.float32)
    )
    per_pixel = differences.mean(axis=2)
    changed = per_pixel[mask] >= float(pixel_threshold)
    defect_lab = v2.rgb_pixels_to_lab(defect_array[mask])
    clean_lab = v2.rgb_pixels_to_lab(clean_array[mask])
    delta_e = np.linalg.norm(defect_lab - clean_lab, axis=1)
    return {
        "area": int(mask.sum()),
        "bbox_x": x,
        "bbox_y": y,
        "bbox_w": width,
        "bbox_h": height,
        "major": max(width, height),
        "minor": min(width, height),
        "diag": round(math.hypot(width, height), 6),
        "mean_abs_delta": round(float(per_pixel[mask].mean()), 6),
        "delta_e76_p50": round(float(np.median(delta_e)), 6),
        "changed_fraction": round(float(changed.mean()), 6),
    }


def luma_values(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    return 0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]


def render_scene(
    context: Context, plan: dict[str, Any], attempt: int
) -> dict[str, Any]:
    config = context.config
    width = int(config["scene_width"])
    height = int(config["scene_height"])
    layout = config["layout"]
    qc = config["qc"]
    background, background_params = make_background(context, plan, attempt)
    transformed = [
        transform_instance(context, plan, item, attempt)
        for item in plan["instances"]
    ]
    nominal_scene_masks: list[np.ndarray] = []
    visible_scene_masks: list[np.ndarray] = []
    pre_erosion_nominal_scene_masks: list[np.ndarray] = []
    pre_erosion_visible_scene_masks: list[np.ndarray] = []
    defect_scene_masks: list[np.ndarray] = []
    placed_boxes: list[tuple[int, int, int, int]] = []
    for item in transformed:
        cell = int(item["grid_cell"])
        column = cell % int(layout["columns"])
        row = cell // int(layout["columns"])
        center_x = int(round((column + 0.5) * width / int(layout["columns"])))
        center_y = int(round((row + 0.5) * height / int(layout["rows"])))
        center_x += int(item["jitter_x"])
        center_y += int(item["jitter_y"])
        object_width, object_height = item["nominal_rendered"].size
        left = int(round(center_x - object_width / 2))
        top = int(round(center_y - object_height / 2))
        right = left + object_width
        bottom = top + object_height
        item["left"] = left
        item["top"] = top
        if left < 0 or top < 0 or right > width or bottom > height:
            return {"failures": [f"frame_truncation:{item['instance_index']}"]}
        pre_erosion_nominal_local = (
            np.asarray(item["nominal_hard"], dtype=np.uint8) >= 128
        )
        pre_erosion_visible_local = (
            np.asarray(item["visible_hard"], dtype=np.uint8) >= 128
        )
        nominal_local = np.asarray(item["nominal_rendered"], dtype=np.uint8) >= 128
        visible_local = np.asarray(item["visible_rendered"], dtype=np.uint8) >= 128
        semantic_local = np.asarray(item["semantic"], dtype=np.uint8) > 0
        if item["class_name"] not in config["component_alpha"]["missing_material_classes"]:
            semantic_local &= visible_local
        nominal_scene = np.zeros((height, width), dtype=bool)
        visible_scene = np.zeros((height, width), dtype=bool)
        pre_erosion_nominal_scene = np.zeros((height, width), dtype=bool)
        pre_erosion_visible_scene = np.zeros((height, width), dtype=bool)
        defect_scene = np.zeros((height, width), dtype=bool)
        nominal_scene[top:bottom, left:right] = nominal_local
        visible_scene[top:bottom, left:right] = visible_local
        pre_erosion_nominal_scene[top:bottom, left:right] = pre_erosion_nominal_local
        pre_erosion_visible_scene[top:bottom, left:right] = pre_erosion_visible_local
        defect_scene[top:bottom, left:right] = semantic_local
        nominal_box = bbox_xyxy(nominal_scene)
        if nominal_box is None:
            return {"failures": [f"empty_nominal:{item['instance_index']}"]}
        if any(iou_xyxy(nominal_box, prior) > float(qc["maximum_component_overlap_iou"]) for prior in placed_boxes):
            return {"failures": [f"bbox_overlap:{item['instance_index']}"]}
        if any(
            bbox_gap_xyxy(nominal_box, prior) < float(layout["minimum_gap_px"])
            for prior in placed_boxes
        ):
            return {"failures": [f"bbox_gap:{item['instance_index']}"]}
        if any(np.any(nominal_scene & prior) for prior in nominal_scene_masks):
            return {"failures": [f"mask_overlap:{item['instance_index']}"]}
        placed_boxes.append(nominal_box)
        nominal_scene_masks.append(nominal_scene)
        visible_scene_masks.append(visible_scene)
        pre_erosion_nominal_scene_masks.append(pre_erosion_nominal_scene)
        pre_erosion_visible_scene_masks.append(pre_erosion_visible_scene)
        defect_scene_masks.append(defect_scene)

    nominal_union = np.logical_or.reduce(nominal_scene_masks)
    visible_union = np.logical_or.reduce(visible_scene_masks)
    # Contact shadows are allowed to darken the belt; they never brighten it.
    shadow_config = config["component_shadow"]
    shadow_rng = random.Random(stable_seed(plan["scene_seed"], attempt, "shadow"))
    shadow_canvas = Image.new("L", (width, height), 0)
    for item in transformed:
        opacity = uniform_pair(shadow_rng, shadow_config["opacity"])
        offset_x = shadow_rng.randint(*shadow_config["offset_x_px"])
        offset_y = shadow_rng.randint(*shadow_config["offset_y_px"])
        alpha = item["visible_paste"].point(
            lambda value, opacity=opacity: int(round(value * opacity))
        )
        item["shadow_params"] = {
            "opacity": round(opacity, 8),
            "offset_x_px": offset_x,
            "offset_y_px": offset_y,
        }
        shadow_canvas.paste(
            alpha,
            (int(item["left"]) + offset_x, int(item["top"]) + offset_y),
        )
    shadow_blur = uniform_pair(shadow_rng, shadow_config["blur_radius_px"])
    shadow_canvas = shadow_canvas.filter(ImageFilter.GaussianBlur(shadow_blur))
    background_array = np.asarray(background, dtype=np.float32)
    shadow_strength = np.asarray(shadow_canvas, dtype=np.float32) / 255.0
    shadowed_array = background_array * (1.0 - shadow_strength[..., None])
    shadowed_background = Image.fromarray(
        np.clip(shadowed_array, 0, 255).astype(np.uint8), mode="RGB"
    )
    defect_scene = shadowed_background.copy()
    clean_scene = shadowed_background.copy()
    for item in transformed:
        location = (int(item["left"]), int(item["top"]))
        defect_scene.paste(item["defect_image"], location, item["visible_paste"])
        clean_scene.paste(item["clean_image"], location, item["nominal_paste"])

    positive_spill = np.maximum(
        0.0,
        np.asarray(defect_scene, dtype=np.float32)
        - np.asarray(shadowed_background, dtype=np.float32),
    ).max(axis=2)
    spill_region = ~dilate(nominal_union, 4)
    spill_values = positive_spill[spill_region]
    spill_p99 = float(np.percentile(spill_values, 99)) if spill_values.size else 0.0
    spill_max = float(spill_values.max()) if spill_values.size else 0.0
    if spill_p99 > 1.0 or spill_max > 3.0:
        return {"failures": [f"component_light_spill:{spill_p99:.3f}/{spill_max:.3f}"]}

    sensor_rng = random.Random(
        stable_seed(plan["scene_seed"], attempt, "scene-sensor")
    )
    sensor_config = config["scene_sensor"]
    sensor_params = {
        "noise_sigma": uniform_pair(sensor_rng, sensor_config["noise_sigma"]),
        "noise_seed": sensor_rng.randrange(0, 2**32),
        "blur_radius_px": uniform_pair(sensor_rng, sensor_config["blur_radius_px"]),
        "jpeg_quality": sensor_rng.randint(*sensor_config["jpeg_quality"]),
    }
    defect_sensor = apply_scene_sensor(defect_scene, sensor_params)
    clean_sensor = apply_scene_sensor(clean_scene, sensor_params)
    background_reference_sensor = apply_scene_sensor(background, sensor_params)
    defect_jpeg, defect_payload = v2.jpeg_roundtrip(
        defect_sensor, int(sensor_params["jpeg_quality"])
    )
    clean_jpeg, _ = v2.jpeg_roundtrip(
        clean_sensor, int(sensor_params["jpeg_quality"])
    )
    background_reference_jpeg, _ = v2.jpeg_roundtrip(
        background_reference_sensor, int(sensor_params["jpeg_quality"])
    )

    luma = luma_values(defect_jpeg)
    background_reference_luma = luma_values(background_reference_jpeg)
    background_region = ~dilate(nominal_union, 4)
    background_luma = background_reference_luma[background_region]
    component_luma = luma[visible_union]
    background_mean = float(background_luma.mean())
    background_std = float(background_luma.std())
    background_p99 = float(np.percentile(background_luma, 99))
    component_mean = float(component_luma.mean())
    luma_delta = component_mean - background_mean
    luma_ratio = component_mean / max(background_mean, 1e-6)
    failures: list[str] = []
    if background_mean > float(qc["maximum_background_mean_luma"]):
        failures.append(f"background_mean:{background_mean:.4f}")
    if background_p99 > float(qc["maximum_background_p99_luma"]):
        failures.append(f"background_p99:{background_p99:.4f}")
    if luma_delta < float(qc["minimum_component_background_luma_delta"]):
        failures.append(f"component_background_delta:{luma_delta:.4f}")
    if luma_ratio < float(qc["minimum_component_background_luma_ratio"]):
        failures.append(f"component_background_ratio:{luma_ratio:.4f}")

    instance_map = np.zeros((height, width), dtype=np.uint16)
    semantic_map = np.zeros((height, width), dtype=np.uint8)
    instance_results: list[dict[str, Any]] = []
    for (
        item,
        nominal_mask,
        visible_mask,
        pre_erosion_nominal_mask,
        pre_erosion_visible_mask,
        defect_mask,
    ) in zip(
        transformed,
        nominal_scene_masks,
        visible_scene_masks,
        pre_erosion_nominal_scene_masks,
        pre_erosion_visible_scene_masks,
        defect_scene_masks,
        strict=True,
    ):
        instance_id = int(item["instance_index"])
        instance_map[visible_mask] = instance_id
        class_name = item["class_name"]
        if class_name != "normal_proxy":
            semantic_id = int(config["defect_semantic_ids"][class_name])
            semantic_map[defect_mask] = semantic_id
        nominal_bbox = mask_bbox(nominal_mask)
        visible_bbox = mask_bbox(visible_mask)
        pre_erosion_nominal_bbox = mask_bbox(pre_erosion_nominal_mask)
        pre_erosion_visible_bbox = mask_bbox(pre_erosion_visible_mask)
        if nominal_bbox is None or visible_bbox is None:
            failures.append(f"empty_component:{instance_id}")
            continue
        if visible_bbox[2] < int(qc["minimum_component_width_px"]):
            failures.append(f"component_width:{instance_id}:{visible_bbox[2]}")
        if visible_bbox[3] < int(qc["minimum_component_height_px"]):
            failures.append(f"component_height:{instance_id}:{visible_bbox[3]}")
        final_visible_long_side = max(visible_bbox[2], visible_bbox[3])
        final_long_side_bounds = layout["component_final_visible_long_side_px"]
        if not (
            int(final_long_side_bounds[0])
            <= final_visible_long_side
            <= int(final_long_side_bounds[1])
        ):
            failures.append(
                f"component_final_visible_long_side:{instance_id}:"
                f"{final_visible_long_side}"
            )
        visible_fraction = float(visible_mask.sum() / max(1, nominal_mask.sum()))
        # Missing-material defects intentionally reduce the visible fraction.
        if class_name not in config["component_alpha"]["missing_material_classes"]:
            if visible_fraction < float(qc["minimum_component_visible_fraction"]):
                failures.append(
                    f"component_visible_fraction:{instance_id}:{visible_fraction:.6f}"
                )
        annulus_outer = dilate(
            nominal_mask, int(qc["local_belt_annulus_outer_radius_px"])
        )
        annulus_exclusion = dilate(
            nominal_union, int(qc["local_belt_annulus_inner_radius_px"])
        )
        local_belt_annulus = annulus_outer & ~annulus_exclusion
        if int(local_belt_annulus.sum()) < 100:
            failures.append(f"local_belt_annulus_area:{instance_id}")
            local_background_mean_luma = 0.0
            instance_component_mean_luma = float(luma[visible_mask].mean())
            instance_luma_delta = 0.0
            instance_luma_ratio = 0.0
        else:
            local_background_mean_luma = float(
                background_reference_luma[local_belt_annulus].mean()
            )
            instance_component_mean_luma = float(luma[visible_mask].mean())
            instance_luma_delta = (
                instance_component_mean_luma - local_background_mean_luma
            )
            instance_luma_ratio = instance_component_mean_luma / max(
                local_background_mean_luma, 1e-6
            )
            if instance_luma_delta < float(
                qc["minimum_instance_component_background_luma_delta"]
            ):
                failures.append(
                    f"instance_component_background_delta:{instance_id}:"
                    f"{instance_luma_delta:.4f}"
                )
            if instance_luma_ratio < float(
                qc["minimum_instance_component_background_luma_ratio"]
            ):
                failures.append(
                    f"instance_component_background_ratio:{instance_id}:"
                    f"{instance_luma_ratio:.4f}"
                )
        defect_bbox = mask_bbox(defect_mask)
        visibility_metrics: dict[str, Any] | None = None
        defect_envelope_fraction = 1.0
        if class_name != "normal_proxy":
            if defect_bbox is None:
                failures.append(f"empty_defect:{instance_id}")
            else:
                envelope = dilate(nominal_mask, 14)
                defect_envelope_fraction = float(
                    np.logical_and(defect_mask, envelope).sum()
                    / max(1, defect_mask.sum())
                )
                if defect_envelope_fraction < 0.98:
                    failures.append(
                        f"defect_outside_envelope:{instance_id}:{defect_envelope_fraction:.6f}"
                    )
                visibility_metrics = detector_visibility_metrics(
                    defect_jpeg,
                    clean_jpeg,
                    defect_mask,
                    int(qc["detector_input_size"]),
                    float(qc["pixel_change_threshold"]),
                )
                if int(visibility_metrics["area"]) < int(
                    qc["minimum_defect_area_at_detector_input_px"]
                ):
                    failures.append(
                        f"defect_area:{instance_id}:{visibility_metrics['area']}"
                    )
                if float(visibility_metrics["diag"]) < float(
                    qc["minimum_defect_diagonal_at_detector_input_px"]
                ):
                    failures.append(
                        f"defect_diag:{instance_id}:{visibility_metrics['diag']}"
                    )
                if float(visibility_metrics["mean_abs_delta"]) < float(
                    qc["minimum_mean_abs_delta"][class_name]
                ):
                    failures.append(
                        f"defect_mad:{instance_id}:{visibility_metrics['mean_abs_delta']}"
                    )
                if float(visibility_metrics["changed_fraction"]) < float(
                    qc["minimum_changed_fraction"][class_name]
                ):
                    failures.append(
                        f"defect_changed_fraction:{instance_id}:{visibility_metrics['changed_fraction']}"
                    )
        instance_results.append(
            {
                **item,
                "nominal_bbox": nominal_bbox,
                "visible_bbox": visible_bbox,
                "nominal_area": int(nominal_mask.sum()),
                "visible_area": int(visible_mask.sum()),
                "final_visible_long_side_px": int(final_visible_long_side),
                "pre_erosion_nominal_bbox": pre_erosion_nominal_bbox,
                "pre_erosion_visible_bbox": pre_erosion_visible_bbox,
                "pre_erosion_nominal_area": int(pre_erosion_nominal_mask.sum()),
                "pre_erosion_visible_area": int(pre_erosion_visible_mask.sum()),
                "visible_render_boundary_change_fraction": float(
                    np.logical_xor(visible_mask, pre_erosion_visible_mask).sum()
                    / max(
                        1,
                        np.logical_or(visible_mask, pre_erosion_visible_mask).sum(),
                    )
                ),
                "component_visible_fraction": visible_fraction,
                "local_belt_annulus_area": int(local_belt_annulus.sum()),
                "local_background_mean_luma": local_background_mean_luma,
                "instance_component_mean_luma": instance_component_mean_luma,
                "instance_component_background_luma_delta": instance_luma_delta,
                "instance_component_background_luma_ratio": instance_luma_ratio,
                "defect_bbox": defect_bbox,
                "defect_area": int(defect_mask.sum()),
                "defect_envelope_fraction": defect_envelope_fraction,
                "visibility_metrics_1024": visibility_metrics,
            }
        )
    return {
        "failures": failures,
        "image": defect_jpeg,
        "image_payload": defect_payload,
        "clean_image": clean_jpeg,
        "component_instance_mask": instance_map,
        "defect_semantic_mask": semantic_map,
        "instances": instance_results,
        "attempt": attempt,
        "background_params": background_params,
        "sensor_params": sensor_params,
        "shadow_blur_radius_px": shadow_blur,
        "component_light_spill_p99": spill_p99,
        "component_light_spill_max": spill_max,
        "background_mean_luma": background_mean,
        "background_std_luma": background_std,
        "background_p99_luma": background_p99,
        "component_mean_luma": component_mean,
        "component_background_luma_delta": luma_delta,
        "component_background_luma_ratio": luma_ratio,
        "background_luma_reference": "SHADOWLESS_SAME_SENSOR_JPEG",
    }


def repository_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def release_relative(path: Path, release_root: Path) -> str:
    return path.resolve().relative_to(release_root.resolve()).as_posix()


def yolo_box_line(
    class_id: int,
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> str:
    x, y, box_width, box_height = bbox
    center_x = (x + box_width / 2.0) / width
    center_y = (y + box_height / 2.0) / height
    return (
        f"{class_id} {center_x:.10f} {center_y:.10f} "
        f"{box_width / width:.10f} {box_height / height:.10f}"
    )


def color_for_class(class_name: str) -> tuple[int, int, int]:
    colors = {
        "normal_proxy": (68, 214, 111),
        "scratch": (255, 84, 84),
        "surface_spot": (255, 179, 0),
        "discoloration": (72, 149, 239),
        "contamination": (177, 95, 255),
        "lead_breakage": (255, 73, 200),
        "body_chip": (50, 220, 205),
        "body_crack": (255, 124, 43),
    }
    return colors[class_name]


def overlay_scene(
    image: Image.Image, instances: list[dict[str, Any]], line_width: int = 4
) -> Image.Image:
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    for item in instances:
        bbox_value = item.get("visible_bbox", item.get("visible_bbox_xywh"))
        if bbox_value is None:
            raise ValueError("overlay instance is missing visible bbox")
        x, y, width, height = bbox_value
        class_name = item.get("class_name", item.get("component_status_class"))
        if class_name is None:
            raise ValueError("overlay instance is missing component status class")
        color = color_for_class(class_name)
        draw.rectangle(
            (x, y, x + width - 1, y + height - 1),
            outline=color,
            width=line_width,
        )
        label = class_name
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        draw.rectangle(
            (x, max(0, y - text_height - 6), x + text_width + 6, y),
            fill=color,
        )
        draw.text(
            (x + 3, max(0, y - text_height - 4)),
            label,
            fill=(0, 0, 0),
            font=font,
        )
        defect_bbox = item.get("defect_bbox", item.get("defect_bbox_xywh"))
        if defect_bbox is not None:
            dx, dy, dw, dh = defect_bbox
            draw.rectangle(
                (dx, dy, dx + dw - 1, dy + dh - 1),
                outline=(255, 255, 0),
                width=max(2, line_width // 2),
            )
    return output


def create_profile_contact_sheet(
    context: Context,
    scene_records: list[dict[str, Any]],
    instances_by_scene: dict[str, list[dict[str, Any]]],
    profile: str,
    overlay: bool,
) -> Path:
    selected = [row for row in scene_records if row["lighting_profile"] == profile]
    thumb_width, thumb_height, label_height = 160, 90, 14
    columns = 12
    rows = math.ceil(len(selected) / columns)
    sheet = Image.new(
        "RGB", (columns * thumb_width, rows * (thumb_height + label_height)), (18, 18, 18)
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, row in enumerate(selected):
        image = Image.open(ROOT / row["image_path"]).convert("RGB")
        if overlay:
            image = overlay_scene(image, instances_by_scene[row["scene_id"]], 4)
        image = image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        left = (index % columns) * thumb_width
        top = (index // columns) * (thumb_height + label_height)
        sheet.paste(image, (left, top))
        draw.text(
            (left + 2, top + thumb_height + 1),
            row["scene_id"],
            fill=(225, 225, 225),
            font=font,
        )
    suffix = "overlay" if overlay else "raw"
    output = (
        context.release_root
        / "annotations"
        / "contact_sheets"
        / f"{profile}_{suffix}_96_at_160.jpg"
    )
    sheet.save(output, quality=90, subsampling=1, optimize=True)
    return output


def create_overview_contact_sheet(
    context: Context,
    scene_records: list[dict[str, Any]],
    instances_by_scene: dict[str, list[dict[str, Any]]],
) -> Path:
    profiles = list(context.config["lighting_profiles"])
    tile_width, tile_height, label_height = 320, 180, 18
    columns = 4
    sheet = Image.new(
        "RGB",
        (columns * tile_width, len(profiles) * (tile_height + label_height)),
        (15, 15, 15),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for profile_index, profile in enumerate(profiles):
        profile_rows = [row for row in scene_records if row["lighting_profile"] == profile]
        selected = [profile_rows[8], profile_rows[56]]
        for pair_index, row in enumerate(selected):
            raw = Image.open(ROOT / row["image_path"]).convert("RGB")
            overlay = overlay_scene(raw, instances_by_scene[row["scene_id"]], 5)
            for variant_index, image in enumerate((raw, overlay)):
                column = pair_index * 2 + variant_index
                left = column * tile_width
                top = profile_index * (tile_height + label_height)
                image = image.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
                sheet.paste(image, (left, top))
                variant = "raw" if variant_index == 0 else "boxes"
                draw.text(
                    (left + 3, top + tile_height + 2),
                    f"{profile} | {row['scene_id']} | {variant}",
                    fill=(235, 235, 235),
                    font=font,
                )
    output = context.release_root / "contact_sheet.jpg"
    sheet.save(output, quality=92, subsampling=1, optimize=True)
    return output


def generate_release(context: Context, force: bool) -> dict[str, Any]:
    config = context.config
    release_root = context.release_root
    safe_prepare_release(release_root, force)
    width = int(config["scene_width"])
    height = int(config["scene_height"])
    scene_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    instances_by_scene: dict[str, list[dict[str, Any]]] = {}
    component_coco_images: list[dict[str, Any]] = []
    component_coco_annotations: list[dict[str, Any]] = []
    defect_coco_annotations: list[dict[str, Any]] = []
    component_annotation_id = 1
    defect_annotation_id = 1
    image_hashes: set[str] = set()
    max_attempts = int(config["max_scene_attempts"])

    for plan in context.plans:
        result: dict[str, Any] | None = None
        last_failures: list[str] = []
        for attempt in range(max_attempts):
            candidate = render_scene(context, plan, attempt)
            last_failures = list(candidate.get("failures", []))
            if not last_failures:
                result = candidate
                break
        if result is None:
            raise RuntimeError(
                f"scene generation failed {plan['scene_id']}: {last_failures}"
            )

        scene_id = plan["scene_id"]
        image_path = release_root / "images" / "train" / f"{scene_id}.jpg"
        component_mask_path = (
            release_root
            / "masks"
            / "component_visible_instances"
            / "train"
            / f"{scene_id}.png"
        )
        defect_mask_path = (
            release_root
            / "masks"
            / "defect_semantic"
            / "train"
            / f"{scene_id}.png"
        )
        component_yolo_path = (
            release_root
            / "labels"
            / "yolo_component_status"
            / "train"
            / f"{scene_id}.txt"
        )
        defect_yolo_path = (
            release_root
            / "labels"
            / "yolo_defects"
            / "train"
            / f"{scene_id}.txt"
        )
        image_path.write_bytes(result["image_payload"])
        Image.fromarray(result["component_instance_mask"]).save(
            component_mask_path, optimize=True
        )
        Image.fromarray(result["defect_semantic_mask"], mode="L").save(
            defect_mask_path, optimize=True
        )

        component_lines: list[str] = []
        defect_lines: list[str] = []
        published_instances: list[dict[str, Any]] = []
        for item in result["instances"]:
            class_name = item["class_name"]
            component_annotation_mask = (
                result["component_instance_mask"] == int(item["instance_index"])
            )
            component_class_id = int(config["yolo_class_ids"][class_name])
            component_lines.append(
                yolo_box_line(
                    component_class_id, item["visible_bbox"], width, height
                )
            )
            if item["defect_bbox"] is not None:
                defect_class_id = int(config["defect_semantic_ids"][class_name]) - 1
                defect_lines.append(
                    yolo_box_line(
                        defect_class_id, item["defect_bbox"], width, height
                    )
                )

            parent = item["source_parent"]
            nominal_bbox = list(item["nominal_bbox"])
            visible_bbox = list(item["visible_bbox"])
            pre_erosion_nominal_bbox = (
                list(item["pre_erosion_nominal_bbox"])
                if item["pre_erosion_nominal_bbox"] is not None
                else None
            )
            pre_erosion_visible_bbox = (
                list(item["pre_erosion_visible_bbox"])
                if item["pre_erosion_visible_bbox"] is not None
                else None
            )
            defect_bbox = (
                list(item["defect_bbox"])
                if item["defect_bbox"] is not None
                else None
            )
            component_coco_annotations.append(
                {
                    "id": component_annotation_id,
                    "image_id": plan["image_id"],
                    "category_id": int(config["coco_category_ids"][class_name]),
                    "bbox": visible_bbox,
                    "area": int(item["visible_area"]),
                    "iscrowd": 0,
                    "segmentation": coco_uncompressed_rle(
                        component_annotation_mask
                    ),
                    "attributes": {
                        "scene_instance_id": int(item["instance_index"]),
                        "nominal_bbox": nominal_bbox,
                        "nominal_area": int(item["nominal_area"]),
                        "pre_erosion_nominal_bbox": pre_erosion_nominal_bbox,
                        "pre_erosion_nominal_area": int(
                            item["pre_erosion_nominal_area"]
                        ),
                        "pre_erosion_visible_bbox": pre_erosion_visible_bbox,
                        "pre_erosion_visible_area": int(
                            item["pre_erosion_visible_area"]
                        ),
                        "visible_bbox": visible_bbox,
                        "visible_area": int(item["visible_area"]),
                        "visible_instance_mask": release_relative(
                            component_mask_path, release_root
                        ),
                        "source_parent_sample_id": parent["sample_id"],
                        "normal_proxy_from_paired_clean": bool(
                            item["normal_proxy_from_paired_clean"]
                        ),
                    },
                }
            )
            current_component_annotation_id = component_annotation_id
            component_annotation_id += 1
            current_defect_annotation_id: int | None = None
            if defect_bbox is not None:
                current_defect_annotation_id = defect_annotation_id
                defect_annotation_mask = (
                    result["defect_semantic_mask"]
                    == int(config["defect_semantic_ids"][class_name])
                )
                defect_coco_annotations.append(
                    {
                        "id": defect_annotation_id,
                        "image_id": plan["image_id"],
                        "category_id": int(config["defect_semantic_ids"][class_name]),
                        "bbox": defect_bbox,
                        "area": int(item["defect_area"]),
                        "iscrowd": 0,
                        "segmentation": coco_uncompressed_rle(
                            defect_annotation_mask
                        ),
                        "attributes": {
                            "component_annotation_id": current_component_annotation_id,
                            "scene_instance_id": int(item["instance_index"]),
                            "semantic_mask": release_relative(
                                defect_mask_path, release_root
                            ),
                        },
                    }
                )
                defect_annotation_id += 1

            instance_row = {
                "scene_id": scene_id,
                "image_id": int(plan["image_id"]),
                "instance_index": int(item["instance_index"]),
                "component_annotation_id": current_component_annotation_id,
                "defect_annotation_id": current_defect_annotation_id,
                "component_status_class": class_name,
                "component_yolo_class_id": component_class_id,
                "component_coco_category_id": int(
                    config["coco_category_ids"][class_name]
                ),
                "defect_class": None if class_name == "normal_proxy" else class_name,
                "defect_semantic_id": (
                    None
                    if class_name == "normal_proxy"
                    else int(config["defect_semantic_ids"][class_name])
                ),
                "source_parent_sample_id": parent["sample_id"],
                "source_parent_image_path": parent["image_path"],
                "source_parent_mask_path": parent["mask_path"],
                "source_parent_image_sha256": parent["image_sha256"],
                "source_parent_mask_sha256": parent["mask_sha256"],
                "source_parent_class": parent["primary_class"],
                "source_parent_severity": parent["severity"],
                "source_parent_model_split": "gradient_train",
                "normal_proxy_from_paired_clean": bool(
                    item["normal_proxy_from_paired_clean"]
                ),
                "normal_status": (
                    config["normal_status"] if class_name == "normal_proxy" else None
                ),
                "base_group_id": parent["base_group_id"],
                "source_specimen_group": parent["source_specimen_group"],
                "view": parent["view"],
                "family_split_id": parent["sample_id"],
                "composition_family_id": scene_id,
                "lighting_profile": plan["lighting_profile"],
                "placement_slot": int(item["placement_slot"]),
                "grid_cell": int(item["grid_cell"]),
                "scale": round(float(item["scale"]), 10),
                "target_long_side_px": round(float(item["target_long_side_px"]), 6),
                "final_visible_long_side_px": int(item["final_visible_long_side_px"]),
                "rotation_degrees": round(float(item["rotation_degrees"]), 8),
                "left": int(item["left"]),
                "top": int(item["top"]),
                "nominal_bbox_xywh": nominal_bbox,
                "visible_bbox_xywh": visible_bbox,
                "nominal_area": int(item["nominal_area"]),
                "visible_area": int(item["visible_area"]),
                "pre_erosion_nominal_bbox_xywh": pre_erosion_nominal_bbox,
                "pre_erosion_visible_bbox_xywh": pre_erosion_visible_bbox,
                "pre_erosion_nominal_area": int(item["pre_erosion_nominal_area"]),
                "pre_erosion_visible_area": int(item["pre_erosion_visible_area"]),
                "visible_render_boundary_change_fraction": round(
                    float(item["visible_render_boundary_change_fraction"]), 8
                ),
                "component_visible_fraction": round(
                    float(item["component_visible_fraction"]), 8
                ),
                "defect_bbox_xywh": defect_bbox,
                "defect_area": int(item["defect_area"]),
                "defect_envelope_fraction": round(
                    float(item["defect_envelope_fraction"]), 8
                ),
                "visibility_metrics_1024": item["visibility_metrics_1024"],
                "component_light_params": item["light_params"],
                "component_shadow_params": item["shadow_params"],
                "local_belt_annulus_area": int(item["local_belt_annulus_area"]),
                "local_background_mean_luma": round(
                    float(item["local_background_mean_luma"]), 6
                ),
                "instance_component_mean_luma": round(
                    float(item["instance_component_mean_luma"]), 6
                ),
                "instance_component_background_luma_delta": round(
                    float(item["instance_component_background_luma_delta"]), 6
                ),
                "instance_component_background_luma_ratio": round(
                    float(item["instance_component_background_luma_ratio"]), 6
                ),
                "training_use": config["training_use"],
                "evaluation_eligible": config["evaluation_eligible"],
                "classification_eligible": config["classification_eligible"],
            }
            instance_rows.append(instance_row)
            published_instances.append(instance_row)

        component_yolo_text = "\n".join(component_lines) + "\n"
        defect_yolo_text = "\n".join(defect_lines) + "\n"
        component_yolo_path.write_text(
            component_yolo_text, encoding="ascii", newline="\n"
        )
        defect_yolo_path.write_text(defect_yolo_text, encoding="ascii", newline="\n")
        image_sha = sha256_bytes(result["image_payload"])
        if image_sha in image_hashes:
            raise RuntimeError(f"duplicate scene image SHA: {scene_id}")
        image_hashes.add(image_sha)
        component_mask_sha = sha256_file(component_mask_path)
        defect_mask_sha = sha256_file(defect_mask_path)
        component_yolo_sha = sha256_file(component_yolo_path)
        defect_yolo_sha = sha256_file(defect_yolo_path)
        class_names = [item["class_name"] for item in result["instances"]]
        parent_ids = [
            item["source_parent"]["sample_id"] for item in result["instances"]
        ]
        scene_row = {
            "scene_id": scene_id,
            "image_id": int(plan["image_id"]),
            "image_path": repository_relative(image_path),
            "component_instance_mask_path": repository_relative(component_mask_path),
            "defect_semantic_mask_path": repository_relative(defect_mask_path),
            "component_yolo_path": repository_relative(component_yolo_path),
            "defect_yolo_path": repository_relative(defect_yolo_path),
            "domain": "synthetic_black_conveyor_multi_instance",
            "split": config["split"],
            "task_type": config["task_type"],
            "training_use": config["training_use"],
            "evaluation_eligible": config["evaluation_eligible"],
            "classification_eligible": config["classification_eligible"],
            "lighting_profile": plan["lighting_profile"],
            "background_asset_id": config["background"]["asset_id"],
            "scene_seed": int(plan["scene_seed"]),
            "attempt": int(result["attempt"]),
            "instance_count": len(result["instances"]),
            "distinct_class_count": len(set(class_names)),
            "component_status_labels": "|".join(class_names),
            "source_parent_ids": "|".join(parent_ids),
            "composition_family_id": scene_id,
            "width": width,
            "height": height,
            "image_sha256": image_sha,
            "component_mask_sha256": component_mask_sha,
            "defect_mask_sha256": defect_mask_sha,
            "component_yolo_sha256": component_yolo_sha,
            "defect_yolo_sha256": defect_yolo_sha,
            "background_mean_luma": f"{result['background_mean_luma']:.6f}",
            "background_std_luma": f"{result['background_std_luma']:.6f}",
            "background_p99_luma": f"{result['background_p99_luma']:.6f}",
            "component_mean_luma": f"{result['component_mean_luma']:.6f}",
            "component_background_luma_delta": f"{result['component_background_luma_delta']:.6f}",
            "component_background_luma_ratio": f"{result['component_background_luma_ratio']:.6f}",
            "component_light_spill_p99": f"{result['component_light_spill_p99']:.6f}",
            "component_light_spill_max": f"{result['component_light_spill_max']:.6f}",
            "background_luma_reference": result["background_luma_reference"],
            "config_sha256": context.config_sha256,
            "generator_version": config["generator_version"],
            "qc_gate_version": config["qc_gate_version"],
            "qc_status": "AUTO_PASS_MULTI_INSTANCE_PAIRED_CLEAN_1024",
            "human_verified": "NO",
            "background_params_json": canonical_json(result["background_params"]),
            "sensor_params_json": canonical_json(result["sensor_params"]),
            "shadow_params_json": canonical_json(
                {
                    "blur_radius_px": result["shadow_blur_radius_px"],
                    "instances": [
                        {
                            "instance_index": int(item["instance_index"]),
                            **item["component_shadow_params"],
                        }
                        for item in published_instances
                    ],
                }
            ),
        }
        scene_rows.append(scene_row)
        instances_by_scene[scene_id] = published_instances
        component_coco_images.append(
            {
                "id": int(plan["image_id"]),
                "file_name": release_relative(image_path, release_root),
                "width": width,
                "height": height,
                "scene_id": scene_id,
            }
        )
        if (plan["scene_index"] + 1) % 24 == 0:
            print(
                f"generated scenes={plan['scene_index'] + 1}/{len(context.plans)} "
                f"instances={len(instance_rows)}",
                flush=True,
            )

    manifest_fields = [
        "scene_id",
        "image_id",
        "image_path",
        "component_instance_mask_path",
        "defect_semantic_mask_path",
        "component_yolo_path",
        "defect_yolo_path",
        "domain",
        "split",
        "task_type",
        "training_use",
        "evaluation_eligible",
        "classification_eligible",
        "lighting_profile",
        "background_asset_id",
        "scene_seed",
        "attempt",
        "instance_count",
        "distinct_class_count",
        "component_status_labels",
        "source_parent_ids",
        "composition_family_id",
        "width",
        "height",
        "image_sha256",
        "component_mask_sha256",
        "defect_mask_sha256",
        "component_yolo_sha256",
        "defect_yolo_sha256",
        "background_mean_luma",
        "background_std_luma",
        "background_p99_luma",
        "component_mean_luma",
        "component_background_luma_delta",
        "component_background_luma_ratio",
        "component_light_spill_p99",
        "component_light_spill_max",
        "background_luma_reference",
        "config_sha256",
        "generator_version",
        "qc_gate_version",
        "qc_status",
        "human_verified",
        "background_params_json",
        "sensor_params_json",
        "shadow_params_json",
    ]
    manifest_path = release_root / "annotations" / "manifest.csv"
    write_csv(manifest_path, scene_rows, manifest_fields)
    instances_path = release_root / "annotations" / "instances.jsonl"
    with instances_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in instance_rows:
            stream.write(canonical_json(row) + "\n")

    component_categories = [
        {"id": int(config["coco_category_ids"][name]), "name": name}
        for name in config["classes"]
    ]
    defect_categories = [
        {"id": int(config["defect_semantic_ids"][name]), "name": name}
        for name in config["classes"][1:]
    ]
    component_coco_path = (
        release_root / "annotations" / "coco" / "component_status_train.json"
    )
    defect_coco_path = release_root / "annotations" / "coco" / "defects_train.json"
    common_info = {
        "description": "Synthetic black-conveyor multi-instance train-only release",
        "version": config["generator_version"],
        "year": 2026,
    }
    write_json(
        component_coco_path,
        {
            "info": common_info,
            "licenses": [],
            "images": component_coco_images,
            "annotations": component_coco_annotations,
            "categories": component_categories,
        },
    )
    write_json(
        defect_coco_path,
        {
            "info": common_info,
            "licenses": [],
            "images": component_coco_images,
            "annotations": defect_coco_annotations,
            "categories": defect_categories,
        },
    )

    class_counts = Counter(row["component_status_class"] for row in instance_rows)
    profile_scene_counts = Counter(row["lighting_profile"] for row in scene_rows)
    class_profile_counts = Counter(
        (row["component_status_class"], row["lighting_profile"])
        for row in instance_rows
    )
    defect_parent_usage = Counter(
        row["source_parent_sample_id"]
        for row in instance_rows
        if not row["normal_proxy_from_paired_clean"]
    )
    defect_parent_profile_usage = Counter(
        (row["source_parent_sample_id"], row["lighting_profile"])
        for row in instance_rows
        if not row["normal_proxy_from_paired_clean"]
    )
    defect_parent_slot_usage = Counter(
        (row["source_parent_sample_id"], row["placement_slot"])
        for row in instance_rows
        if not row["normal_proxy_from_paired_clean"]
    )
    normal_parent_usage = Counter(
        row["source_parent_sample_id"]
        for row in instance_rows
        if row["normal_proxy_from_paired_clean"]
    )
    class_slot_counts = Counter(
        (
            row["component_status_class"],
            row["lighting_profile"],
            row["placement_slot"],
        )
        for row in instance_rows
    )
    class_grid_counts = Counter(
        (
            row["component_status_class"],
            row["lighting_profile"],
            row["grid_cell"],
        )
        for row in instance_rows
    )
    profile_grid_counts = Counter(
        (row["lighting_profile"], row["grid_cell"]) for row in instance_rows
    )
    pair_count_by_profile: dict[str, Counter[tuple[str, str]]] = {}
    for profile in config["lighting_profiles"]:
        pairs: Counter[tuple[str, str]] = Counter()
        for scene in scene_rows:
            if scene["lighting_profile"] != profile:
                continue
            labels = scene["component_status_labels"].split("|")
            pairs.update(combinations(sorted(labels), 2))
        pair_count_by_profile[profile] = pairs

    expected_class_count = int(config["expected_instances_per_class"])
    expected_class_profile = int(config["expected_instances_per_class_profile"])
    if any(class_counts[name] != expected_class_count for name in config["classes"]):
        raise RuntimeError(f"class output count mismatch: {class_counts}")
    if any(
        class_profile_counts[(name, profile)] != expected_class_profile
        for name in config["classes"]
        for profile in config["lighting_profiles"]
    ):
        raise RuntimeError("class x profile output count mismatch")
    if any(value != 12 for value in class_slot_counts.values()):
        raise RuntimeError("class x profile x placement slot must equal 12")
    expected_grid_range = set(
        int(value)
        for value in config["expected_instances_per_class_profile_grid_cell"]
    )
    if set(class_grid_counts.values()) != expected_grid_range:
        raise RuntimeError(
            f"class x profile x grid cell balance mismatch: "
            f"{Counter(class_grid_counts.values())}"
        )
    if set(profile_grid_counts.values()) != {60}:
        raise RuntimeError(f"profile x grid cell balance mismatch: {profile_grid_counts}")
    if set(defect_parent_usage.values()) != {10} or len(defect_parent_usage) != 168:
        raise RuntimeError("defect parent output reuse mismatch")
    expected_parent_profile = set(
        int(value)
        for value in config["source"]["expected_parent_reuse_per_lighting_profile"]
    )
    if (
        set(defect_parent_profile_usage.values()) != expected_parent_profile
        or len(defect_parent_profile_usage) != 168 * len(config["lighting_profiles"])
    ):
        raise RuntimeError("defect parent x lighting profile output reuse mismatch")
    if (
        set(defect_parent_slot_usage.values())
        != {int(config["source"]["expected_parent_reuse_per_placement_slot"])}
        or len(defect_parent_slot_usage) != 168 * int(config["instances_per_scene"])
    ):
        raise RuntimeError("defect parent x placement slot output reuse mismatch")
    if min(normal_parent_usage.values()) != 1 or max(normal_parent_usage.values()) != 2:
        raise RuntimeError("normal parent output reuse mismatch")

    summary_rows: list[dict[str, Any]] = []
    for class_name in config["classes"]:
        summary_rows.append(
            {"dimension": "component_status_class", "value": class_name, "count": class_counts[class_name]}
        )
    for profile in config["lighting_profiles"]:
        summary_rows.append(
            {"dimension": "lighting_profile_scene", "value": profile, "count": profile_scene_counts[profile]}
        )
        pair_values = list(pair_count_by_profile[profile].values())
        summary_rows.append(
            {
                "dimension": "class_pair_range",
                "value": profile,
                "count": f"{min(pair_values)}..{max(pair_values)}",
            }
        )
    summary_path = release_root / "annotations" / "summary.csv"
    write_csv(summary_path, summary_rows, ["dimension", "value", "count"])

    contact_sheet_paths: list[Path] = []
    for profile in config["lighting_profiles"]:
        contact_sheet_paths.append(
            create_profile_contact_sheet(
                context, scene_rows, instances_by_scene, profile, overlay=False
            )
        )
        contact_sheet_paths.append(
            create_profile_contact_sheet(
                context, scene_rows, instances_by_scene, profile, overlay=True
            )
        )
    overview_path = create_overview_contact_sheet(
        context, scene_rows, instances_by_scene
    )
    contact_sheet_paths.append(overview_path)

    tracked_files = [path for path in release_root.rglob("*") if path.is_file()]
    payload_bytes = sum(path.stat().st_size for path in tracked_files)
    maximum_payload = float(config["qc"]["maximum_new_payload_mib"]) * 1024 * 1024
    if payload_bytes > maximum_payload:
        raise RuntimeError(
            f"release payload {payload_bytes / 1024 / 1024:.2f} MiB exceeds gate"
        )
    release_path = release_root / "annotations" / "release.json"
    release_metadata = {
        "release": config["release"],
        "generator_version": config["generator_version"],
        "qc_gate_version": config["qc_gate_version"],
        "task_type": config["task_type"],
        "split": config["split"],
        "training_use": config["training_use"],
        "evaluation_eligible": config["evaluation_eligible"],
        "classification_eligible": config["classification_eligible"],
        "config_path": repository_relative(context.config_path),
        "config_sha256": context.config_sha256,
        "generator_script": repository_relative(Path(__file__)),
        "generator_script_sha256": sha256_file(Path(__file__)),
        "runtime_contract": current_runtime_versions(),
        "requirements_path": config["runtime_contract"]["requirements_path"],
        "requirements_sha256": config["runtime_contract"][
            "requirements_sha256"
        ],
        "helper_script_sha256": {
            helper["path"]: helper["sha256"]
            for helper in config["runtime_contract"]["helper_scripts"]
        },
        "scene_count": len(scene_rows),
        "instances_per_scene": int(config["instances_per_scene"]),
        "component_instance_count": len(instance_rows),
        "defect_annotation_count": len(defect_coco_annotations),
        "normal_proxy_instance_count": class_counts["normal_proxy"],
        "class_counts": {name: class_counts[name] for name in config["classes"]},
        "profile_scene_counts": {
            profile: profile_scene_counts[profile]
            for profile in config["lighting_profiles"]
        },
        "class_profile_counts": {
            name: {
                profile: class_profile_counts[(name, profile)]
                for profile in config["lighting_profiles"]
            }
            for name in config["classes"]
        },
        "class_profile_grid_cell_count_range": [
            min(class_grid_counts.values()),
            max(class_grid_counts.values()),
        ],
        "profile_grid_cell_counts": {
            profile: {
                str(cell): profile_grid_counts[(profile, cell)] for cell in range(8)
            }
            for profile in config["lighting_profiles"]
        },
        "defect_parent_count": len(defect_parent_usage),
        "defect_parent_reuse_range": [
            min(defect_parent_usage.values()),
            max(defect_parent_usage.values()),
        ],
        "defect_parent_profile_reuse_range": [
            min(defect_parent_profile_usage.values()),
            max(defect_parent_profile_usage.values()),
        ],
        "defect_parent_placement_slot_reuse_range": [
            min(defect_parent_slot_usage.values()),
            max(defect_parent_slot_usage.values()),
        ],
        "normal_parent_count": len(normal_parent_usage),
        "normal_parent_reuse_range": [
            min(normal_parent_usage.values()),
            max(normal_parent_usage.values()),
        ],
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
        "overview_contact_sheet_sha256": sha256_file(overview_path),
        "contact_sheet_sha256": {
            repository_relative(path): sha256_file(path) for path in contact_sheet_paths[:-1]
        },
        "unique_image_sha256_count": len(image_hashes),
        "tracked_payload_bytes_before_release_json": payload_bytes,
        "limitations": [
            "All scenes are train-only composites derived from one synthetic-restored physical base family.",
            "normal_proxy is paired-clean synthetic data and is not confirmed real OK data.",
            "This balanced pilot uses five distinct status classes per scene; it does not model normal-heavy or repeated-status production prevalence.",
            "These scenes must never be repartitioned into validation or test data.",
            "Existing ResNet-18 checkpoints do not process full multi-instance scenes.",
        ],
    }
    for _ in range(5):
        current_payload_bytes = sum(
            path.stat().st_size for path in release_root.rglob("*") if path.is_file()
        )
        if release_metadata.get("tracked_payload_bytes") == current_payload_bytes:
            break
        release_metadata["tracked_payload_bytes"] = current_payload_bytes
        write_json(release_path, release_metadata)
    else:
        raise RuntimeError("release payload byte count did not stabilize")
    final_payload_bytes = sum(
        path.stat().st_size for path in release_root.rglob("*") if path.is_file()
    )
    if final_payload_bytes != int(release_metadata["tracked_payload_bytes"]):
        raise RuntimeError("final release payload byte count mismatch")
    if final_payload_bytes > maximum_payload:
        raise RuntimeError(
            f"final release payload {final_payload_bytes / 1024 / 1024:.2f} MiB exceeds gate"
        )
    print(
        f"PASS generated_scenes={len(scene_rows)} instances={len(instance_rows)} "
        f"defects={len(defect_coco_annotations)} profiles={len(config['lighting_profiles'])} "
        f"train_only=YES evaluation_eligible=NO payload_mib="
        f"{final_payload_bytes / 1024 / 1024:.2f}",
        flush=True,
    )
    return release_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    context = load_context(args.config, args.release)
    generate_release(context, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
