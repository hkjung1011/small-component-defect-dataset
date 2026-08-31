from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "synthetic_v1_450.json"
DEFAULT_OUTPUT = ROOT / "synthetic" / "v1_450"

PALETTE = {
    0: (0, 0, 0),
    1: (255, 70, 70),
    2: (255, 180, 40),
    3: (80, 160, 255),
    4: (160, 90, 40),
    5: (255, 70, 220),
    6: (80, 220, 120),
    7: (180, 100, 255),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_sample_seed(global_seed: int, release: str, base_sha: str, label: str, ordinal: int) -> int:
    payload = f"{global_seed}|{release}|{base_sha}|{label}|{ordinal}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


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
    if radius <= 0:
        return mask.copy()
    kernel = radius * 2 + 1
    return np.array(
        Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(kernel))
    ) > 0


def build_rois(config: dict[str, Any]) -> dict[str, np.ndarray]:
    size = int(config["image_size"])
    raw = config["rois_normalized"]
    metal = polygon_mask(size, raw["metal_tab"])
    slot = ellipse_mask(size, raw["mount_slot"])
    metal &= ~slot
    body_full = polygon_mask(size, raw["body"])
    body_visible = body_full & ~metal & ~slot
    left = polygon_mask(size, raw["left_lead"])
    center = polygon_mask(size, raw["center_lead"])
    right = polygon_mask(size, raw["right_lead"])
    leads = left | center | right
    component = body_full | metal | leads
    return {
        "metal_tab": metal,
        "mount_slot": slot,
        "body_full": body_full,
        "body_visible": body_visible,
        "left_lead": left,
        "center_lead": center,
        "right_lead": right,
        "outer_leads": left | right,
        "all_leads": leads,
        "component": component,
        "body_damage_allowed": dilate(body_visible, 10),
        "lead_damage_allowed": dilate(left | right, 12),
    }


def choose_weighted(rng: random.Random, weights: dict[str, float]) -> str:
    labels = list(weights)
    values = [float(weights[label]) for label in labels]
    return rng.choices(labels, weights=values, k=1)[0]


def severity_scale(severity: str) -> float:
    return {"mild": 0.72, "moderate": 1.0, "severe": 1.35}[severity]


def random_point(mask: np.ndarray, rng: random.Random) -> tuple[int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("empty ROI")
    index = rng.randrange(len(xs))
    return int(xs[index]), int(ys[index])


def quadratic_points(
    start: tuple[float, float], control: tuple[float, float], end: tuple[float, float], count: int = 36
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for index in range(count):
        t = index / (count - 1)
        u = 1.0 - t
        x = u * u * start[0] + 2 * u * t * control[0] + t * t * end[0]
        y = u * u * start[1] + 2 * u * t * control[1] + t * t * end[1]
        points.append((x, y))
    return points


def antialiased_line(
    size: int, points: list[tuple[float, float]], width: float, blur: float = 0.0
) -> np.ndarray:
    scale = 4
    canvas = Image.new("L", (size * scale, size * scale), 0)
    draw = ImageDraw.Draw(canvas)
    scaled = [(round(x * scale), round(y * scale)) for x, y in points]
    draw.line(scaled, fill=255, width=max(1, round(width * scale)), joint="curve")
    canvas = canvas.resize((size, size), Image.Resampling.LANCZOS)
    if blur > 0:
        canvas = canvas.filter(ImageFilter.GaussianBlur(blur))
    return np.array(canvas, dtype=np.uint8)


def irregular_blob(
    size: int,
    center: tuple[int, int],
    radius_x: float,
    radius_y: float,
    rng: random.Random,
    vertices: int,
    feather: float,
) -> np.ndarray:
    scale = 4
    points: list[tuple[int, int]] = []
    phase = rng.uniform(0, math.tau)
    for index in range(vertices):
        angle = phase + math.tau * index / vertices
        radial = rng.uniform(0.72, 1.25)
        x = center[0] + math.cos(angle) * radius_x * radial
        y = center[1] + math.sin(angle) * radius_y * radial
        points.append((round(x * scale), round(y * scale)))
    canvas = Image.new("L", (size * scale, size * scale), 0)
    ImageDraw.Draw(canvas).polygon(points, fill=255)
    canvas = canvas.resize((size, size), Image.Resampling.LANCZOS)
    if feather > 0:
        canvas = canvas.filter(ImageFilter.GaussianBlur(feather))
    return np.array(canvas, dtype=np.uint8)


def apply_tint(image: np.ndarray, alpha: np.ndarray, color: tuple[int, int, int], opacity: float) -> np.ndarray:
    weight = (alpha.astype(np.float32) / 255.0 * opacity)[..., None]
    out = image.astype(np.float32) * (1.0 - weight) + np.array(color, dtype=np.float32) * weight
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_delta(image: np.ndarray, alpha: np.ndarray, delta: tuple[float, float, float], opacity: float) -> np.ndarray:
    weight = (alpha.astype(np.float32) / 255.0 * opacity)[..., None]
    out = image.astype(np.float32) + np.array(delta, dtype=np.float32) * weight
    return np.clip(out, 0, 255).astype(np.uint8)


def make_scratch(
    image: np.ndarray,
    rois: dict[str, np.ndarray],
    occupied: np.ndarray,
    rng: random.Random,
    severity: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    size = image.shape[0]
    allowed = rois["metal_tab"] & ~occupied
    scale = severity_scale(severity)
    count = 1 if severity == "mild" else rng.randint(1, 3)
    combined = np.zeros((size, size), dtype=np.uint8)
    widths: list[float] = []
    lengths: list[float] = []
    for _ in range(count):
        accepted = False
        for _attempt in range(80):
            start = random_point(allowed, rng)
            end = random_point(allowed, rng)
            length = math.dist(start, end)
            if 55 * scale <= length <= 175 * scale:
                accepted = True
                break
        if not accepted:
            continue
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        dx, dy = end[0] - start[0], end[1] - start[1]
        norm = max(1.0, math.hypot(dx, dy))
        curvature = rng.uniform(-0.10, 0.10) * length
        control = (midpoint[0] - dy / norm * curvature, midpoint[1] + dx / norm * curvature)
        width = rng.uniform(1.2, 3.4) * scale
        alpha = antialiased_line(size, quadratic_points(start, control, end), width, blur=rng.uniform(0.0, 0.7))
        alpha[~allowed] = 0
        combined = np.maximum(combined, alpha)
        widths.append(round(width, 3))
        lengths.append(round(length, 3))
    if not np.any(combined > 16):
        raise ValueError("scratch generator produced empty mask")
    if rng.random() < 0.72:
        delta = tuple([rng.uniform(-66, -30)] * 3)
    else:
        value = rng.uniform(24, 46)
        delta = (value, value, value * 0.9)
    output = apply_delta(image, combined, delta, rng.uniform(0.62, 0.90))
    mask = (combined > 18).astype(np.uint8)
    return output, mask, {"surface": "metal_tab", "count": count, "width_px": widths, "length_px": lengths}


def make_surface_spot(
    image: np.ndarray,
    rois: dict[str, np.ndarray],
    occupied: np.ndarray,
    rng: random.Random,
    severity: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    size = image.shape[0]
    allowed = rois["metal_tab"] & ~occupied
    scale = severity_scale(severity)
    count = 1 if severity == "mild" else rng.randint(1, 3)
    combined = np.zeros((size, size), dtype=np.uint8)
    radii: list[list[float]] = []
    for _ in range(count):
        center = random_point(allowed, rng)
        rx = rng.uniform(5, 14) * scale
        ry = rx * rng.uniform(0.55, 1.45)
        alpha = irregular_blob(size, center, rx, ry, rng, rng.randint(7, 12), rng.uniform(0.7, 2.0))
        alpha[~allowed] = 0
        combined = np.maximum(combined, alpha)
        radii.append([round(rx, 3), round(ry, 3)])
    color = rng.choice([(58, 55, 48), (82, 66, 45), (45, 48, 52)])
    output = apply_tint(image, combined, color, rng.uniform(0.24, 0.48))
    mask = (combined > 24).astype(np.uint8)
    return output, mask, {"surface": "metal_tab", "count": count, "radii_px": radii, "recipe_subtype": "unknown_dark_spot"}


def make_discoloration(
    image: np.ndarray,
    rois: dict[str, np.ndarray],
    occupied: np.ndarray,
    rng: random.Random,
    severity: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    size = image.shape[0]
    allowed = rois["metal_tab"] & ~occupied
    scale = severity_scale(severity)
    center = random_point(allowed, rng)
    rx = rng.uniform(28, 58) * scale
    ry = rng.uniform(20, 52) * scale
    alpha = irregular_blob(size, center, rx, ry, rng, rng.randint(9, 15), rng.uniform(8, 17))
    alpha[~allowed] = 0
    colors = [(155, 126, 72), (104, 126, 145), (120, 92, 70), (160, 145, 105)]
    output = apply_tint(image, alpha, rng.choice(colors), rng.uniform(0.12, 0.30))
    mask = (alpha > 18).astype(np.uint8)
    return output, mask, {"surface": "metal_tab", "span_px": [round(rx * 2, 3), round(ry * 2, 3)], "recipe_subtype": "broad_color_shift"}


def make_contamination(
    image: np.ndarray,
    rois: dict[str, np.ndarray],
    occupied: np.ndarray,
    rng: random.Random,
    severity: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    size = image.shape[0]
    allowed = (rois["metal_tab"] | rois["body_visible"]) & ~occupied
    scale = severity_scale(severity)
    center = random_point(allowed, rng)
    smear = irregular_blob(
        size,
        center,
        rng.uniform(14, 36) * scale,
        rng.uniform(8, 26) * scale,
        rng,
        rng.randint(8, 14),
        rng.uniform(2.5, 7.0),
    )
    specks = np.zeros((size, size), dtype=np.uint8)
    draw = ImageDraw.Draw(Image.fromarray(specks))
    # Draw on a persistent canvas because ImageDraw does not update a detached ndarray reliably.
    speck_canvas = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(speck_canvas)
    speck_count = round(rng.uniform(8, 25) * scale)
    for _ in range(speck_count):
        angle = rng.uniform(0, math.tau)
        distance = abs(rng.gauss(0, 18 * scale))
        x = round(center[0] + math.cos(angle) * distance)
        y = round(center[1] + math.sin(angle) * distance)
        radius = rng.randint(1, max(1, round(3 * scale)))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=rng.randint(110, 255))
    specks = np.array(speck_canvas.filter(ImageFilter.GaussianBlur(0.45)), dtype=np.uint8)
    combined = np.maximum((smear.astype(np.float32) * rng.uniform(0.35, 0.70)).astype(np.uint8), specks)
    combined[~allowed] = 0
    color = rng.choice([(65, 48, 31), (45, 43, 38), (91, 70, 42)])
    output = apply_tint(image, combined, color, rng.uniform(0.28, 0.58))
    mask = (combined > 18).astype(np.uint8)
    return output, mask, {"surface": "metal_tab_or_body", "speck_count": speck_count, "recipe_subtype": "smear_and_particles"}


def clone_background(image: np.ndarray, mask: np.ndarray, dx: int, feather: float = 1.2) -> np.ndarray:
    height, width = mask.shape
    ys, xs = np.indices((height, width))
    source_x = np.clip(xs + dx, 0, width - 1)
    replacement = image[ys, source_x]
    alpha = np.array(
        Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(feather)),
        dtype=np.float32,
    )[..., None] / 255.0
    output = image.astype(np.float32) * (1.0 - alpha) + replacement.astype(np.float32) * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def make_lead_breakage(
    image: np.ndarray,
    rois: dict[str, np.ndarray],
    occupied: np.ndarray,
    rng: random.Random,
    severity: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    side = rng.choice(["left", "right"])
    lead = rois[f"{side}_lead"] & ~occupied
    ys, xs = np.where(lead)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    scale = severity_scale(severity)
    fraction = min(0.72, rng.uniform(0.25, 0.48) * scale)
    cut_y = round(y1 - (y1 - y0) * fraction)
    jitter = rng.randint(1, 4)
    polygon = np.zeros_like(lead)
    canvas = Image.fromarray(polygon.astype(np.uint8) * 255)
    ImageDraw.Draw(canvas).polygon(
        [(x0, cut_y + jitter), ((x0 + x1) // 2, cut_y - jitter), (x1, cut_y + rng.randint(-2, 2)), (x1, y1 + 2), (x0, y1 + 2)],
        fill=255,
    )
    mask = (np.array(canvas) > 0) & lead
    if not np.any(mask):
        raise ValueError("lead breakage mask empty")
    output = clone_background(image, mask, -62 if side == "left" else 62, feather=1.0)
    fracture = np.zeros_like(mask, dtype=np.uint8)
    fracture[max(0, cut_y - 1):min(mask.shape[0], cut_y + 2), x0:x1 + 1] = lead[max(0, cut_y - 1):min(mask.shape[0], cut_y + 2), x0:x1 + 1]
    output = apply_tint(output, fracture * 255, (82, 76, 65), 0.35)
    return output, mask.astype(np.uint8), {"surface": f"{side}_lead", "removed_fraction": round(fraction, 4), "recipe_subtype": "distal_lead_missing"}


def make_body_chip(
    image: np.ndarray,
    rois: dict[str, np.ndarray],
    occupied: np.ndarray,
    rng: random.Random,
    severity: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    side = rng.choice(["left", "right"])
    body = rois["body_visible"] & ~occupied
    ys, xs = np.where(body)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    scale = severity_scale(severity)
    depth = round(rng.uniform(7, 18) * scale)
    span = round(rng.uniform(12, 34) * scale)
    center_y = rng.randint(max(y0 + 8, 145), min(y1 - 8, 286))
    if side == "left":
        points = [(x0 - 2, center_y - span // 2), (x0 + depth, center_y), (x0 - 2, center_y + span // 2)]
        dx = -55
    else:
        points = [(x1 + 2, center_y - span // 2), (x1 - depth, center_y), (x1 + 2, center_y + span // 2)]
        dx = 55
    canvas = Image.new("L", body.shape[::-1], 0)
    ImageDraw.Draw(canvas).polygon(points, fill=255)
    mask = (np.array(canvas) > 0) & body
    if not np.any(mask):
        raise ValueError("body chip mask empty")
    output = clone_background(image, mask, dx, feather=1.1)
    return output, mask.astype(np.uint8), {"surface": f"body_{side}_edge", "depth_px": depth, "span_px": span, "recipe_subtype": "edge_material_loss"}


def make_body_crack(
    image: np.ndarray,
    rois: dict[str, np.ndarray],
    occupied: np.ndarray,
    rng: random.Random,
    severity: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    size = image.shape[0]
    body = rois["body_visible"] & ~occupied
    scale = severity_scale(severity)
    side = rng.choice(["left", "right", "bottom"])
    if side == "left":
        start = (141, rng.randint(160, 280))
    elif side == "right":
        start = (331, rng.randint(160, 280))
    else:
        start = (rng.randint(165, 305), 300)
    end = random_point(body, rng)
    length = math.dist(start, end)
    midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    control = (midpoint[0] + rng.uniform(-14, 14), midpoint[1] + rng.uniform(-14, 14))
    width = rng.uniform(1.0, 2.4) * scale
    alpha = antialiased_line(size, quadratic_points(start, control, end, 28), width, blur=0.35)
    alpha[~body] = 0
    branch_count = 0 if severity == "mild" else rng.randint(0, 2)
    for _ in range(branch_count):
        branch_start = quadratic_points(start, control, end, 12)[rng.randint(4, 8)]
        branch_end = (branch_start[0] + rng.uniform(-28, 28) * scale, branch_start[1] + rng.uniform(10, 35) * scale)
        branch = antialiased_line(size, [branch_start, branch_end], max(1.0, width * 0.65), blur=0.3)
        branch[~body] = 0
        alpha = np.maximum(alpha, branch)
    if not np.any(alpha > 18):
        raise ValueError("body crack mask empty")
    output = apply_tint(image, alpha, (100, 96, 86), rng.uniform(0.38, 0.66))
    core = (alpha > 110).astype(np.uint8) * 255
    output = apply_tint(output, core, (8, 8, 7), 0.35)
    mask = (alpha > 18).astype(np.uint8)
    return output, mask, {"surface": "body_visible", "length_px": round(length, 3), "width_px": round(width, 3), "branches": branch_count, "recipe_subtype": "edge_started_crack"}


GENERATORS = {
    "scratch": make_scratch,
    "surface_spot": make_surface_spot,
    "discoloration": make_discoloration,
    "contamination": make_contamination,
    "lead_breakage": make_lead_breakage,
    "body_chip": make_body_chip,
    "body_crack": make_body_crack,
}


def recipe_allowed_roi(name: str, rois: dict[str, np.ndarray]) -> np.ndarray:
    if name in {"scratch", "surface_spot", "discoloration"}:
        return rois["metal_tab"]
    if name == "contamination":
        return rois["metal_tab"] | rois["body_visible"]
    if name == "lead_breakage":
        return rois["lead_damage_allowed"]
    if name in {"body_chip", "body_crack"}:
        return rois["body_damage_allowed"]
    raise KeyError(name)


def apply_recipe(
    base: Image.Image,
    recipe: list[str],
    severity: str,
    rois: dict[str, np.ndarray],
    class_ids: dict[str, int],
    rng: random.Random,
) -> tuple[Image.Image, Image.Image, list[dict[str, Any]], float]:
    image = np.array(base.convert("RGB"), dtype=np.uint8)
    semantic = np.zeros((base.height, base.width), dtype=np.uint8)
    instances: list[dict[str, Any]] = []
    inside_pixels = 0
    all_pixels = 0
    for name in recipe:
        occupied = semantic > 0
        output, mask, params = GENERATORS[name](image, rois, occupied, rng, severity)
        mask = (mask > 0) & ~occupied
        if not np.any(mask):
            raise ValueError(f"{name} produced empty non-overlapping mask")
        allowed = recipe_allowed_roi(name, rois)
        inside_pixels += int(np.count_nonzero(mask & allowed))
        all_pixels += int(np.count_nonzero(mask))
        image = np.where(mask[..., None], output, image)
        semantic[mask] = int(class_ids[name])
        instances.append({"category": name, "category_id": int(class_ids[name]), "severity": severity, **params})
    inside_ratio = 1.0 if all_pixels == 0 else inside_pixels / all_pixels
    return Image.fromarray(image), Image.fromarray(semantic, mode="L"), instances, inside_ratio


def geometry_transform(
    image: Image.Image,
    semantic: Image.Image,
    config: dict[str, Any],
    rng: random.Random,
) -> tuple[Image.Image, Image.Image, dict[str, float]]:
    size = int(config["image_size"])
    params = config["domain_randomization"]
    pad = 64
    image_array = np.pad(np.array(image), ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    mask_array = np.pad(np.array(semantic), ((pad, pad), (pad, pad)), mode="constant")
    padded_image = Image.fromarray(image_array)
    padded_mask = Image.fromarray(mask_array, mode="L")
    angle = rng.uniform(*params["rotation_deg"])
    rotated_image = padded_image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
    rotated_mask = padded_mask.rotate(angle, resample=Image.Resampling.NEAREST, expand=False, fillcolor=0)
    scale = rng.uniform(*params["scale"])
    crop_size = round(size / scale)
    center = (size + pad * 2) / 2
    tx = rng.uniform(*params["translation_px"])
    ty = rng.uniform(*params["translation_px"])
    left = round(center + tx - crop_size / 2)
    top = round(center + ty - crop_size / 2)
    box = (left, top, left + crop_size, top + crop_size)
    out_image = rotated_image.crop(box).resize((size, size), Image.Resampling.LANCZOS)
    out_mask = rotated_mask.crop(box).resize((size, size), Image.Resampling.NEAREST)
    return out_image, out_mask, {"rotation_deg": round(angle, 6), "scale": round(scale, 6), "translate_x": round(tx, 6), "translate_y": round(ty, 6)}


def photometric_transform(image: Image.Image, config: dict[str, Any], rng: random.Random, np_rng: np.random.Generator) -> tuple[Image.Image, dict[str, float]]:
    params = config["domain_randomization"]
    brightness = rng.uniform(*params["brightness"])
    contrast = rng.uniform(*params["contrast"])
    saturation = rng.uniform(*params["saturation"])
    blur = rng.uniform(*params["blur_radius"])
    noise_sigma = rng.uniform(*params["noise_sigma"])
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Color(image).enhance(saturation)
    array = np.array(image, dtype=np.float32)
    gains = np.array([rng.uniform(*params["channel_gain"]) for _ in range(3)], dtype=np.float32)
    array *= gains[None, None, :]
    if noise_sigma > 0:
        array += np_rng.normal(0.0, noise_sigma, array.shape)
    image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    if blur > 0.04:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    return image, {
        "brightness": round(brightness, 6),
        "contrast": round(contrast, 6),
        "saturation": round(saturation, 6),
        "channel_gain_r": round(float(gains[0]), 6),
        "channel_gain_g": round(float(gains[1]), 6),
        "channel_gain_b": round(float(gains[2]), 6),
        "blur_radius": round(blur, 6),
        "noise_sigma": round(noise_sigma, 6),
    }


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, 0, 0
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def make_overlay(image: Image.Image, mask: Image.Image) -> Image.Image:
    base = np.array(image.convert("RGB"), dtype=np.float32)
    labels = np.array(mask, dtype=np.uint8)
    overlay = base.copy()
    for class_id, color in PALETTE.items():
        if class_id == 0:
            continue
        region = labels == class_id
        if np.any(region):
            overlay[region] = overlay[region] * 0.45 + np.array(color) * 0.55
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def create_contact_sheet(output: Path, rows: list[dict[str, str]], classes: list[str]) -> None:
    selected: list[dict[str, str]] = []
    for label in classes:
        selected.extend([row for row in rows if row["primary_class"] == label][:3])
    tile = 192
    label_height = 22
    columns = 3
    sheet = Image.new("RGB", (columns * tile * 2, len(classes) * (tile + label_height)), (28, 28, 28))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for row_index, label in enumerate(classes):
        label_rows = [row for row in selected if row["primary_class"] == label]
        for column, row in enumerate(label_rows):
            image = Image.open(ROOT / row["image_path"]).convert("RGB")
            mask = Image.open(ROOT / row["mask_path"]).convert("L")
            overlay = make_overlay(image, mask)
            image = image.resize((tile, tile), Image.Resampling.LANCZOS)
            overlay = overlay.resize((tile, tile), Image.Resampling.LANCZOS)
            x = column * tile * 2
            y = row_index * (tile + label_height) + label_height
            sheet.paste(image, (x, y))
            sheet.paste(overlay, (x + tile, y))
        draw.text((5, row_index * (tile + label_height) + 5), f"{label}: image | mask overlay", fill=(245, 245, 245), font=font)
    sheet.save(output / "contact_sheet.jpg", quality=92, subsampling=0)


def prepare_output(path: Path, force: bool) -> None:
    marker = path / ".synthetic_release_marker"
    if path.exists():
        if not force:
            raise FileExistsError(f"output already exists: {path}; use --force to regenerate")
        resolved = path.resolve()
        synthetic_root = (ROOT / "synthetic").resolve()
        if synthetic_root not in resolved.parents or not marker.exists():
            raise RuntimeError(f"refusing to remove unmarked or out-of-scope directory: {resolved}")
        shutil.rmtree(path)
    (path / "images").mkdir(parents=True)
    (path / "masks").mkdir(parents=True)
    (path / "annotations").mkdir(parents=True)
    marker.write_text("synthetic release generated by scripts/generate_synthetic.py\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic-v1-450 defect data with masks and labels."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-per-class", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if args.samples_per_class is not None:
        if args.samples_per_class <= 0:
            raise ValueError("--samples-per-class must be positive")
        config["samples_per_primary_class"] = args.samples_per_class
        config_bytes = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    config_sha = sha256_bytes(config_bytes)
    output = args.output.resolve()
    prepare_output(output, args.force)

    base_path = ROOT / config["base"]["path"]
    base = Image.open(base_path).convert("RGB")
    expected_size = int(config["image_size"])
    if base.size != (expected_size, expected_size):
        raise ValueError(f"base size {base.size} does not match config {expected_size}")
    base_sha = sha256_file(base_path)
    rois = build_rois(config)
    class_ids = {key: int(value) for key, value in config["class_ids"].items()}
    primary_classes = list(config["primary_classes"])
    per_class = int(config["samples_per_primary_class"])
    rows: list[dict[str, str]] = []
    instance_records: list[dict[str, Any]] = []

    for primary_class in primary_classes:
        (output / "images" / primary_class).mkdir(parents=True, exist_ok=True)
        (output / "masks" / primary_class).mkdir(parents=True, exist_ok=True)
        for ordinal in range(per_class):
            sample_seed = stable_sample_seed(
                int(config["global_seed"]), config["release"], base_sha, primary_class, ordinal
            )
            successful = False
            last_error: Exception | None = None
            for attempt in range(20):
                effective_seed = sample_seed + attempt * 104729
                rng = random.Random(effective_seed)
                np_rng = np.random.default_rng(effective_seed)
                severity = "none" if primary_class == "normal_proxy" else choose_weighted(rng, config["severity_weights"])
                if primary_class == "normal_proxy":
                    recipe: list[str] = []
                elif primary_class == "multi_defect":
                    recipe = list(rng.choice(config["multi_defect_pairs"]))
                else:
                    recipe = [primary_class]
                try:
                    generated, semantic, instances, inside_ratio = apply_recipe(
                        base, recipe, severity, rois, class_ids, rng
                    )
                    pre_mask = np.array(semantic, dtype=np.uint8) > 0
                    if np.any(pre_mask):
                        pre_delta = np.abs(
                            np.array(generated, dtype=np.float32)
                            - np.array(base, dtype=np.float32)
                        )
                        pre_mean_abs_delta = float(pre_delta[pre_mask].mean())
                    else:
                        pre_mean_abs_delta = 0.0
                    minimum_delta = float(
                        config["minimum_pre_transform_mean_abs_delta"][primary_class]
                    )
                    if pre_mean_abs_delta < minimum_delta:
                        raise ValueError(
                            f"visible delta {pre_mean_abs_delta:.3f} below {minimum_delta:.3f}"
                        )
                    generated, semantic, geometry = geometry_transform(generated, semantic, config, rng)
                    generated, photometric = photometric_transform(generated, config, rng, np_rng)
                    mask_array = np.array(semantic, dtype=np.uint8)
                    if recipe and not np.any(mask_array):
                        raise ValueError("defect mask disappeared after transform")
                    if not recipe and np.any(mask_array):
                        raise ValueError("normal proxy contains non-zero mask")
                    successful = True
                    break
                except Exception as error:  # deterministic retry with a derived seed
                    last_error = error
            if not successful:
                raise RuntimeError(f"failed to generate {primary_class} #{ordinal}: {last_error}")

            sample_id = f"{config['sample_id_prefix']}-{primary_class}-{ordinal:04d}"
            image_path = output / "images" / primary_class / f"{sample_id}.jpg"
            mask_path = output / "masks" / primary_class / f"{sample_id}.png"
            generated.save(
                image_path,
                format="JPEG",
                quality=int(config["jpeg_quality"]),
                subsampling=0,
                optimize=False,
                progressive=False,
            )
            semantic.save(mask_path, format="PNG", optimize=False)
            mask_array = np.array(semantic, dtype=np.uint8)
            present_ids = sorted(int(item) for item in np.unique(mask_array) if item != 0)
            id_to_name = {value: key for key, value in class_ids.items() if key != "background"}
            labels = [id_to_name[item] for item in present_ids]
            bbox = mask_bbox(mask_array)
            defect_pixels = int(np.count_nonzero(mask_array))
            per_class_instances: list[dict[str, Any]] = []
            for item in present_ids:
                class_mask = mask_array == item
                per_class_instances.append(
                    {
                        "category": id_to_name[item],
                        "category_id": item,
                        "area_px": int(np.count_nonzero(class_mask)),
                        "bbox_xywh": list(mask_bbox(class_mask)),
                    }
                )
            parameter_record = {
                "attempt": attempt,
                "effective_seed": effective_seed,
                "recipe": recipe,
                "severity": severity,
                "geometry": geometry,
                "photometric": photometric,
                "instances_pre_transform": instances,
            }
            image_rel = image_path.relative_to(ROOT).as_posix()
            mask_rel = mask_path.relative_to(ROOT).as_posix()
            training_use = "TRAIN_ONLY_PSEUDO_NORMAL" if primary_class == "normal_proxy" else "TRAIN_ONLY_SYNTHETIC_NG"
            row = {
                "sample_id": sample_id,
                "image_path": image_rel,
                "mask_path": mask_rel,
                "domain": config["source_domain"],
                "split": config["split"],
                "base_image_id": config["base"]["base_image_id"],
                "base_group_id": config["base"]["base_group_id"],
                "source_specimen_group": config["base"]["source_specimen_group"],
                "view": config["base"]["view"],
                "primary_class": primary_class,
                "visible_multilabel": ";".join(labels),
                "severity": severity,
                "global_seed": str(config["global_seed"]),
                "sample_seed": str(sample_seed),
                "generator_version": config["generator_version"],
                "config_sha256": config_sha,
                "base_sha256": base_sha,
                "image_sha256": sha256_file(image_path),
                "mask_sha256": sha256_file(mask_path),
                "width": str(expected_size),
                "height": str(expected_size),
                "defect_pixels": str(defect_pixels),
                "mask_area_ratio_image": f"{defect_pixels / (expected_size * expected_size):.10f}",
                "roi_inside_ratio": f"{inside_ratio:.10f}",
                "pre_transform_mean_abs_delta": f"{pre_mean_abs_delta:.10f}",
                "bbox_x": str(bbox[0]),
                "bbox_y": str(bbox[1]),
                "bbox_w": str(bbox[2]),
                "bbox_h": str(bbox[3]),
                "training_use": training_use,
                "evaluation_eligible": config["evaluation_eligible"],
                "qc_status": "AUTO_PASS",
                "human_verified": "NO",
                "parameters_json": json.dumps(parameter_record, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            }
            rows.append(row)
            instance_records.append(
                {
                    "sample_id": sample_id,
                    "primary_class": primary_class,
                    "visible_multilabel": labels,
                    "semantic_mask_path": mask_rel,
                    "instances": per_class_instances,
                }
            )

    manifest_path = output / "annotations" / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    with (output / "annotations" / "instances.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for record in instance_records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    counts = Counter(row["primary_class"] for row in rows)
    severities = Counter(row["severity"] for row in rows)
    with (output / "annotations" / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["axis", "value", "count"])
        for key in primary_classes:
            writer.writerow(["primary_class", key, counts[key]])
        for key in ["none", "mild", "moderate", "severe"]:
            writer.writerow(["severity", key, severities[key]])
    release = {
        "release": config["release"],
        "generator_version": config["generator_version"],
        "generator_script": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "generator_script_sha256": sha256_file(Path(__file__).resolve()),
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": config_sha,
        "base_path": base_path.relative_to(ROOT).as_posix(),
        "base_sha256": base_sha,
        "source_domain": config["source_domain"],
        "split": config["split"],
        "sample_count": len(rows),
        "manifest_sha256": sha256_file(manifest_path),
        "instances_sha256": sha256_file(output / "annotations" / "instances.jsonl"),
        "summary_sha256": sha256_file(output / "annotations" / "summary.csv"),
        "class_counts": dict(counts),
        "notes": [
            "This release is train-only synthetic data.",
            "normal_proxy is restored synthetic imagery, not a confirmed real normal specimen.",
            "Real-only validation/test performance remains NOT VERIFIED.",
        ],
    }
    with (output / "annotations" / "release.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        stream.write(json.dumps(release, ensure_ascii=False, indent=2) + "\n")
    create_contact_sheet(output, rows, primary_classes)
    print(
        f"PASS: generated={len(rows)}, classes={len(primary_classes)}, "
        f"per_class={per_class}, output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
