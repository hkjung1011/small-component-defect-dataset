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
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

import generate_synthetic_v1_450 as legacy


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "synthetic_v2_700.json"
DEFAULT_OUTPUT = ROOT / "synthetic" / "v2_700"
MARKER_NAME = ".synthetic_v2_release_marker"


def sha256_file(path: Path) -> str:
    return legacy.sha256_file(path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    kernel = radius * 2 + 1
    return np.array(
        Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(kernel))
    ) > 0


def choose_contrast_color(
    image: np.ndarray,
    region: np.ndarray,
    palette: list[tuple[int, int, int]],
    rng: random.Random,
    jitter: int = 10,
) -> tuple[int, int, int]:
    if np.any(region):
        mean = image[region].astype(np.float32).mean(axis=0)
    else:
        mean = image.reshape(-1, 3).astype(np.float32).mean(axis=0)
    candidates: list[tuple[int, int, int]] = []
    for color in palette:
        candidates.append(
            tuple(int(np.clip(channel + rng.randint(-jitter, jitter), 0, 255)) for channel in color)
        )
    return max(candidates, key=lambda color: float(np.linalg.norm(np.array(color) - mean)))


def make_scratch_visible(
    image: np.ndarray,
    rois: dict[str, np.ndarray],
    occupied: np.ndarray,
    rng: random.Random,
    severity: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    size = image.shape[0]
    allowed = rois["metal_tab"] & ~occupied
    length_range, width_range, count_range = {
        "mild": ((58.0, 96.0), (2.6, 3.8), (1, 1)),
        "moderate": ((82.0, 138.0), (3.5, 5.0), (1, 2)),
        "severe": ((115.0, 180.0), (4.8, 6.8), (2, 3)),
    }[severity]
    combined = np.zeros((size, size), dtype=np.uint8)
    lengths: list[float] = []
    widths: list[float] = []
    count = rng.randint(*count_range)
    for _ in range(count):
        for _attempt in range(160):
            start = legacy.random_point(allowed, rng)
            end = legacy.random_point(allowed, rng)
            length = math.dist(start, end)
            if length_range[0] <= length <= length_range[1]:
                break
        else:
            raise ValueError("unable to place scratch with required length")
        midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
        dx, dy = end[0] - start[0], end[1] - start[1]
        norm = max(1.0, math.hypot(dx, dy))
        curvature = rng.uniform(-0.08, 0.08) * length
        control = (midpoint[0] - dy / norm * curvature, midpoint[1] + dx / norm * curvature)
        width = rng.uniform(*width_range)
        alpha = legacy.antialiased_line(
            size,
            legacy.quadratic_points(start, control, end, 42),
            width,
            blur=rng.uniform(0.15, 0.5),
        )
        alpha[~allowed] = 0
        combined = np.maximum(combined, alpha)
        lengths.append(round(length, 3))
        widths.append(round(width, 3))
    if not np.any(combined > 20):
        raise ValueError("scratch mask empty")
    color = choose_contrast_color(
        image,
        combined > 100,
        [(34, 34, 32), (58, 46, 35), (26, 38, 46)],
        rng,
        jitter=6,
    )
    opacity = rng.uniform(*{"mild": (0.74, 0.86), "moderate": (0.78, 0.90), "severe": (0.82, 0.94)}[severity])
    output = legacy.apply_tint(image, combined, color, opacity)
    mask = (combined > 20).astype(np.uint8)
    return output, mask, {
        "surface": "metal_tab",
        "count": count,
        "length_px": lengths,
        "width_px": widths,
        "opacity": round(opacity, 4),
        "color_rgb": list(color),
        "recipe_subtype": "high_visibility_curved_scratch",
    }


def make_surface_spot(
    image: np.ndarray,
    rois: dict[str, np.ndarray],
    occupied: np.ndarray,
    rng: random.Random,
    severity: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    size = image.shape[0]
    allowed = erode(rois["metal_tab"], 9) & ~occupied
    ranges = {
        "mild": ((9.0, 13.0), 1),
        "moderate": ((14.0, 22.0), rng.randint(1, 2)),
        "severe": ((23.0, 34.0), rng.randint(2, 3)),
    }
    radius_range, count = ranges[severity]
    combined = np.zeros((size, size), dtype=np.uint8)
    radii: list[list[float]] = []
    for _ in range(count):
        rx = rng.uniform(*radius_range)
        ry = rx * rng.uniform(0.65, 1.35)
        # A large Pillow MinFilter is prohibitively slow.  A fixed interior seed
        # band plus the post-transform area gate safely rejects excessive clipping.
        safe = erode(allowed, 12)
        center = legacy.random_point(safe if np.any(safe) else allowed, rng)
        alpha = legacy.irregular_blob(
            size,
            center,
            rx,
            ry,
            rng,
            rng.randint(9, 16),
            rng.uniform(1.0, 2.6),
        )
        alpha[~allowed] = 0
        combined = np.maximum(combined, alpha)
        radii.append([round(rx, 3), round(ry, 3)])
    if not np.any(combined > 28):
        raise ValueError("surface spot mask empty")
    color = choose_contrast_color(
        image,
        combined > 110,
        [(50, 48, 44), (84, 56, 28), (38, 48, 58), (105, 65, 32)],
        rng,
    )
    opacity = rng.uniform(*{"mild": (0.58, 0.70), "moderate": (0.64, 0.78), "severe": (0.70, 0.86)}[severity])
    output = legacy.apply_tint(image, combined, color, opacity)
    mask = (combined > 28).astype(np.uint8)
    return output, mask, {
        "surface": "metal_tab_interior",
        "count": count,
        "radii_px": radii,
        "opacity": round(opacity, 4),
        "color_rgb": list(color),
        "recipe_subtype": "contrast_adaptive_irregular_spot",
    }


def make_discoloration(
    image: np.ndarray,
    rois: dict[str, np.ndarray],
    occupied: np.ndarray,
    rng: random.Random,
    severity: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    size = image.shape[0]
    allowed = erode(rois["metal_tab"], 5) & ~occupied
    dimensions = {
        "mild": ((35.0, 52.0), (28.0, 44.0), (0.38, 0.52)),
        "moderate": ((50.0, 72.0), (40.0, 62.0), (0.48, 0.64)),
        "severe": ((68.0, 96.0), (54.0, 82.0), (0.58, 0.76)),
    }
    rx_range, ry_range, opacity_range = dimensions[severity]
    center = legacy.random_point(erode(allowed, 12), rng)
    rx = rng.uniform(*rx_range)
    ry = rng.uniform(*ry_range)
    alpha = legacy.irregular_blob(
        size,
        center,
        rx,
        ry,
        rng,
        rng.randint(11, 18),
        rng.uniform(5.0, 11.0),
    )
    alpha[~allowed] = 0
    color = choose_contrast_color(
        image,
        alpha > 100,
        [(154, 105, 48), (82, 120, 158), (128, 72, 54), (176, 147, 82), (88, 138, 116)],
        rng,
        jitter=14,
    )
    opacity = rng.uniform(*opacity_range)
    output = legacy.apply_tint(image, alpha, color, opacity)
    mask = (alpha > 25).astype(np.uint8)
    if not np.any(mask):
        raise ValueError("discoloration mask empty")
    return output, mask, {
        "surface": "metal_tab",
        "span_px": [round(rx * 2, 3), round(ry * 2, 3)],
        "opacity": round(opacity, 4),
        "color_rgb": list(color),
        "recipe_subtype": "high_visibility_broad_color_shift",
    }


def make_contamination(
    image: np.ndarray,
    rois: dict[str, np.ndarray],
    occupied: np.ndarray,
    rng: random.Random,
    severity: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    size = image.shape[0]
    surface_name = rng.choice(["metal_tab", "body_visible"])
    allowed = erode(rois[surface_name], 8) & ~occupied
    geometry = {
        "mild": ((16.0, 26.0), (10.0, 18.0), (3, 7), (0.58, 0.72)),
        "moderate": ((26.0, 42.0), (18.0, 30.0), (8, 16), (0.66, 0.80)),
        "severe": ((42.0, 64.0), (28.0, 46.0), (16, 28), (0.72, 0.88)),
    }
    rx_range, ry_range, speck_range, opacity_range = geometry[severity]
    center = legacy.random_point(erode(allowed, 8), rng)
    rx = rng.uniform(*rx_range)
    ry = rng.uniform(*ry_range)
    smear = legacy.irregular_blob(
        size,
        center,
        rx,
        ry,
        rng,
        rng.randint(10, 18),
        rng.uniform(2.0, 5.0),
    )
    speck_canvas = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(speck_canvas)
    speck_count = rng.randint(*speck_range)
    radii: list[int] = []
    for _ in range(speck_count):
        angle = rng.uniform(0.0, math.tau)
        distance = abs(rng.gauss(0.0, max(rx, ry) * 0.65))
        x = round(center[0] + math.cos(angle) * distance)
        y = round(center[1] + math.sin(angle) * distance)
        radius = rng.randint(4, 8 if severity != "severe" else 10)
        radii.append(radius)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=rng.randint(170, 255))
    specks = np.array(speck_canvas.filter(ImageFilter.GaussianBlur(0.55)), dtype=np.uint8)
    combined = np.maximum((smear.astype(np.float32) * rng.uniform(0.62, 0.82)).astype(np.uint8), specks)
    combined[~allowed] = 0
    local_region = combined > 90
    local_luma = float(
        np.mean(image[local_region] @ np.array([0.2126, 0.7152, 0.0722]))
    ) if np.any(local_region) else 128.0
    if local_luma < 105.0:
        palette = [(205, 188, 150), (176, 144, 88), (164, 170, 164), (195, 119, 62)]
        polarity = "light_on_dark"
    else:
        palette = [(54, 40, 28), (72, 48, 25), (36, 42, 38), (96, 58, 25)]
        polarity = "dark_on_light"
    color = choose_contrast_color(image, local_region, palette, rng, jitter=12)
    opacity = rng.uniform(*opacity_range)
    output = legacy.apply_tint(image, combined, color, opacity)
    mask = (combined > 24).astype(np.uint8)
    if not np.any(mask):
        raise ValueError("contamination mask empty")
    return output, mask, {
        "surface": surface_name,
        "span_px": [round(rx * 2, 3), round(ry * 2, 3)],
        "speck_count": speck_count,
        "speck_radii_px": radii,
        "polarity": polarity,
        "local_luma": round(local_luma, 3),
        "opacity": round(opacity, 4),
        "color_rgb": list(color),
        "recipe_subtype": "contrast_adaptive_smear_and_particles",
    }


def make_body_chip(
    image: np.ndarray,
    rois: dict[str, np.ndarray],
    occupied: np.ndarray,
    rng: random.Random,
    severity: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    body = rois["body_visible"] & ~occupied
    ys, xs = np.where(body)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    depth_range, span_range = {
        "mild": ((13, 19), (24, 36)),
        "moderate": ((22, 32), (40, 62)),
        "severe": ((35, 49), (66, 92)),
    }[severity]
    depth = rng.randint(*depth_range)
    span = rng.randint(*span_range)
    half = span // 2
    lower = y0 + half + 8
    upper = y1 - half - 8
    if lower >= upper:
        raise ValueError("body chip span exceeds available edge")
    center_y = rng.randint(lower, upper)
    side = rng.choice(["left", "right"])
    y_values = np.linspace(center_y - half, center_y + half, 7)
    profile = [0.0, rng.uniform(0.25, 0.42), rng.uniform(0.55, 0.78), 1.0,
               rng.uniform(0.52, 0.76), rng.uniform(0.22, 0.45), 0.0]
    if side == "left":
        points = [(x0 - 3 if index in {0, 6} else round(x0 + depth * profile[index]), round(y))
                  for index, y in enumerate(y_values)]
        dx = -rng.randint(55, 82)
    else:
        points = [(x1 + 3 if index in {0, 6} else round(x1 - depth * profile[index]), round(y))
                  for index, y in enumerate(y_values)]
        dx = rng.randint(55, 82)
    canvas = Image.new("L", body.shape[::-1], 0)
    ImageDraw.Draw(canvas).polygon(points, fill=255)
    mask = (np.array(canvas, dtype=np.uint8) > 0) & body
    if not np.any(mask):
        raise ValueError("body chip mask empty")
    output = legacy.clone_background(image, mask, dx, feather=0.8)
    rim = mask & ~erode(mask, 2)
    fracture_color = choose_contrast_color(image, rim, [(118, 103, 82), (72, 69, 62)], rng, jitter=8)
    output = legacy.apply_tint(output, rim.astype(np.uint8) * 255, fracture_color, rng.uniform(0.35, 0.55))
    return output, mask.astype(np.uint8), {
        "surface": f"body_{side}_edge",
        "depth_px": depth,
        "span_px": span,
        "vertices": len(points),
        "clone_offset_px": dx,
        "recipe_subtype": "irregular_edge_material_loss",
    }


def make_lead_breakage_visible(
    image: np.ndarray,
    rois: dict[str, np.ndarray],
    occupied: np.ndarray,
    rng: random.Random,
    severity: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    output, mask, params = legacy.make_lead_breakage(image, rois, occupied, rng, severity)
    removed = mask > 0
    # The fabric background and oxidized lead can have similar gray values.  A
    # localized cavity shadow makes the missing silhouette survive 224 px while
    # preserving the semantics of removed material instead of lowering the gate.
    shadow_color = choose_contrast_color(
        image,
        removed,
        [(72, 68, 60), (88, 78, 64), (54, 58, 58)],
        rng,
        jitter=6,
    )
    shadow_opacity = rng.uniform(0.46, 0.62)
    output = legacy.apply_tint(
        output,
        removed.astype(np.uint8) * 255,
        shadow_color,
        shadow_opacity,
    )
    params.update(
        {
            "cavity_shadow_rgb": list(shadow_color),
            "cavity_shadow_opacity": round(shadow_opacity, 4),
            "recipe_subtype": "distal_lead_missing_with_cavity_shadow",
        }
    )
    return output, mask, params


def make_body_crack(
    image: np.ndarray,
    rois: dict[str, np.ndarray],
    occupied: np.ndarray,
    rng: random.Random,
    severity: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    size = image.shape[0]
    body = erode(rois["body_visible"], 5) & ~occupied
    chord_range, width_range, branches_range, knot_range = {
        "mild": ((45.0, 72.0), (3.0, 4.2), (0, 1), (4, 5)),
        "moderate": ((78.0, 118.0), (4.3, 6.0), (1, 2), (5, 6)),
        "severe": ((125.0, 174.0), (6.0, 8.2), (2, 4), (6, 7)),
    }[severity]
    start = end = (0, 0)
    for _ in range(240):
        candidate_start = legacy.random_point(body, rng)
        candidate_end = legacy.random_point(body, rng)
        chord = math.dist(candidate_start, candidate_end)
        if chord_range[0] <= chord <= chord_range[1]:
            start, end = candidate_start, candidate_end
            break
    else:
        raise ValueError("unable to place body crack with required chord")
    dx, dy = end[0] - start[0], end[1] - start[1]
    chord = max(1.0, math.hypot(dx, dy))
    nx, ny = -dy / chord, dx / chord
    knot_count = rng.randint(*knot_range)
    points: list[tuple[float, float]] = []
    for index in range(knot_count):
        t = index / (knot_count - 1)
        base_x = start[0] + dx * t
        base_y = start[1] + dy * t
        offset = 0.0 if index in {0, knot_count - 1} else rng.uniform(-0.08, 0.08) * chord
        x = base_x + nx * offset
        y = base_y + ny * offset
        xi, yi = int(np.clip(round(x), 0, size - 1)), int(np.clip(round(y), 0, size - 1))
        if not body[yi, xi]:
            x, y = base_x, base_y
        points.append((x, y))
    width = rng.uniform(*width_range)
    alpha = legacy.antialiased_line(size, points, width, blur=rng.uniform(0.25, 0.55))
    branch_count = rng.randint(*branches_range)
    for _ in range(branch_count):
        branch_start = points[rng.randint(1, len(points) - 2)]
        branch_length = rng.uniform(18.0, 35.0) * (1.0 if severity == "mild" else 1.25)
        base_angle = math.atan2(dy, dx)
        branch_angle = base_angle + rng.choice([-1, 1]) * rng.uniform(math.radians(28), math.radians(68))
        branch_end = (
            branch_start[0] + math.cos(branch_angle) * branch_length,
            branch_start[1] + math.sin(branch_angle) * branch_length,
        )
        branch = legacy.antialiased_line(
            size,
            [branch_start, branch_end],
            max(1.8, width * rng.uniform(0.48, 0.68)),
            blur=0.3,
        )
        branch[~body] = 0
        alpha = np.maximum(alpha, branch)
    alpha[~body] = 0
    if not np.any(alpha > 18):
        raise ValueError("body crack mask empty")
    light_color = choose_contrast_color(image, alpha > 90, [(120, 112, 96), (148, 134, 104)], rng, jitter=10)
    output = legacy.apply_tint(image, alpha, light_color, rng.uniform(0.68, 0.88))
    core = (alpha > 165).astype(np.uint8) * 255
    output = legacy.apply_tint(output, core, (18, 18, 16), rng.uniform(0.16, 0.28))
    mask = (alpha > 18).astype(np.uint8)
    path_length = sum(math.dist(points[index - 1], points[index]) for index in range(1, len(points)))
    return output, mask, {
        "surface": "body_visible_interior",
        "chord_length_px": round(chord, 3),
        "path_length_px": round(path_length, 3),
        "width_px": round(width, 3),
        "branches": branch_count,
        "knots": knot_count,
        "recipe_subtype": "irregular_tapered_body_crack",
    }


GENERATORS = {
    "scratch": make_scratch_visible,
    "surface_spot": make_surface_spot,
    "discoloration": make_discoloration,
    "contamination": make_contamination,
    "lead_breakage": make_lead_breakage_visible,
    "body_chip": make_body_chip,
    "body_crack": make_body_crack,
}


def apply_single_recipe(
    base: Image.Image,
    class_name: str,
    severity: str,
    rois: dict[str, np.ndarray],
    class_ids: dict[str, int],
    rng: random.Random,
) -> tuple[Image.Image, Image.Image, dict[str, Any], float, float]:
    image = np.array(base.convert("RGB"), dtype=np.uint8)
    output, mask, params = GENERATORS[class_name](
        image, rois, np.zeros(mask_shape := image.shape[:2], dtype=bool), rng, severity
    )
    mask = mask > 0
    if not np.any(mask):
        raise ValueError(f"{class_name} produced empty mask")
    allowed = legacy.recipe_allowed_roi(class_name, rois)
    inside_ratio = float(np.count_nonzero(mask & allowed) / np.count_nonzero(mask))
    generated_array = np.where(mask[..., None], output, image)
    pre_delta = np.abs(generated_array.astype(np.float32) - image.astype(np.float32))
    pre_mean_abs_delta = float(pre_delta[mask].mean())
    semantic = np.zeros(mask_shape, dtype=np.uint8)
    semantic[mask] = int(class_ids[class_name])
    return (
        Image.fromarray(generated_array),
        Image.fromarray(semantic, mode="L"),
        params,
        inside_ratio,
        pre_mean_abs_delta,
    )


def sample_geometry(config: dict[str, Any], rng: random.Random) -> dict[str, float]:
    params = config["domain_randomization"]
    return {
        "rotation_deg": rng.uniform(*params["rotation_deg"]),
        "scale": rng.uniform(*params["scale"]),
        "translate_x": rng.uniform(*params["translation_px"]),
        "translate_y": rng.uniform(*params["translation_px"]),
    }


def apply_geometry(
    image: Image.Image,
    semantic: Image.Image,
    geometry: dict[str, float],
    size: int,
) -> tuple[Image.Image, Image.Image]:
    pad = 64
    image_array = np.pad(np.array(image), ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    mask_array = np.pad(np.array(semantic), ((pad, pad), (pad, pad)), mode="constant")
    padded_image = Image.fromarray(image_array)
    padded_mask = Image.fromarray(mask_array, mode="L")
    rotated_image = padded_image.rotate(
        float(geometry["rotation_deg"]), resample=Image.Resampling.BICUBIC, expand=False
    )
    rotated_mask = padded_mask.rotate(
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
    box = (left, top, left + crop_size, top + crop_size)
    return (
        rotated_image.crop(box).resize((size, size), Image.Resampling.LANCZOS),
        rotated_mask.crop(box).resize((size, size), Image.Resampling.NEAREST),
    )


def sample_photometric(config: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    params = config["domain_randomization"]
    return {
        "brightness": rng.uniform(*params["brightness"]),
        "contrast": rng.uniform(*params["contrast"]),
        "saturation": rng.uniform(*params["saturation"]),
        "channel_gain_r": rng.uniform(*params["channel_gain"]),
        "channel_gain_g": rng.uniform(*params["channel_gain"]),
        "channel_gain_b": rng.uniform(*params["channel_gain"]),
        "blur_radius": rng.uniform(*params["blur_radius"]),
        "noise_sigma": rng.uniform(*params["noise_sigma"]),
        "noise_seed": rng.randrange(0, 2**32),
    }


def apply_photometric(image: Image.Image, params: dict[str, Any]) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(float(params["brightness"]))
    image = ImageEnhance.Contrast(image).enhance(float(params["contrast"]))
    image = ImageEnhance.Color(image).enhance(float(params["saturation"]))
    array = np.array(image, dtype=np.float32)
    gains = np.array(
        [params["channel_gain_r"], params["channel_gain_g"], params["channel_gain_b"]],
        dtype=np.float32,
    )
    array *= gains[None, None, :]
    noise_sigma = float(params["noise_sigma"])
    if noise_sigma > 0:
        noise_rng = np.random.default_rng(int(params["noise_seed"]))
        array += noise_rng.normal(0.0, noise_sigma, array.shape)
    image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    blur = float(params["blur_radius"])
    if blur > 0.04:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    return image


def jpeg_roundtrip(image: Image.Image, quality: int) -> tuple[Image.Image, bytes]:
    stream = io.BytesIO()
    image.convert("RGB").save(
        stream,
        format="JPEG",
        quality=quality,
        subsampling=0,
        optimize=False,
        progressive=False,
    )
    payload = stream.getvalue()
    decoded = Image.open(io.BytesIO(payload)).convert("RGB")
    decoded.load()
    return decoded, payload


def rgb_pixels_to_lab(pixels: np.ndarray) -> np.ndarray:
    rgb = pixels.astype(np.float32) / 255.0
    rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    matrix = np.array(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]],
        dtype=np.float32,
    )
    xyz = rgb @ matrix.T
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    delta = 6.0 / 29.0
    transformed = np.where(
        xyz > delta**3,
        np.cbrt(xyz),
        xyz / (3.0 * delta**2) + 4.0 / 29.0,
    )
    return np.stack(
        [116.0 * transformed[:, 1] - 16.0,
         500.0 * (transformed[:, 0] - transformed[:, 1]),
         200.0 * (transformed[:, 1] - transformed[:, 2])],
        axis=1,
    )


def metrics_at_resolution(
    defect: Image.Image,
    clean: Image.Image,
    semantic: Image.Image,
    class_id: int,
    resolution: int,
    pixel_change_threshold: float,
) -> dict[str, float | int]:
    if defect.size != (resolution, resolution):
        defect = defect.resize((resolution, resolution), Image.Resampling.BILINEAR)
        clean = clean.resize((resolution, resolution), Image.Resampling.BILINEAR)
        semantic = semantic.resize((resolution, resolution), Image.Resampling.NEAREST)
    defect_array = np.array(defect, dtype=np.uint8)
    clean_array = np.array(clean, dtype=np.uint8)
    mask = np.array(semantic, dtype=np.uint8) == class_id
    area = int(np.count_nonzero(mask))
    if area == 0:
        return {
            "area": 0,
            "bbox_w": 0,
            "bbox_h": 0,
            "major": 0,
            "minor": 0,
            "diag": 0.0,
            "mean_abs_delta": 0.0,
            "delta_e76_p50": 0.0,
            "changed_fraction": 0.0,
        }
    x, y, width, height = legacy.mask_bbox(mask)
    del x, y
    differences = np.abs(defect_array.astype(np.float32) - clean_array.astype(np.float32))
    pixel_mad = differences.mean(axis=2)[mask]
    defect_lab = rgb_pixels_to_lab(defect_array[mask])
    clean_lab = rgb_pixels_to_lab(clean_array[mask])
    delta_e = np.linalg.norm(defect_lab - clean_lab, axis=1)
    return {
        "area": area,
        "bbox_w": width,
        "bbox_h": height,
        "major": max(width, height),
        "minor": min(width, height),
        "diag": round(math.hypot(width, height), 6),
        "mean_abs_delta": round(float(pixel_mad.mean()), 6),
        # Euclidean distance in CIELAB is DeltaE76; p50 is the masked median.
        "delta_e76_p50": round(float(np.median(delta_e)), 6),
        "changed_fraction": round(float(np.mean(pixel_mad >= pixel_change_threshold)), 6),
    }


def effective_gate(config: dict[str, Any], class_name: str, severity: str) -> dict[str, Any]:
    gate = dict(config["qc"]["classes"][class_name])
    severity_map = gate.pop("severity", {})
    gate.update(severity_map.get(severity, {}))
    return gate


def evaluate_visibility(
    defect: Image.Image,
    clean: Image.Image,
    semantic: Image.Image,
    class_name: str,
    severity: str,
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, float | int]], list[str]]:
    class_id = int(config["class_ids"][class_name])
    pixel_threshold = float(config["qc"]["pixel_change_threshold"])
    metrics = {
        "512": metrics_at_resolution(defect, clean, semantic, class_id, 512, pixel_threshold),
        "224": metrics_at_resolution(defect, clean, semantic, class_id, int(config["model_input_size"]), pixel_threshold),
    }
    gate = effective_gate(config, class_name, severity)
    failures: list[str] = []
    metric_names = ["area", "major", "minor", "diag", "mean_abs_delta", "delta_e76_p50", "changed_fraction"]
    for resolution, values in metrics.items():
        for metric_name in metric_names:
            gate_key = f"min_{metric_name}_{resolution}"
            if gate_key not in gate:
                continue
            actual = float(values[metric_name])
            minimum = float(gate[gate_key])
            if not math.isfinite(actual) or actual < minimum:
                failures.append(f"{gate_key}={actual:.6f}<{minimum:.6f}")
    return metrics, failures


def severity_schedule(config: dict[str, Any], class_name: str) -> list[str]:
    quotas = config["severity_quotas"]
    schedule = [severity for severity, count in quotas.items() for _ in range(int(count))]
    if len(schedule) != int(config["samples_per_primary_class"]):
        raise ValueError("severity quotas do not sum to samples_per_primary_class")
    seed_material = f"{config['release']}|{class_name}|severity".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    random.Random(seed).shuffle(schedule)
    return schedule


def create_full_contact_sheets(
    output: Path,
    rows: list[dict[str, str]],
    classes: list[str],
) -> dict[str, str]:
    sheet_dir = output / "annotations" / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    tile = 224
    label_height = 16
    columns = 10
    rows_per_sheet = 10
    font = ImageFont.load_default()
    hashes: dict[str, str] = {}
    for class_name in classes:
        class_rows = [row for row in rows if row["primary_class"] == class_name]
        if len(class_rows) != columns * rows_per_sheet:
            raise ValueError(f"full contact sheet expects 100 rows for {class_name}")
        raw_sheet = Image.new(
            "RGB", (columns * tile, rows_per_sheet * (tile + label_height)), (24, 24, 24)
        )
        overlay_sheet = raw_sheet.copy()
        raw_draw = ImageDraw.Draw(raw_sheet)
        overlay_draw = ImageDraw.Draw(overlay_sheet)
        for index, row in enumerate(class_rows):
            image = Image.open(ROOT / row["image_path"]).convert("RGB")
            mask = Image.open(ROOT / row["mask_path"]).convert("L")
            overlay = legacy.make_overlay(image, mask)
            image = image.resize((tile, tile), Image.Resampling.BILINEAR)
            overlay = overlay.resize((tile, tile), Image.Resampling.BILINEAR)
            column = index % columns
            sheet_row = index // columns
            x = column * tile
            y = sheet_row * (tile + label_height)
            raw_sheet.paste(image, (x, y + label_height))
            overlay_sheet.paste(overlay, (x, y + label_height))
            label = f"{index:04d} {row['severity'][0].upper()}"
            raw_draw.text((x + 3, y + 2), label, fill=(245, 245, 245), font=font)
            overlay_draw.text((x + 3, y + 2), label, fill=(245, 245, 245), font=font)
        raw_path = sheet_dir / f"{class_name}_raw_100_at_224.jpg"
        overlay_path = sheet_dir / f"{class_name}_overlay_100_at_224.jpg"
        raw_sheet.save(raw_path, format="JPEG", quality=92, subsampling=0)
        overlay_sheet.save(overlay_path, format="JPEG", quality=92, subsampling=0)
        hashes[raw_path.relative_to(ROOT).as_posix()] = sha256_file(raw_path)
        hashes[overlay_path.relative_to(ROOT).as_posix()] = sha256_file(overlay_path)
    return hashes


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
    (path / "annotations").mkdir(parents=True)
    marker.write_text("synthetic v2 release generated by generate_synthetic_v2_700.py\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic seven-class synthetic-v2 data with post-JPEG 512/224 QC."
    )
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
    prepare_output(output, args.force)

    base_path = ROOT / config["base"]["path"]
    base = Image.open(base_path).convert("RGB")
    size = int(config["image_size"])
    if base.size != (size, size):
        raise ValueError(f"base size {base.size} does not match config {size}")
    base_sha = sha256_file(base_path)
    rois = legacy.build_rois(config)
    class_ids = {key: int(value) for key, value in config["class_ids"].items()}
    classes = list(config["primary_classes"])
    per_class = int(config["samples_per_primary_class"])
    max_attempts = int(config["max_generation_attempts"])
    rows: list[dict[str, str]] = []
    instance_records: list[dict[str, Any]] = []

    for class_name in classes:
        (output / "images" / class_name).mkdir(parents=True)
        (output / "masks" / class_name).mkdir(parents=True)
        schedule = severity_schedule(config, class_name)
        for ordinal, severity in enumerate(schedule):
            sample_seed = legacy.stable_sample_seed(
                int(config["global_seed"]), config["release"], base_sha, class_name, ordinal
            )
            last_error: Exception | None = None
            for attempt in range(max_attempts):
                effective_seed = sample_seed + attempt * 104729
                rng = random.Random(effective_seed)
                try:
                    generated, semantic, instance_params, inside_ratio, pre_delta = apply_single_recipe(
                        base, class_name, severity, rois, class_ids, rng
                    )
                    minimum_pre = float(config["minimum_pre_transform_mean_abs_delta"][class_name])
                    if pre_delta < minimum_pre:
                        raise ValueError(f"pre-transform MAD {pre_delta:.3f} below {minimum_pre:.3f}")
                    geometry = sample_geometry(config, rng)
                    photometric = sample_photometric(config, rng)
                    generated, semantic = apply_geometry(generated, semantic, geometry, size)
                    clean, _ = apply_geometry(base, Image.new("L", base.size, 0), geometry, size)
                    generated = apply_photometric(generated, photometric)
                    clean = apply_photometric(clean, photometric)
                    generated_jpeg, image_payload = jpeg_roundtrip(generated, int(config["jpeg_quality"]))
                    clean_jpeg, _ = jpeg_roundtrip(clean, int(config["jpeg_quality"]))
                    metrics, failures = evaluate_visibility(
                        generated_jpeg, clean_jpeg, semantic, class_name, severity, config
                    )
                    if failures:
                        raise ValueError("; ".join(failures))
                    successful = True
                    break
                except Exception as error:
                    successful = False
                    last_error = error
            if not successful:
                raise RuntimeError(f"failed to generate {class_name} #{ordinal}: {last_error}")

            sample_id = f"{config['sample_id_prefix']}-{class_name}-{ordinal:04d}"
            image_path = output / "images" / class_name / f"{sample_id}.jpg"
            mask_path = output / "masks" / class_name / f"{sample_id}.png"
            image_path.write_bytes(image_payload)
            semantic.save(mask_path, format="PNG", optimize=False)
            mask_array = np.array(semantic, dtype=np.uint8)
            class_id = class_ids[class_name]
            class_mask = mask_array == class_id
            bbox = legacy.mask_bbox(class_mask)
            defect_pixels = int(np.count_nonzero(class_mask))
            image_rel = image_path.relative_to(ROOT).as_posix()
            mask_rel = mask_path.relative_to(ROOT).as_posix()
            parameter_record = {
                "attempt": attempt,
                "effective_seed": effective_seed,
                "recipe": [class_name],
                "severity": severity,
                "geometry": geometry,
                "photometric": photometric,
                "instance_pre_transform": instance_params,
            }
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
                "primary_class": class_name,
                "visible_multilabel": class_name,
                "severity": severity,
                "global_seed": str(config["global_seed"]),
                "sample_seed": str(sample_seed),
                "generator_version": config["generator_version"],
                "qc_gate_version": config["qc_gate_version"],
                "config_sha256": config_sha,
                "base_sha256": base_sha,
                "image_sha256": sha256_file(image_path),
                "mask_sha256": sha256_file(mask_path),
                "width": str(size),
                "height": str(size),
                "defect_pixels": str(defect_pixels),
                "mask_area_ratio_image": f"{defect_pixels / (size * size):.10f}",
                "roi_inside_ratio": f"{inside_ratio:.10f}",
                "pre_transform_mean_abs_delta": f"{pre_delta:.10f}",
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
                "training_use": "TRAIN_ONLY_SYNTHETIC_NG",
                "evaluation_eligible": config["evaluation_eligible"],
                "qc_status": "AUTO_PASS_POST_JPEG_512_224",
                "human_verified": "NO",
                "qc_metrics_json": json.dumps(metrics, sort_keys=True, separators=(",", ":")),
                "parameters_json": json.dumps(parameter_record, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            }
            rows.append(row)
            instance_records.append(
                {
                    "sample_id": sample_id,
                    "primary_class": class_name,
                    "visible_multilabel": [class_name],
                    "semantic_mask_path": mask_rel,
                    "instances": [{
                        "category": class_name,
                        "category_id": class_id,
                        "area_px": defect_pixels,
                        "bbox_xywh": list(bbox),
                    }],
                }
            )
        print(f"generated {class_name}: {per_class}", flush=True)

    manifest_path = output / "annotations" / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    instances_path = output / "annotations" / "instances.jsonl"
    with instances_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in instance_records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    counts = Counter(row["primary_class"] for row in rows)
    severity_counts = Counter((row["primary_class"], row["severity"]) for row in rows)
    summary_path = output / "annotations" / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["axis", "class", "value", "count"])
        for class_name in classes:
            writer.writerow(["primary_class", class_name, class_name, counts[class_name]])
            for severity in ("mild", "moderate", "severe"):
                writer.writerow(["severity", class_name, severity, severity_counts[(class_name, severity)]])

    legacy.create_contact_sheet(output, rows, classes)
    full_contact_sheet_hashes = create_full_contact_sheets(output, rows, classes)

    release = {
        "release": config["release"],
        "generator_version": config["generator_version"],
        "qc_gate_version": config["qc_gate_version"],
        "generator_script": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "generator_script_sha256": sha256_file(Path(__file__).resolve()),
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": config_sha,
        "base_path": base_path.relative_to(ROOT).as_posix(),
        "base_sha256": base_sha,
        "source_domain": config["source_domain"],
        "split": config["split"],
        "evaluation_eligible": config["evaluation_eligible"],
        "sample_count": len(rows),
        "manifest_sha256": sha256_file(manifest_path),
        "instances_sha256": sha256_file(instances_path),
        "summary_sha256": sha256_file(summary_path),
        "overview_contact_sheet_sha256": sha256_file(output / "contact_sheet.jpg"),
        "full_contact_sheet_sha256": full_contact_sheet_hashes,
        "class_counts": dict(counts),
        "severity_counts": {
            class_name: {severity: severity_counts[(class_name, severity)] for severity in ("mild", "moderate", "severe")}
            for class_name in classes
        },
        "notes": [
            "All images passed paired-clean post-JPEG visibility gates at 512 and 224 pixels.",
            "delta_e76_p50 is the median CIE76 distance over masked pixels, not CIEDE2000.",
            "This release contains seven synthetic defect classes and is train-only.",
            "The restored single base is not a confirmed real normal specimen.",
            "Real-only validation/test performance remains NOT VERIFIED.",
        ],
    }
    release_path = output / "annotations" / "release.json"
    release_path.write_text(json.dumps(release, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"PASS: generated={len(rows)}, classes={len(classes)}, per_class={per_class}, output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
