#!/usr/bin/env python3
"""Train-only Faster R-CNN transfer-learning pipeline for synthetic v4/v5.

This script deliberately has no validation or test split.  Every reported loss
and sample prediction is a training diagnostic, never an independent performance
estimate.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from detection_common import (
    STATUS_CATEGORIES,
    ComponentDetectionDataset,
    FamilyVariantSampler,
    PipelineError,
    build_plain_transform,
    build_train_transforms,
    code_fingerprints,
    detection_collate_fn,
    runtime_environment,
    sha256_file,
    validate_pipeline,
    write_json,
)


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "synthetic_v4_v5_component_detector.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer-learn a FasterRCNN-MobileNetV3-FPN component detector from "
            "the family-balanced synthetic v4/v5 TRAIN_ONLY releases."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--task",
        choices=("component_localization", "component_status"),
        default=None,
        help="Default: component_localization. component_status uses the separate eight-class namespace.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run all config/data/path/hash/family/weight gates without model construction or optimizer steps.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run at most eight optimizer steps and write a diagnostic-only checkpoint/artifacts.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a concrete torch device such as cuda:0.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Artifact directory. Existing detector artifacts are never overwritten.",
    )
    return parser.parse_args(argv)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _select_device(requested: str) -> Any:
    try:
        import torch
    except ImportError as error:
        raise PipelineError(
            "training requires the Codex bundled Python runtime with torch and torchvision"
        ) from error
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as error:
        raise PipelineError(f"invalid torch device: {requested}") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise PipelineError("CUDA was requested but is not available in this runtime")
    return device


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _build_model(preflight: Any, task: str) -> tuple[Any, int]:
    try:
        import torch
        from torchvision.models.detection import FasterRCNN
        from torchvision.models.detection.backbone_utils import mobilenet_backbone
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.rpn import AnchorGenerator
        from torchvision.ops.misc import FrozenBatchNorm2d
    except ImportError as error:
        raise PipelineError(
            "training requires the Codex bundled Python runtime with torch and torchvision"
        ) from error

    model_config = preflight.config["model"]
    # Reproduce torchvision's pretrained construction path with FrozenBatchNorm2d,
    # but pass weights=None to the backbone.  The already hash-verified detector
    # cache is loaded directly below, so no torchvision URL/download path runs.
    trainable_layers = int(model_config["trainable_backbone_layers"])
    backbone = mobilenet_backbone(
        backbone_name="mobilenet_v3_large",
        weights=None,
        fpn=True,
        norm_layer=FrozenBatchNorm2d,
        trainable_layers=trainable_layers,
    )
    anchor_sizes = ((32, 64, 128, 256, 512),) * 3
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    model = FasterRCNN(
        backbone,
        num_classes=91,
        rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios),
        min_size=int(model_config["min_size"]),
        max_size=int(model_config["max_size"]),
        rpn_score_thresh=0.05,
    )
    try:
        official_state = torch.load(preflight.weight_path, map_location="cpu", weights_only=True)
        model.load_state_dict(official_state, strict=True)
    except Exception as error:
        raise PipelineError(
            "the verified official Faster R-CNN cache file is incompatible with this torchvision runtime"
        ) from error

    foreground_count = 1 if task == "component_localization" else len(STATUS_CATEGORIES)
    num_classes = foreground_count + 1  # background is always class 0
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model, num_classes


def _prepare_output(output: Path, repository_root: Path, smoke: bool) -> Path:
    output = output.expanduser()
    if not output.is_absolute():
        output = repository_root / output
    output = output.resolve()
    artifact_names = {
        "checkpoint_smoke.pt" if smoke else "model_final.pt",
        "run_metadata.json",
        "training_history.json",
        "sample_train_diagnostic.jpg",
        "sample_train_diagnostic.json",
    }
    collisions = sorted(name for name in artifact_names if (output / name).exists())
    if collisions:
        raise PipelineError(
            "refusing to overwrite existing detector artifacts; choose another --output: "
            + ", ".join(collisions)
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def _assert_model_state_finite(model: Any, context: str) -> None:
    """Fail before a checkpoint can contain a non-finite parameter or buffer."""

    import torch

    for kind, values in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, tensor in values:
            detached = tensor.detach()
            if (torch.is_floating_point(detached) or torch.is_complex(detached)) and not bool(
                torch.isfinite(detached).all().item()
            ):
                raise PipelineError(f"non-finite model {kind} {name!r} after {context}")


def _robust_gradient_norm(model: Any) -> float:
    """Compute a float64 L2 norm so finite fp32 gradients cannot overflow the norm."""

    import torch

    parameter_norms = [
        torch.linalg.vector_norm(parameter.grad.detach(), ord=2, dtype=torch.float64)
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not parameter_norms:
        raise PipelineError("optimizer update has no gradients")
    return float(torch.linalg.vector_norm(torch.stack(parameter_norms), ord=2).cpu())


def _clip_gradients(model: Any, gradient_norm: float, max_norm: float) -> None:
    if not (math.isfinite(gradient_norm) and math.isfinite(max_norm) and max_norm > 0.0):
        raise PipelineError("gradient clipping requires finite positive norms")
    coefficient = min(1.0, max_norm / (gradient_norm + 1e-12))
    if coefficient < 1.0:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(coefficient)


def _train(
    *,
    preflight: Any,
    task: str,
    device: Any,
    smoke: bool,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    training_config = preflight.config["training"]
    seed = int(training_config["seed"])
    augmentation_seed = int(training_config["augmentation_seed"])
    _seed_everything(seed)
    model, num_classes = _build_model(preflight, task)
    model.to(device)
    _assert_model_state_finite(model, "verified pretrained load and head replacement")
    model.train()

    dataset = ComponentDetectionDataset(
        preflight.scenes,
        task=task,
        transforms=build_train_transforms(preflight.config),
    )
    sampler = FamilyVariantSampler(preflight.family_to_indices, seed=augmentation_seed)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        sampler=sampler,
        num_workers=int(training_config["num_workers"]),
        collate_fn=detection_collate_fn,
        pin_memory=device.type == "cuda",
        generator=loader_generator,
    )
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training_config["learning_rate"]),
        momentum=float(training_config["momentum"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    amp_effective = bool(training_config["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_effective)
    max_steps = int(training_config["smoke_max_steps"]) if smoke else None
    requested_epochs = int(training_config["epochs"])

    step_history: list[dict[str, Any]] = []
    epoch_history: list[dict[str, Any]] = []
    total_steps = 0
    total_forward_backward_attempts = 0
    total_nonfinite_gradient_skips = 0
    model_state_finite_check_count = 1
    retry_limit = int(training_config["nonfinite_gradient_max_retries_per_batch"])
    start_time = time.perf_counter()
    for epoch in range(requested_epochs):
        sampler.set_epoch(epoch)
        epoch_selection = list(sampler)
        # Torchvision v2 random transforms draw from torch's RNG.  Reseeding per
        # epoch makes the family selection and weak augmentation replayable.
        _seed_everything(augmentation_seed + epoch)
        epoch_losses: dict[str, list[float]] = defaultdict(list)
        epoch_steps = 0
        epoch_samples = 0
        epoch_forward_backward_attempts = 0
        epoch_nonfinite_gradient_skips = 0
        model.train()
        for images, targets in loader:
            epoch_samples += len(images)
            images = [image.to(device, non_blocking=device.type == "cuda") for image in images]
            targets = [
                {name: value.to(device) if hasattr(value, "to") else value for name, value in target.items()}
                for target in targets
            ]
            batch_retry_count = 0
            while True:
                total_forward_backward_attempts += 1
                epoch_forward_backward_attempts += 1
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=amp_effective):
                    loss_dict = model(images, targets)
                    total_loss = sum(loss for loss in loss_dict.values())
                nonfinite_losses = [
                    name
                    for name, value in loss_dict.items()
                    if not bool(torch.isfinite(value).all().item())
                ]
                if nonfinite_losses or not bool(torch.isfinite(total_loss).all().item()):
                    raise PipelineError(
                        "non-finite forward loss cannot be recovered by gradient scaling at "
                        f"optimizer update {total_steps + 1}: {nonfinite_losses or ['loss_total']}"
                    )
                loss_values = {
                    name: float(value.detach().cpu()) for name, value in loss_dict.items()
                }
                loss_values["loss_total"] = float(total_loss.detach().cpu())
                if any(not math.isfinite(value) for value in loss_values.values()):
                    raise PipelineError("non-finite loss scalar refused before history serialization")

                amp_scale_before = float(scaler.get_scale())
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm_value = _robust_gradient_norm(model)
                if not math.isfinite(gradient_norm_value):
                    if not amp_effective:
                        raise PipelineError(
                            f"non-finite gradient norm without AMP at optimizer update {total_steps + 1}"
                        )
                    # Never call optimizer.step for this attempt.  Explicitly
                    # lower the scaler, clear its per-optimizer inf state, and
                    # retry this exact batch.
                    scaler.update(new_scale=amp_scale_before * 0.5)
                    amp_scale_after_skip = float(scaler.get_scale())
                    optimizer.zero_grad(set_to_none=True)
                    total_nonfinite_gradient_skips += 1
                    epoch_nonfinite_gradient_skips += 1
                    batch_retry_count += 1
                    if not (
                        math.isfinite(amp_scale_before)
                        and math.isfinite(amp_scale_after_skip)
                        and amp_scale_after_skip < amp_scale_before
                    ):
                        raise PipelineError(
                            "GradScaler did not lower its scale after a non-finite gradient"
                        )
                    if batch_retry_count > retry_limit:
                        raise PipelineError(
                            "non-finite gradients exceeded the configured retries for one batch"
                        )
                    continue

                _clip_gradients(
                    model,
                    gradient_norm_value,
                    float(training_config["gradient_clip_norm"]),
                )
                scaler.step(optimizer)
                scaler.update()
                amp_scale_after = float(scaler.get_scale())
                if not math.isfinite(amp_scale_after):
                    raise PipelineError("GradScaler produced a non-finite scale")
                _assert_model_state_finite(model, f"optimizer update {total_steps + 1}")
                model_state_finite_check_count += 1

                total_steps += 1
                epoch_steps += 1
                for name, value in loss_values.items():
                    epoch_losses[name].append(value)
                if smoke:
                    step_history.append(
                        {
                            "step": total_steps,
                            "epoch": epoch + 1,
                            "metric_scope": "TRAIN_DIAGNOSTIC_ONLY",
                            "losses": loss_values,
                            "gradient_norm_before_clip": gradient_norm_value,
                            "amp_scale_before": amp_scale_before,
                            "amp_scale_after": amp_scale_after,
                            "nonfinite_gradient_retries_before_update": batch_retry_count,
                            "model_state_finite_after_update": True,
                            "optimizer_update_completed": True,
                        }
                    )
                break
            if max_steps is not None and total_steps >= max_steps:
                break

        epoch_history.append(
            {
                "epoch": epoch + 1,
                "optimizer_steps": epoch_steps,
                "forward_backward_attempts": epoch_forward_backward_attempts,
                "nonfinite_gradient_skips": epoch_nonfinite_gradient_skips,
                "consumed_family_count": epoch_samples,
                "consumed_variant_counts": dict(
                    sorted(
                        Counter(
                            preflight.scenes[index].variant_name
                            for index in epoch_selection[:epoch_samples]
                        ).items()
                    )
                ),
                "metric_scope": "TRAIN_DIAGNOSTIC_ONLY",
                "mean_train_losses": {
                    name: sum(values) / len(values) for name, values in sorted(epoch_losses.items())
                },
            }
        )
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "steps": epoch_steps,
                    "train_loss": epoch_history[-1]["mean_train_losses"].get("loss_total"),
                    "scope": "TRAIN_DIAGNOSTIC_ONLY",
                },
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )
        if max_steps is not None and total_steps >= max_steps:
            break

    history = {
        "schema_version": "component-detector-training-history-v1",
        "mode": "SMOKE_MAX_8_STEPS" if smoke else "FIXED_EPOCHS_NO_VALIDATION",
        "task": task,
        "metric_scope": "TRAIN_DIAGNOSTIC_ONLY",
        "performance_claim_permitted": False,
        "validation_run": False,
        "test_run": False,
        "requested_epochs": requested_epochs,
        "completed_epochs": len(epoch_history),
        "optimizer_steps": total_steps,
        "forward_backward_attempts": total_forward_backward_attempts,
        "nonfinite_gradient_skips": total_nonfinite_gradient_skips,
        "all_optimizer_updates_had_finite_gradient_norm": True,
        "model_state_finite_check_count": model_state_finite_check_count,
        "model_state_finite_after_training": True,
        "samples_per_complete_epoch": len(sampler),
        "num_model_classes_including_background": num_classes,
        "amp_requested": bool(training_config["amp"]),
        "amp_effective": amp_effective,
        "elapsed_seconds": time.perf_counter() - start_time,
        "epochs": epoch_history,
        "smoke_steps": step_history,
    }
    return model, history


def _label_name(task: str, label: int) -> str:
    if label == 0:
        return "background"
    if task == "component_localization":
        return "component" if label == 1 else f"unknown_{label}"
    return dict(STATUS_CATEGORIES).get(label, f"unknown_{label}")


def _write_sample_diagnostic(
    *,
    model: Any,
    preflight: Any,
    task: str,
    device: Any,
    output: Path,
) -> dict[str, Any]:
    import torch
    from PIL import Image, ImageDraw

    first_family = sorted(preflight.family_to_indices)[0]
    sample_index = preflight.family_to_indices[first_family][0]
    dataset = ComponentDetectionDataset(
        preflight.scenes,
        task=task,
        transforms=build_plain_transform(),
    )
    image, target = dataset[sample_index]
    model.eval()
    with torch.inference_mode():
        prediction = model([image.to(device)])[0]
    score_threshold = float(preflight.config["diagnostics"]["sample_score_threshold"])
    max_predictions = int(preflight.config["diagnostics"]["sample_max_predictions"])
    scores = prediction["scores"].detach().cpu()
    boxes = prediction["boxes"].detach().cpu()
    labels = prediction["labels"].detach().cpu()
    if not bool(torch.isfinite(scores).all().item()) or not bool(
        torch.isfinite(boxes).all().item()
    ):
        raise PipelineError("sample diagnostic prediction contains non-finite score or bbox")
    selected = [
        index
        for index, score in enumerate(scores.tolist())
        if score >= score_threshold
    ][:max_predictions]

    scene = preflight.scenes[sample_index]
    with Image.open(scene.image_absolute_path) as opened:
        rendered = opened.convert("RGB")
    draw = ImageDraw.Draw(rendered)
    for annotation in scene.annotations:
        draw.rectangle(annotation.bbox_xyxy, outline=(30, 220, 80), width=3)
    predictions_json: list[dict[str, Any]] = []
    for prediction_index in selected:
        box = tuple(float(value) for value in boxes[prediction_index].tolist())
        score = float(scores[prediction_index])
        label = int(labels[prediction_index])
        draw.rectangle(box, outline=(255, 210, 30), width=3)
        draw.text(
            (box[0] + 2, max(0.0, box[1] - 12)),
            f"{_label_name(task, label)} {score:.2f}",
            fill=(255, 235, 70),
        )
        predictions_json.append(
            {
                "bbox_xyxy": [round(value, 4) for value in box],
                "label_id": label,
                "label": _label_name(task, label),
                "score": score,
            }
        )
    image_path = output / "sample_train_diagnostic.jpg"
    rendered.save(image_path, format="JPEG", quality=94, subsampling=0)
    diagnostic = {
        "schema_version": "component-detector-sample-diagnostic-v1",
        "scope": "TRAIN_DIAGNOSTIC_ONLY",
        "performance_claim_permitted": False,
        "note": "Green boxes are training targets; yellow boxes are same-training-sample predictions.",
        "task": task,
        "release_key": scene.release_key,
        "release_image_id": scene.release_image_id,
        "scene_id": scene.scene_id,
        "composition_family_id": scene.family_id,
        "score_threshold": score_threshold,
        "prediction_count_shown": len(predictions_json),
        "predictions": predictions_json,
        "image_sha256": sha256_file(image_path),
    }
    write_json(output / "sample_train_diagnostic.json", diagnostic)
    return diagnostic


def _write_artifacts(
    *,
    model: Any,
    history: dict[str, Any],
    preflight: Any,
    task: str,
    device: Any,
    output: Path,
    smoke: bool,
) -> dict[str, Any]:
    import torch

    _assert_model_state_finite(model, "artifact serialization")
    history_path = output / "training_history.json"
    write_json(history_path, history)
    diagnostic = _write_sample_diagnostic(
        model=model,
        preflight=preflight,
        task=task,
        device=device,
        output=output,
    )
    code_hashes = code_fingerprints(preflight.repository_root)
    checkpoint_path = output / ("checkpoint_smoke.pt" if smoke else "model_final.pt")
    cpu_state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save(
        {
            "schema_version": "component-detector-checkpoint-v1",
            "mode": "SMOKE_MAX_8_STEPS" if smoke else "FIXED_EPOCHS_NO_VALIDATION",
            "task": task,
            "class_names": (
                ["background", "component"]
                if task == "component_localization"
                else ["background"] + [name for _, name in STATUS_CATEGORIES]
            ),
            "model_architecture": preflight.config["model"]["architecture"],
            "model_state_dict": cpu_state,
            "optimizer_state_included": False,
            "resume_supported": False,
            "optimizer_steps": history["optimizer_steps"],
            "fingerprints": {
                **preflight.fingerprints,
                "code": code_hashes,
            },
            "metric_scope": "TRAIN_DIAGNOSTIC_ONLY",
            "performance_claim_permitted": False,
            "model_state_finite": True,
        },
        checkpoint_path,
    )
    checkpoint_sha = sha256_file(checkpoint_path)
    metadata = {
        "schema_version": "component-detector-run-metadata-v1",
        "created_at_utc": _timestamp(),
        "mode": "SMOKE_MAX_8_STEPS" if smoke else "FIXED_EPOCHS_NO_VALIDATION",
        "task": task,
        "dataset_scope": "SYNTHETIC_V4_V5_TRAIN_ONLY",
        "metric_scope": "TRAIN_DIAGNOSTIC_ONLY",
        "performance_claim_permitted": False,
        "validation_created_or_run": False,
        "test_created_or_run": False,
        "seed": int(preflight.config["training"]["seed"]),
        "augmentation_seed": int(preflight.config["training"]["augmentation_seed"]),
        "fingerprints": {
            **preflight.fingerprints,
            "code": code_hashes,
            "checkpoint_sha256": checkpoint_sha,
            "training_history_sha256": sha256_file(history_path),
            "sample_diagnostic_image_sha256": diagnostic["image_sha256"],
            "sample_diagnostic_json_sha256": sha256_file(
                output / "sample_train_diagnostic.json"
            ),
        },
        "preflight_summary": preflight.summary,
        "training_summary": {
            key: history[key]
            for key in (
                "requested_epochs",
                "completed_epochs",
                "optimizer_steps",
                "forward_backward_attempts",
                "nonfinite_gradient_skips",
                "all_optimizer_updates_had_finite_gradient_norm",
                "model_state_finite_after_training",
                "samples_per_complete_epoch",
                "amp_requested",
                "amp_effective",
            )
        },
        "environment": runtime_environment(device=str(device)),
        "artifacts": {
            "checkpoint": checkpoint_path.name,
            "training_history": history_path.name,
            "sample_diagnostic_image": "sample_train_diagnostic.jpg",
            "sample_diagnostic_json": "sample_train_diagnostic.json",
        },
        "limitations": [
            "All inputs are TRAIN_ONLY synthetic scenes from one connected source-parent graph.",
            "No validation or test split was created or evaluated.",
            "Training losses and the same-training-sample overlay are diagnostics, not performance metrics.",
            "normal_proxy is not confirmed real OK data.",
        ],
    }
    metadata_path = output / "run_metadata.json"
    write_json(metadata_path, metadata)
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        preflight = validate_pipeline(args.config)
        task = args.task or str(preflight.config["default_task"])
        check_report = {
            **preflight.summary,
            "task": task,
            "fingerprints": {
                **preflight.fingerprints,
                "code": code_fingerprints(preflight.repository_root),
            },
            "environment": runtime_environment(),
            "network_download_permitted": False,
        }
        if args.check_only:
            print(
                json.dumps(
                    check_report,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            return 0

        device = _select_device(args.device)
        default_output_name = (
            "training/results/component-detector-smoke"
            if args.smoke
            else "training/results/component-detector-fixed-epochs"
        )
        output = _prepare_output(
            args.output or Path(default_output_name),
            preflight.repository_root,
            smoke=args.smoke,
        )
        model, history = _train(
            preflight=preflight,
            task=task,
            device=device,
            smoke=args.smoke,
        )
        metadata = _write_artifacts(
            model=model,
            history=history,
            preflight=preflight,
            task=task,
            device=device,
            output=output,
            smoke=args.smoke,
        )
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "output": str(output),
                    "mode": metadata["mode"],
                    "task": task,
                    "optimizer_steps": history["optimizer_steps"],
                    "metric_scope": "TRAIN_DIAGNOSTIC_ONLY",
                    "performance_claim_permitted": False,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    except PipelineError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("ERROR: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
