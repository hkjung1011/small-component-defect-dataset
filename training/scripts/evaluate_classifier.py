"""Re-evaluate a saved classifier checkpoint on the immutable 504-image test set."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from classification_common import (
    PipelineError,
    build_model,
    choose_device,
    create_dataset_class,
    deterministic_split,
    evaluate_model,
    load_and_validate_manifest,
    load_config,
    load_ml_dependencies,
    split_samples,
    write_evaluation_artifacts,
    write_json,
    write_split_artifacts,
)


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_CONFIG = SCRIPT_PATH.parents[1] / "configs" / "synthetic_v2_700_classifier.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved synthetic-v2-700 classifier checkpoint."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def _load_checkpoint(path: Path, torch_module: Any) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise PipelineError(f"checkpoint not found: {path}")
    try:
        checkpoint = torch_module.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch_module.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise PipelineError(f"unsupported checkpoint structure: {path}")
    return checkpoint


def main() -> int:
    args = parse_args()
    config, _config_path, repository_root = load_config(args.config)
    samples, manifest_audit = load_and_validate_manifest(config, repository_root)
    records, split_audit = deterministic_split(samples, config)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    architecture = str(config["model"]["architecture"])
    torch_module, image_module, image_ops_module = load_ml_dependencies(
        require_torchvision=architecture == "resnet18"
    )
    device = choose_device(args.device, torch_module)
    checkpoint = _load_checkpoint(args.checkpoint, torch_module)

    classes: list[str] = list(config["classes"])
    if checkpoint.get("classes") != classes:
        raise PipelineError("checkpoint class order differs from config class order")
    if checkpoint.get("release") != config["release"]:
        raise PipelineError("checkpoint release differs from config release")
    if checkpoint.get("manifest_sha256") != manifest_audit["manifest_sha256"]:
        raise PipelineError("checkpoint manifest SHA differs from current manifest")
    if (
        checkpoint.get("split_fingerprint_sha256")
        != split_audit["fingerprint_sha256"]
    ):
        raise PipelineError("checkpoint split fingerprint differs from current split")
    if checkpoint.get("fixed_component_roi_xyxy") != config["model"][
        "fixed_component_roi_xyxy"
    ]:
        raise PipelineError("checkpoint fixed ROI differs from config")
    if checkpoint.get("normalization") != config["model"]["normalization"]:
        raise PipelineError("checkpoint normalization differs from config")

    model, _fresh_model_audit = build_model(
        config,
        len(classes),
        torch_module,
        weights_mode_override="none",
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)

    class_to_index = {class_name: index for index, class_name in enumerate(classes)}
    model_config = config["model"]
    normalization = model_config["normalization"]
    dataset_class = create_dataset_class(
        torch_module, image_module, image_ops_module
    )
    test_samples = split_samples(records, "test")
    test_dataset = dataset_class(
        test_samples,
        class_to_index,
        int(model_config["input_size"]),
        normalization["mean"],
        normalization["std"],
        model_config["fixed_component_roi_xyxy"],
        training=False,
    )
    test_loader = torch_module.utils.data.DataLoader(
        test_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    predictions, matrix, metrics = evaluate_model(
        model,
        test_loader,
        {sample.sample_id: sample for sample in test_samples},
        classes,
        device,
        torch_module,
    )

    output_directory = args.output.resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise PipelineError(f"output directory is not empty: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=True)
    write_split_artifacts(output_directory, records, manifest_audit, split_audit)
    metadata = {
        "release": config["release"],
        "evaluation_scope": config["evaluation"]["scope"],
        "test_set_used_for_model_selection": False,
        "sample_counts": split_audit["model_counts"],
        "sample_severity_counts": split_audit["model_severity_counts"],
        "samples_per_class_severity": split_audit[
            "model_class_severity_counts"
        ],
        # The checkpoint hash/contents provide provenance; an absolute local
        # path would only disclose workstation-specific information.
        "checkpoint": args.checkpoint.name,
        "checkpoint_selected_epoch": checkpoint.get("selected_epoch"),
        "split_fingerprint_sha256": split_audit["fingerprint_sha256"],
        "manifest_sha256": manifest_audit["manifest_sha256"],
        "base_group_overlap": split_audit["base_group_overlap"],
        "fixed_component_roi_xyxy": model_config[
            "fixed_component_roi_xyxy"
        ],
        "roi_source": "single class-independent config constant",
        "roi_uses_label_mask_or_bbox": False,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch_module.__version__,
            "device": str(device),
        },
        "warnings": sorted(
            set(manifest_audit["warnings"] + split_audit["warnings"])
        ),
    }
    write_evaluation_artifacts(
        output_directory,
        classes,
        predictions,
        matrix,
        metrics,
        metadata,
    )
    write_json(output_directory / "run_metadata.json", metadata)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output_directory),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_avg"]["f1"],
                "weighted_f1": metrics["weighted_avg"]["f1"],
                "evaluation_scope": config["evaluation"]["scope"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
