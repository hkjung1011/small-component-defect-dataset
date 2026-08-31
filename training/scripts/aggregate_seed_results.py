"""Aggregate repeated fixed-split runs as mean/sample-standard-deviation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from classification_common import PipelineError, write_json


SEVERITIES = ("mild", "moderate", "severe")
EXPECTED_TEST_SUPPORT_BY_SEVERITY = {
    "mild": 29,
    "moderate": 29,
    "severe": 14,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate metrics_summary.json from repeated training seeds."
    )
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        required=True,
        help="Two or more run directories containing metrics_summary.json.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _read_summary(run_directory: Path) -> dict[str, Any]:
    path = run_directory.resolve() / "metrics_summary.json"
    if not path.is_file():
        raise PipelineError(f"metrics_summary.json not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PipelineError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise PipelineError(f"summary is not an object: {path}")
    # Keep published aggregates portable and free of workstation paths.
    value["_run_directory"] = run_directory.name
    return value


def _required(mapping: Any, key: str, context: str) -> Any:
    if not isinstance(mapping, dict) or key not in mapping:
        raise PipelineError(f"missing required aggregate identity field: {context}.{key}")
    return mapping[key]


def _experiment_identity(summary: dict[str, Any]) -> tuple[dict[str, Any], int]:
    run = summary["_run_directory"]
    metadata = _required(summary, "metadata", run)
    summary_scope = _required(summary, "evaluation_scope", run)
    metadata_scope = _required(metadata, "evaluation_scope", f"{run}.metadata")
    if summary_scope != metadata_scope:
        raise PipelineError(
            f"summary/metadata evaluation_scope mismatch in run: {run}"
        )
    model = _required(metadata, "model", f"{run}.metadata")
    training = _required(metadata, "training", f"{run}.metadata")
    weights_sha256 = _required(model, "weights_sha256", f"{run}.metadata.model")
    if (
        not isinstance(weights_sha256, str)
        or len(weights_sha256) != 64
        or any(character not in "0123456789abcdef" for character in weights_sha256)
    ):
        raise PipelineError(f"invalid or missing pretrained weights SHA-256: {run}")
    training_seed = _required(training, "seed", f"{run}.metadata.training")
    if not isinstance(training_seed, int):
        raise PipelineError(f"training seed must be an integer: {run}")

    identity = {
        "release": _required(metadata, "release", f"{run}.metadata"),
        "classes": _required(summary, "classes", run),
        "evaluation_scope": summary_scope,
        "split_fingerprint_sha256": _required(
            metadata, "split_fingerprint_sha256", f"{run}.metadata"
        ),
        "manifest_sha256": _required(
            metadata, "manifest_sha256", f"{run}.metadata"
        ),
        "model": {
            "architecture": _required(
                model, "architecture", f"{run}.metadata.model"
            ),
            "weights_sha256": weights_sha256,
            "input_size": _required(metadata, "input_size", f"{run}.metadata"),
            "fixed_component_roi_xyxy": _required(
                metadata, "fixed_component_roi_xyxy", f"{run}.metadata"
            ),
            "normalization": _required(
                metadata, "normalization", f"{run}.metadata"
            ),
        },
        "training_hyperparameters": {
            "epochs_requested": _required(
                training, "epochs_requested", f"{run}.metadata.training"
            ),
            "freeze_backbone_epochs": _required(
                training, "freeze_backbone_epochs", f"{run}.metadata.training"
            ),
            "batch_size": _required(
                training, "batch_size", f"{run}.metadata.training"
            ),
            "head_learning_rate": _required(
                training, "head_learning_rate", f"{run}.metadata.training"
            ),
            "fine_tune_learning_rate": _required(
                training,
                "fine_tune_learning_rate",
                f"{run}.metadata.training",
            ),
            "weight_decay": _required(
                training, "weight_decay", f"{run}.metadata.training"
            ),
            "augmentation": _required(
                training, "augmentation", f"{run}.metadata.training"
            ),
            "early_stopping": _required(
                training, "early_stopping", f"{run}.metadata.training"
            ),
        },
    }
    return identity, training_seed


def _first_difference(left: Any, right: Any, prefix: str = "") -> str | None:
    if type(left) is not type(right):
        return prefix or "<root>"
    if isinstance(left, dict):
        if set(left) != set(right):
            return (prefix + ".keys").lstrip(".")
        for key in sorted(left):
            difference = _first_difference(
                left[key], right[key], f"{prefix}.{key}".lstrip(".")
            )
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return (prefix + ".length").lstrip(".")
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            difference = _first_difference(
                left_value, right_value, f"{prefix}[{index}]"
            )
            if difference is not None:
                return difference
        return None
    return None if left == right else (prefix or "<root>")


def _read_predictions(
    run_directory: Path,
    summary: dict[str, Any],
    classes: list[str],
) -> dict[str, Any]:
    path = run_directory / "predictions.csv"
    if not path.is_file():
        raise PipelineError(f"predictions.csv not found: {path}")
    required_columns = {
        "sample_id",
        "true_class",
        "severity",
        "predicted_class",
        "correct",
        "split",
    }
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = sorted(required_columns - set(reader.fieldnames or []))
            if missing:
                raise PipelineError(
                    f"{path} missing required columns: {', '.join(missing)}"
                )
            raw_rows = list(reader)
    except OSError as error:
        raise PipelineError(f"cannot read {path}: {error}") from error

    expected_sample_count = int(_required(summary, "sample_count", str(path)))
    if len(raw_rows) != expected_sample_count:
        raise PipelineError(
            f"prediction count mismatch in {path}: "
            f"expected {expected_sample_count}, got {len(raw_rows)}"
        )

    targets: dict[str, tuple[str, str]] = {}
    predicted_by_id: dict[str, str] = {}
    support_by_class_severity: Counter[tuple[str, str]] = Counter()
    support_by_class: Counter[str] = Counter()
    for row_number, row in enumerate(raw_rows, start=2):
        sample_id = (row.get("sample_id") or "").strip()
        true_class = (row.get("true_class") or "").strip()
        severity = (row.get("severity") or "").strip()
        predicted_class = (row.get("predicted_class") or "").strip()
        correct = (row.get("correct") or "").strip()
        split = (row.get("split") or "").strip()
        if not sample_id:
            raise PipelineError(f"empty sample_id at {path}:{row_number}")
        if sample_id in targets:
            raise PipelineError(f"duplicate sample_id {sample_id} in {path}")
        if true_class not in classes:
            raise PipelineError(
                f"unexpected true_class {true_class!r} at {path}:{row_number}"
            )
        if severity not in SEVERITIES:
            raise PipelineError(
                f"unexpected severity {severity!r} at {path}:{row_number}"
            )
        if predicted_class not in classes:
            raise PipelineError(
                f"unexpected predicted_class {predicted_class!r} at "
                f"{path}:{row_number}"
            )
        if split != "test":
            raise PipelineError(
                f"non-test prediction at {path}:{row_number}: {split!r}"
            )
        expected_correct = "YES" if true_class == predicted_class else "NO"
        if correct != expected_correct:
            raise PipelineError(
                f"correct flag mismatch at {path}:{row_number}: "
                f"expected {expected_correct}, got {correct!r}"
            )
        targets[sample_id] = (true_class, severity)
        predicted_by_id[sample_id] = predicted_class
        support_by_class_severity[(true_class, severity)] += 1
        support_by_class[true_class] += 1

    per_class_rows = _required(summary, "per_class", str(path))
    if not isinstance(per_class_rows, list):
        raise PipelineError(f"summary per_class must be a list: {path}")
    summary_support = {
        str(_required(row, "class", f"{path}.per_class")): int(
            _required(row, "support", f"{path}.per_class")
        )
        for row in per_class_rows
    }
    if set(summary_support) != set(classes):
        raise PipelineError(f"summary per_class labels differ from classes: {path}")
    for class_name in classes:
        if support_by_class[class_name] != summary_support[class_name]:
            raise PipelineError(
                f"prediction/summary support mismatch for {class_name} in {path}"
            )
        for severity in SEVERITIES:
            expected_support = EXPECTED_TEST_SUPPORT_BY_SEVERITY[severity]
            actual_support = support_by_class_severity[(class_name, severity)]
            if actual_support != expected_support:
                raise PipelineError(
                    f"class×severity test support mismatch for "
                    f"{class_name}/{severity} in {path}: "
                    f"expected {expected_support}, got {actual_support}"
                )

    return {
        "targets": targets,
        "predicted_by_id": predicted_by_id,
        "support_by_class_severity": support_by_class_severity,
        "sample_count": len(raw_rows),
    }


def _target_fingerprint(targets: dict[str, tuple[str, str]]) -> str:
    canonical = "".join(
        f"{sample_id}\0{targets[sample_id][0]}\0{targets[sample_id][1]}\n"
        for sample_id in sorted(targets)
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _recall_by_class_severity(
    prediction_runs: list[dict[str, Any]],
    classes: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    class_severity_rows: list[dict[str, Any]] = []
    for class_name in classes:
        for severity in SEVERITIES:
            support = int(
                prediction_runs[0]["support_by_class_severity"][
                    (class_name, severity)
                ]
            )
            recall_values: list[float] = []
            for run in prediction_runs:
                correct = sum(
                    1
                    for sample_id, target in run["targets"].items()
                    if target == (class_name, severity)
                    and run["predicted_by_id"][sample_id] == class_name
                )
                recall_values.append(correct / support)
            aggregate = _mean_std(recall_values)
            class_severity_rows.append(
                {
                    "class": class_name,
                    "severity": severity,
                    "recall_mean": aggregate["mean"],
                    "recall_sample_std": aggregate["sample_std"],
                    "recall_min": aggregate["min"],
                    "recall_max": aggregate["max"],
                    "support_per_run": support,
                }
            )

    overall_severity_rows: list[dict[str, Any]] = []
    for severity in SEVERITIES:
        support = sum(
            int(
                prediction_runs[0]["support_by_class_severity"][
                    (class_name, severity)
                ]
            )
            for class_name in classes
        )
        recall_values = []
        for run in prediction_runs:
            correct = sum(
                1
                for sample_id, (true_class, sample_severity) in run[
                    "targets"
                ].items()
                if sample_severity == severity
                and run["predicted_by_id"][sample_id] == true_class
            )
            recall_values.append(correct / support)
        aggregate = _mean_std(recall_values)
        overall_severity_rows.append(
            {
                "severity": severity,
                "recall_mean": aggregate["mean"],
                "recall_sample_std": aggregate["sample_std"],
                "recall_min": aggregate["min"],
                "recall_max": aggregate["max"],
                "support_per_run": support,
            }
        )
    return class_severity_rows, overall_severity_rows


def main() -> int:
    args = parse_args()
    if len(args.runs) < 2:
        raise PipelineError("at least 2 runs are required; 3 fixed-split seeds are recommended")
    resolved_runs = [run.resolve() for run in args.runs]
    normalized_run_paths = [str(run).casefold() for run in resolved_runs]
    if len(normalized_run_paths) != len(set(normalized_run_paths)):
        raise PipelineError("duplicate run directory supplied to aggregate")
    summaries = [_read_summary(run) for run in resolved_runs]
    identities_and_seeds = [_experiment_identity(summary) for summary in summaries]
    identities = [item[0] for item in identities_and_seeds]
    training_seeds = [item[1] for item in identities_and_seeds]
    if len(training_seeds) != len(set(training_seeds)):
        raise PipelineError("duplicate training seed supplied to aggregate")
    reference = summaries[0]
    classes = reference["classes"]
    reference_scope = reference["evaluation_scope"]
    reference_fingerprint = reference["metadata"]["split_fingerprint_sha256"]
    reference_manifest = reference["metadata"]["manifest_sha256"]
    reference_identity = identities[0]
    for run_index, identity in enumerate(identities[1:], start=1):
        difference = _first_difference(reference_identity, identity)
        if difference is not None:
            raise PipelineError(
                "cannot aggregate different experiment identities: "
                f"{summaries[run_index]['_run_directory']} differs at {difference}"
            )

    # Prediction target-set checks intentionally run only after all experiment
    # identity gates have passed.
    prediction_runs = [
        _read_predictions(run, summary, classes)
        for run, summary in zip(resolved_runs, summaries)
    ]
    reference_targets = prediction_runs[0]["targets"]
    reference_support = prediction_runs[0]["support_by_class_severity"]
    target_fingerprint = _target_fingerprint(reference_targets)
    for run_index, prediction_run in enumerate(prediction_runs[1:], start=1):
        targets = prediction_run["targets"]
        if targets != reference_targets:
            missing = sorted(set(reference_targets) - set(targets))
            extra = sorted(set(targets) - set(reference_targets))
            mismatched = sorted(
                sample_id
                for sample_id in set(reference_targets) & set(targets)
                if reference_targets[sample_id] != targets[sample_id]
            )
            raise PipelineError(
                "test target mapping differs across runs: "
                f"{summaries[run_index]['_run_directory']} "
                f"missing={missing[:3]} extra={extra[:3]} "
                f"label_or_severity_mismatch={mismatched[:3]}"
            )
        if prediction_run["support_by_class_severity"] != reference_support:
            raise PipelineError(
                "class×severity support differs across runs: "
                f"{summaries[run_index]['_run_directory']}"
            )
        if _target_fingerprint(targets) != target_fingerprint:
            raise PipelineError(
                "test target fingerprint differs across runs: "
                f"{summaries[run_index]['_run_directory']}"
            )
    class_severity_recall, overall_severity_recall = _recall_by_class_severity(
        prediction_runs, classes
    )

    scalar_series = {
        "accuracy": [float(summary["accuracy"]) for summary in summaries],
        "macro_precision": [
            float(summary["macro_avg"]["precision"]) for summary in summaries
        ],
        "macro_recall": [
            float(summary["macro_avg"]["recall"]) for summary in summaries
        ],
        "macro_f1": [float(summary["macro_avg"]["f1"]) for summary in summaries],
        "weighted_precision": [
            float(summary["weighted_avg"]["precision"]) for summary in summaries
        ],
        "weighted_recall": [
            float(summary["weighted_avg"]["recall"]) for summary in summaries
        ],
        "weighted_f1": [
            float(summary["weighted_avg"]["f1"]) for summary in summaries
        ],
    }
    aggregate_scalars = {
        name: _mean_std(values) for name, values in scalar_series.items()
    }
    per_class: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(classes):
        row: dict[str, Any] = {"class": class_name, "class_index": class_index}
        for metric in ("precision", "recall", "f1"):
            values = [
                float(summary["per_class"][class_index][metric])
                for summary in summaries
            ]
            aggregate = _mean_std(values)
            for statistic_name, statistic_value in aggregate.items():
                row[f"{metric}_{statistic_name}"] = statistic_value
        supports = {
            int(summary["per_class"][class_index]["support"])
            for summary in summaries
        }
        if len(supports) != 1:
            raise PipelineError(f"support differs between runs for {class_name}")
        row["support_per_run"] = supports.pop()
        per_class.append(row)

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise PipelineError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "schema_version": "1.1",
        "run_count": len(summaries),
        "runs": [summary["_run_directory"] for summary in summaries],
        "training_seeds": training_seeds,
        "classes": classes,
        "evaluation_scope": reference_scope,
        "split_fingerprint_sha256": reference_fingerprint,
        "manifest_sha256": reference_manifest,
        "experiment_identity": reference_identity,
        "test_target_gate": {
            "status": "PASS",
            "sample_count_per_run": prediction_runs[0]["sample_count"],
            "target_fingerprint_sha256": target_fingerprint,
            "support_per_class_severity": {
                class_name: {
                    severity: int(reference_support[(class_name, severity)])
                    for severity in SEVERITIES
                }
                for class_name in classes
            },
        },
        "statistics": "arithmetic mean and sample standard deviation across training seeds",
        "summary": aggregate_scalars,
        "per_class": per_class,
        "recall_by_class_severity": class_severity_recall,
        "overall_recall_by_severity": overall_severity_recall,
        "warning": (
            "All runs use the same 504 synthetic same-base test images; variation only "
            "reflects training stochasticity, not new-specimen uncertainty."
        ),
    }
    write_json(output / "aggregate_metrics.json", aggregate)
    with (output / "aggregate_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["metric", "mean", "sample_std", "min", "max"])
        for metric_name, values in aggregate_scalars.items():
            writer.writerow(
                [
                    metric_name,
                    f"{values['mean']:.10f}",
                    f"{values['sample_std']:.10f}",
                    f"{values['min']:.10f}",
                    f"{values['max']:.10f}",
                ]
            )
    with (output / "aggregate_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fieldnames = list(per_class[0].keys())
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in per_class:
            writer.writerow(
                {
                    key: f"{value:.10f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )
    with (output / "aggregate_recall_by_severity.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fieldnames = [
            "scope",
            "class",
            "severity",
            "recall_mean",
            "recall_sample_std",
            "recall_min",
            "recall_max",
            "support_per_run",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in class_severity_recall:
            writer.writerow(
                {
                    "scope": "class_severity",
                    "class": row["class"],
                    "severity": row["severity"],
                    "recall_mean": f"{row['recall_mean']:.10f}",
                    "recall_sample_std": f"{row['recall_sample_std']:.10f}",
                    "recall_min": f"{row['recall_min']:.10f}",
                    "recall_max": f"{row['recall_max']:.10f}",
                    "support_per_run": row["support_per_run"],
                }
            )
        for row in overall_severity_recall:
            writer.writerow(
                {
                    "scope": "overall_severity",
                    "class": "__all__",
                    "severity": row["severity"],
                    "recall_mean": f"{row['recall_mean']:.10f}",
                    "recall_sample_std": f"{row['recall_sample_std']:.10f}",
                    "recall_min": f"{row['recall_min']:.10f}",
                    "recall_max": f"{row['recall_max']:.10f}",
                    "support_per_run": row["support_per_run"],
                }
            )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
