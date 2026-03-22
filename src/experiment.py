import argparse
import json
import os
from pathlib import Path

import torch

from .config import (
    BATCH_SIZE,
    DEVICE,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_FINETUNE_EPOCHS,
    DEFAULT_FINETUNE_LEARNING_RATE,
    DEFAULT_HEAD_EPOCHS,
    DEFAULT_LABEL_SMOOTHING,
    DEFAULT_LEARNING_RATE,
    DEFAULT_VALIDATION_SIZE,
    DEFAULT_WEIGHT_DECAY,
    NUM_WORKERS,
    RESULTS_DIR,
    SEED,
    TrainingStage,
    set_seed,
)
from .data_loading import prepare_datasets
from .datasets import create_dataloaders, create_transforms
from .modeling import (
    build_alexnet_model,
    build_criterion,
)
from .training import (
    build_submission_dataframe,
    evaluate_model,
    fit_model,
    predict_test_labels,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .visualization import plot_training_history


def _format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _print_run_header(args, prepared_data, output_dir: Path, stages: list[TrainingStage]) -> None:
    invalid_ids = [str(item["id"]) for item in prepared_data.dataset_summary["invalid_entries"]]
    print("AlexNet Coin Classification")
    print(
        "  "
        f"mode={args.mode} | batch_size={args.batch_size} | validation_split={args.validation_size:.2f} | "
        f"seed={args.seed}"
    )
    print(
        "  "
        f"train={len(prepared_data.train_split)} | val={len(prepared_data.val_split)} | "
        f"test={len(prepared_data.test_df)} | classes={len(prepared_data.label_encoder.classes_)}"
    )
    if invalid_ids:
        print(f"  filtered invalid files: {', '.join(invalid_ids)}")
    print(f"  output_dir: {output_dir}")
    print(
        "  stages: "
        + " -> ".join(f"{stage.name} ({stage.epochs} epoch{'s' if stage.epochs > 1 else ''})" for stage in stages)
    )


def _print_run_footer(
    output_dir: Path,
    final_metrics: dict[str, object],
    best_val_accuracy: float | None,
    wrote_submission: bool,
    saved_files: list[Path],
) -> None:
    print()
    print("Run completed")
    print(f"  output_dir: {output_dir}")
    if best_val_accuracy is not None:
        print(f"  best validation accuracy: {_format_percentage(best_val_accuracy)}")
    print(f"  final validation accuracy: {_format_percentage(float(final_metrics['accuracy']))}")
    print(f"  final top-5 accuracy: {_format_percentage(float(final_metrics['top5']))}")
    print(f"  final macro-F1: {float(final_metrics['macro_f1']):.4f}")
    print(f"  final balanced accuracy: {_format_percentage(float(final_metrics['balanced_accuracy']))}")
    if wrote_submission:
        print(f"  submission: {output_dir / 'submission.csv'}")
    print("  saved files:")
    for path in saved_files:
        print(f"    {path.name}")


def build_stages(args) -> list[TrainingStage]:
    if args.mode == "baseline":
        return [
            TrainingStage(
                name="baseline_head",
                epochs=args.head_epochs,
                learning_rate=args.head_lr,
                freeze_features=True,
            )
        ]

    stages = [
        TrainingStage(
            name="head_warmup",
            epochs=args.head_epochs,
            learning_rate=args.head_lr,
            freeze_features=True,
        )
    ]
    if args.finetune_epochs > 0:
        stages.append(
            TrainingStage(
                name="full_finetune",
                epochs=args.finetune_epochs,
                learning_rate=args.finetune_lr,
                freeze_features=False,
            )
        )
    return stages


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the AlexNet coin classifier.")
    parser.add_argument("--mode", choices=["baseline", "improved"], default="improved")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--validation-size", type=float, default=DEFAULT_VALIDATION_SIZE)
    parser.add_argument("--head-epochs", type=int, default=DEFAULT_HEAD_EPOCHS)
    parser.add_argument("--finetune-epochs", type=int, default=DEFAULT_FINETUNE_EPOCHS)
    parser.add_argument("--head-lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--finetune-lr", type=float, default=DEFAULT_FINETUNE_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--label-smoothing", type=float, default=DEFAULT_LABEL_SMOOTHING)
    parser.add_argument("--patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--disable-weighted-sampler", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--skip-submission", action="store_true")
    return parser.parse_args(argv)


def run_experiment(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = DEVICE
    prepared_data = prepare_datasets(
        validation_size=args.validation_size,
        seed=args.seed,
    )
    use_weighted_sampler = args.mode == "improved" and not args.disable_weighted_sampler
    transform_profile = "baseline" if args.mode == "baseline" else "improved"
    stages = build_stages(args)
    _print_run_header(
        args=args,
        prepared_data=prepared_data,
        output_dir=output_dir,
        stages=stages,
    )

    model_bundle = build_alexnet_model(
        num_classes=len(prepared_data.label_encoder.classes_),
        device=device,
        use_pretrained=not args.no_pretrained,
    )
    train_transform, val_transform = create_transforms(
        weights=model_bundle.weights,
        profile=transform_profile,
    )
    _, _, _, train_loader, val_loader, test_loader = create_dataloaders(
        train_split=prepared_data.train_split,
        val_split=prepared_data.val_split,
        test_df=prepared_data.test_df,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=use_weighted_sampler,
    )
    criterion = build_criterion(
        device=device,
        class_weights=None,
        label_smoothing=args.label_smoothing,
    )

    history_df, best_state, best_val_accuracy = fit_model(
        model=model_bundle.model,
        stages=stages,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        criterion=criterion,
        num_classes=len(prepared_data.label_encoder.classes_),
        weight_decay=args.weight_decay,
        patience=args.patience,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )
    if best_state is not None:
        model_bundle.model.load_state_dict(best_state)

    final_metrics, report_dict = evaluate_model(
        model=model_bundle.model,
        val_loader=val_loader,
        device=device,
        criterion=criterion,
        num_classes=len(prepared_data.label_encoder.classes_),
        label_encoder=prepared_data.label_encoder,
        max_eval_batches=args.max_eval_batches,
    )

    legacy_artifacts = [
        output_dir / "history.csv",
        output_dir / "classification_report.txt",
        output_dir / "run_config.json",
        output_dir / "dataset_audit.json",
    ]
    saved_files = []
    for artifact_path in legacy_artifacts:
        artifact_path.unlink(missing_ok=True)

    classification_report_json_path = output_dir / "classification_report.json"
    _write_json(classification_report_json_path, report_dict)
    saved_files.append(classification_report_json_path)

    metrics_payload = {
        "mode": args.mode,
        "transform_profile": transform_profile,
        "use_weighted_sampler": use_weighted_sampler,
        "best_val_accuracy_during_training": best_val_accuracy,
        "final_validation": {
            "loss": float(final_metrics["loss"]),
            "accuracy": float(final_metrics["accuracy"]),
            "top5": float(final_metrics["top5"]),
            "macro_f1": float(final_metrics["macro_f1"]),
            "balanced_accuracy": float(final_metrics["balanced_accuracy"]),
        },
    }
    validation_metrics_path = output_dir / "validation_metrics.json"
    _write_json(validation_metrics_path, metrics_payload)
    saved_files.append(validation_metrics_path)

    figure, _ = plot_training_history(history_df)
    training_history_path = output_dir / "training_history.png"
    figure.savefig(training_history_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    saved_files.append(training_history_path)

    if not args.skip_submission:
        test_ids, test_predictions = predict_test_labels(model_bundle.model, test_loader, device=device)
        submission_df = build_submission_dataframe(
            test_ids=test_ids,
            test_predictions=test_predictions,
            label_encoder=prepared_data.label_encoder,
        )
        submission_path = output_dir / "submission.csv"
        submission_df.to_csv(submission_path, index=False)
        saved_files.append(submission_path)
    _print_run_footer(
        output_dir=output_dir,
        final_metrics=final_metrics,
        best_val_accuracy=best_val_accuracy,
        wrote_submission=not args.skip_submission,
        saved_files=saved_files,
    )
