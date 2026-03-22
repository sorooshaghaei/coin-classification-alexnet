import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    top_k_accuracy_score,
)
from tqdm.auto import tqdm

from .config import TrainingStage
from .modeling import build_optimizer_and_scheduler, set_feature_extractor_trainable


def _format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_stage_name(stage_name: str) -> str:
    return stage_name.replace("_", " ").title()


def _print_stage_header(stage: TrainingStage, stage_index: int, num_stages: int) -> None:
    print()
    print(f"Stage {stage_index}/{num_stages}: {_format_stage_name(stage.name)}")
    print(
        "  "
        f"epochs={stage.epochs} | lr={stage.learning_rate:.2e} | "
        f"freeze_features={'yes' if stage.freeze_features else 'no'}"
    )


def _print_epoch_summary(
    row: dict[str, float | int | str],
    total_epochs: int,
    stage_epochs: int,
    is_best: bool,
) -> None:
    print(
        f"Epoch {row['epoch']}/{total_epochs} "
        f"({row['stage_epoch']}/{stage_epochs} in {_format_stage_name(str(row['stage']))})"
    )
    print(
        "  "
        f"train: loss {row['train_loss']:.4f} | "
        f"acc {_format_percentage(float(row['train_accuracy']))} | "
        f"macro-F1 {row['train_macro_f1']:.4f}"
    )
    print(
        "  "
        f"valid: loss {row['val_loss']:.4f} | "
        f"acc {_format_percentage(float(row['val_accuracy']))} | "
        f"top-5 {_format_percentage(float(row['val_top5']))} | "
        f"macro-F1 {row['val_macro_f1']:.4f} | "
        f"bal acc {_format_percentage(float(row['val_balanced_accuracy']))}"
    )
    status = " | new best" if is_best else ""
    print(f"  lr: {float(row['classifier_lr']):.2e}{status}")


def _build_epoch_metrics(
    all_targets: list[int],
    all_predictions: list[int],
    all_probabilities: list[np.ndarray],
    losses: list[float],
    num_classes: int,
    include_outputs: bool = False,
) -> dict[str, object]:
    proba_matrix = np.concatenate(all_probabilities, axis=0)
    labels = list(range(num_classes))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="y_pred contains classes not in y_true",
            category=UserWarning,
        )
        balanced_accuracy = float(balanced_accuracy_score(all_targets, all_predictions))

    metrics = {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(all_targets, all_predictions)),
        "top5": float(
            top_k_accuracy_score(
                all_targets,
                proba_matrix,
                k=min(5, num_classes),
                labels=labels,
            )
        ),
        "macro_f1": float(
            f1_score(
                all_targets,
                all_predictions,
                average="macro",
                labels=labels,
                zero_division=0,
            )
        ),
        "balanced_accuracy": balanced_accuracy,
    }
    if include_outputs:
        metrics["targets"] = np.array(all_targets)
        metrics["predictions"] = np.array(all_predictions)
    return metrics


def run_epoch(
    model,
    loader,
    device: torch.device,
    criterion,
    num_classes: int,
    optimizer=None,
    max_batches: int | None = None,
    gradient_clip_norm: float | None = 1.0,
    desc: str | None = None,
    include_outputs: bool = False,
):
    is_train = optimizer is not None
    model.train(is_train)

    all_targets: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[np.ndarray] = []
    losses: list[float] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_index, (images, targets) in enumerate(
            tqdm(loader, leave=False, desc=desc),
            start=1,
        ):
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()

            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            losses.append(loss.item())
            all_targets.extend(targets.detach().cpu().numpy().tolist())
            all_predictions.extend(predictions.detach().cpu().numpy().tolist())
            all_probabilities.append(probabilities.detach().cpu().numpy())

            if max_batches is not None and batch_index >= max_batches:
                break

    if not losses:
        raise ValueError("The dataloader produced no batches.")

    return _build_epoch_metrics(
        all_targets=all_targets,
        all_predictions=all_predictions,
        all_probabilities=all_probabilities,
        losses=losses,
        num_classes=num_classes,
        include_outputs=include_outputs,
    )


def fit_model(
    model,
    stages: list[TrainingStage],
    train_loader,
    val_loader,
    device: torch.device,
    criterion,
    num_classes: int,
    weight_decay: float,
    patience: int = 0,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
):
    history = []
    best_state = None
    best_val_accuracy = None
    epochs_without_improvement = 0
    global_epoch = 0
    stop_training = False
    total_epochs = sum(stage.epochs for stage in stages)

    for stage_index, stage in enumerate(stages, start=1):
        _print_stage_header(stage=stage, stage_index=stage_index, num_stages=len(stages))
        set_feature_extractor_trainable(model, trainable=not stage.freeze_features)
        optimizer, scheduler = build_optimizer_and_scheduler(
            model=model,
            stage=stage,
            weight_decay=weight_decay,
        )

        for stage_epoch in range(1, stage.epochs + 1):
            global_epoch += 1
            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                device=device,
                criterion=criterion,
                num_classes=num_classes,
                optimizer=optimizer,
                max_batches=max_train_batches,
                desc=f"train {global_epoch}/{total_epochs}",
            )
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                criterion=criterion,
                num_classes=num_classes,
                optimizer=None,
                max_batches=max_eval_batches,
                gradient_clip_norm=None,
                desc=f"valid {global_epoch}/{total_epochs}",
            )
            scheduler.step()

            row = {
                "epoch": global_epoch,
                "stage": stage.name,
                "stage_epoch": stage_epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_top5": val_metrics["top5"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "classifier_lr": float(optimizer.param_groups[-1]["lr"]),
            }
            history.append(row)

            current_val_accuracy = float(val_metrics["accuracy"])
            is_best = best_val_accuracy is None or current_val_accuracy > best_val_accuracy
            _print_epoch_summary(
                row=row,
                total_epochs=total_epochs,
                stage_epochs=stage.epochs,
                is_best=is_best,
            )

            if is_best:
                best_val_accuracy = current_val_accuracy
                epochs_without_improvement = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            else:
                epochs_without_improvement += 1

            if patience and epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epochs_without_improvement} non-improving epoch(s).")
                stop_training = True
                break

        if stop_training:
            break

    return pd.DataFrame(history), best_state, best_val_accuracy


def evaluate_model(
    model,
    val_loader,
    device: torch.device,
    criterion,
    num_classes: int,
    label_encoder,
    max_eval_batches: int | None = None,
    desc: str | None = "final valid",
):
    val_metrics = run_epoch(
        model=model,
        loader=val_loader,
        device=device,
        criterion=criterion,
        num_classes=num_classes,
        optimizer=None,
        max_batches=max_eval_batches,
        gradient_clip_norm=None,
        desc=desc,
        include_outputs=True,
    )
    labels = list(range(num_classes))
    report_dict = classification_report(
        val_metrics["targets"],
        val_metrics["predictions"],
        labels=labels,
        target_names=label_encoder.classes_,
        zero_division=0,
        output_dict=True,
    )
    return val_metrics, report_dict


def predict_test_labels(model, test_loader, device: torch.device):
    test_predictions = []
    test_ids = []
    model.eval()

    with torch.no_grad():
        for images, ids in tqdm(test_loader, leave=False, desc="test predict"):
            images = images.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1).detach().cpu().numpy()
            test_predictions.extend(predictions.tolist())
            test_ids.extend(ids.numpy().tolist())

    return test_ids, test_predictions


def build_submission_dataframe(test_ids, test_predictions, label_encoder) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Id": test_ids,
            "Class": label_encoder.inverse_transform(test_predictions),
        }
    ).sort_values("Id")
