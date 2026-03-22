from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import (
    DEFAULT_VALIDATION_SIZE,
    SEED,
    TEST_CSV,
    TEST_DIR,
    TRAIN_CSV,
    TRAIN_DIR,
)


@dataclass
class PreparedData:
    train_split: pd.DataFrame
    val_split: pd.DataFrame
    test_df: pd.DataFrame
    label_encoder: LabelEncoder
    dataset_summary: dict[str, object]


def build_image_index(folder: Path) -> dict[str, Path]:
    image_index: dict[str, Path] = {}
    duplicate_stems: list[str] = []

    for path in folder.iterdir():
        if not path.is_file():
            continue
        stem = path.stem
        if stem in image_index:
            duplicate_stems.append(stem)
        image_index[stem] = path

    if duplicate_stems:
        duplicates_preview = ", ".join(sorted(set(duplicate_stems))[:10])
        raise ValueError(f"Duplicate image identifiers detected in {folder}: {duplicates_preview}")

    return image_index


def load_annotation_frames(
    train_csv: Path = TRAIN_CSV,
    test_csv: Path = TEST_CSV,
    train_dir: Path = TRAIN_DIR,
    test_dir: Path = TEST_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_index = build_image_index(train_dir)
    test_index = build_image_index(test_dir)

    train_df["image_path"] = train_df["Id"].astype(str).map(train_index)
    test_df["image_path"] = test_df["Id"].astype(str).map(test_index)
    return train_df, test_df


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def filter_invalid_entries(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clean_train_df = train_df.dropna(subset=["image_path"]).copy()
    clean_test_df = test_df.dropna(subset=["image_path"]).copy()

    clean_train_df["is_valid"] = clean_train_df["image_path"].map(is_valid_image)
    clean_test_df["is_valid"] = clean_test_df["image_path"].map(is_valid_image)

    invalid_train_df = clean_train_df.loc[~clean_train_df["is_valid"], ["Id", "image_path"]].copy()
    invalid_train_df["split"] = "train"
    invalid_test_df = clean_test_df.loc[~clean_test_df["is_valid"], ["Id", "image_path"]].copy()
    invalid_test_df["split"] = "test"
    invalid_df = pd.concat([invalid_train_df, invalid_test_df], ignore_index=True)

    clean_train_df = (
        clean_train_df.loc[clean_train_df["is_valid"]]
        .drop(columns=["is_valid"])
        .reset_index(drop=True)
    )
    clean_test_df = (
        clean_test_df.loc[clean_test_df["is_valid"]]
        .drop(columns=["is_valid"])
        .reset_index(drop=True)
    )
    return clean_train_df, clean_test_df, invalid_df


def encode_labels(train_df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    label_encoder = LabelEncoder()
    encoded_df = train_df.copy()
    encoded_df["label_id"] = label_encoder.fit_transform(encoded_df["Class"])
    return encoded_df, label_encoder


def split_training_data(
    train_df: pd.DataFrame,
    test_size: float = DEFAULT_VALIDATION_SIZE,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_split, val_split = train_test_split(
        train_df,
        test_size=test_size,
        random_state=seed,
        stratify=train_df["label_id"],
    )
    return train_split, val_split


def build_dataset_summary(
    original_train_df: pd.DataFrame,
    clean_train_df: pd.DataFrame,
    clean_test_df: pd.DataFrame,
    invalid_df: pd.DataFrame,
) -> dict[str, object]:
    class_counts = clean_train_df["Class"].value_counts()
    summary = {
        "train_rows_raw": int(len(original_train_df)),
        "train_rows_clean": int(len(clean_train_df)),
        "test_rows_clean": int(len(clean_test_df)),
        "num_classes": int(class_counts.shape[0]),
        "min_class_count": int(class_counts.min()),
        "max_class_count": int(class_counts.max()),
        "median_class_count": float(class_counts.median()),
        "mean_class_count": float(class_counts.mean()),
        "invalid_entries": [
            {
                "split": row["split"],
                "id": int(row["Id"]),
                "image_path": str(row["image_path"]),
            }
            for _, row in invalid_df.iterrows()
        ],
    }
    return summary


def prepare_datasets(
    validation_size: float = DEFAULT_VALIDATION_SIZE,
    seed: int = SEED,
) -> PreparedData:
    train_df, test_df = load_annotation_frames()
    clean_train_df, clean_test_df, invalid_df = filter_invalid_entries(train_df, test_df)
    encoded_train_df, label_encoder = encode_labels(clean_train_df)
    train_split, val_split = split_training_data(
        encoded_train_df,
        test_size=validation_size,
        seed=seed,
    )
    dataset_summary = build_dataset_summary(train_df, encoded_train_df, clean_test_df, invalid_df)

    return PreparedData(
        train_split=train_split,
        val_split=val_split,
        test_df=clean_test_df,
        label_encoder=label_encoder,
        dataset_summary=dataset_summary,
    )
