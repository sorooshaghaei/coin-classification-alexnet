import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def plot_training_history(history_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    history_df.plot(
        x="epoch",
        y=["train_loss", "val_loss"],
        ax=axes[0],
        title="Loss",
    )
    axes[0].set_ylabel("loss")

    history_df.plot(
        x="epoch",
        y=["train_accuracy", "val_accuracy", "val_top5", "val_macro_f1"],
        ax=axes[1],
        title="Validation Metrics",
    )
    axes[1].set_ylabel("score")

    fig.suptitle("AlexNet training history")
    fig.tight_layout()
    return fig, axes


def plot_class_distribution(train_df: pd.DataFrame, top_n: int = 20):
    class_counts = train_df["Class"].value_counts().head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        [label.split(",")[0] + f" ({label.split(',')[1].strip()})" for label in class_counts.index],
        class_counts.values,
        color="#355C88",
    )
    ax.set_title(f"Top {top_n} Most Frequent Classes")
    ax.set_xlabel("Number of images")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_image_size_distribution(train_df: pd.DataFrame):
    widths = []
    heights = []

    for image_path in train_df["image_path"]:
        with Image.open(image_path) as image:
            width, height = image.size
        widths.append(width)
        heights.append(height)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].hist(widths, bins=20, color="#8EC5FC", edgecolor="white")
    axes[0].set_title("Width Distribution")
    axes[0].set_xlabel("Width (pixels)")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(alpha=0.2)

    axes[1].hist(heights, bins=20, color="#A0D8B3", edgecolor="white")
    axes[1].set_title("Height Distribution")
    axes[1].set_xlabel("Height (pixels)")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    return fig, axes


def plot_sample_gallery(train_df: pd.DataFrame, sample_ids: list[int]):
    selected_rows = (
        train_df.loc[train_df["Id"].isin(sample_ids), ["Id", "Class", "image_path"]]
        .drop_duplicates(subset=["Id"])
        .set_index("Id")
        .loc[sample_ids]
        .reset_index()
    )

    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()

    for axis, (_, row) in zip(axes, selected_rows.iterrows()):
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")
            width, height = image.size
            crop_size = int(min(width, height) * 0.92)
            left = max(0, (width - crop_size) // 2)
            top = max(0, (height - crop_size) // 2)
            image = image.crop((left, top, left + crop_size, top + crop_size))
            axis.imshow(image)
        axis.set_title(row["Class"].split(",")[0], fontsize=10)
        axis.axis("off")

    for axis in axes[len(selected_rows):]:
        axis.axis("off")

    fig.suptitle("Sample Training Images", fontsize=20)
    fig.tight_layout()
    return fig, axes
