import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import AlexNet_Weights

from .config import BATCH_SIZE, NUM_WORKERS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CoinDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None, training: bool = True):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.training = training

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row = self.dataframe.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if self.training:
            return image, int(row["label_id"])
        return image, int(row["Id"])


def _resolve_normalization(weights: AlexNet_Weights | None) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if weights is None:
        return IMAGENET_MEAN, IMAGENET_STD
    preset = weights.transforms()
    return tuple(preset.mean), tuple(preset.std)


def create_transforms(
    weights: AlexNet_Weights | None = AlexNet_Weights.IMAGENET1K_V1,
    profile: str = "improved",
):
    mean, std = _resolve_normalization(weights)

    if profile == "baseline":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.02,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    elif profile == "improved":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(
                    brightness=0.18,
                    contrast=0.18,
                    saturation=0.18,
                    hue=0.02,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
                # RandomErasing is disabled for now because it can hide coin-border details
                # transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), value="random"),
            ]
        )
    else:
        raise ValueError(f"Unknown transform profile: {profile}")

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, val_transform


def build_weighted_sampler(train_split: pd.DataFrame) -> WeightedRandomSampler:
    class_counts = train_split["label_id"].value_counts().sort_index()
    sample_weights = train_split["label_id"].map(lambda label_id: 1.0 / class_counts[label_id])
    return WeightedRandomSampler(
        weights=sample_weights.to_numpy(dtype="float64", copy=True),
        num_samples=len(sample_weights),
        replacement=True,
    )


def create_dataloaders(
    train_split: pd.DataFrame,
    val_split: pd.DataFrame,
    test_df: pd.DataFrame,
    train_transform,
    val_transform,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    use_weighted_sampler: bool = False,
):
    train_dataset = CoinDataset(train_split, transform=train_transform, training=True)
    val_dataset = CoinDataset(val_split, transform=val_transform, training=True)
    test_dataset = CoinDataset(test_df, transform=val_transform, training=False)
    sampler = build_weighted_sampler(train_split) if use_weighted_sampler else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader
