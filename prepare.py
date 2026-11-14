"""Split the bike table, prepare the features, and wrap them in a Dataset"""

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from bikes_data import TARGET, make_rentals

NUMERIC = ["hour", "temperature_c", "rain_mm", "wind_kph", "is_weekend"]
CATEGORICAL = ["station_type"]


def make_preprocessor() -> ColumnTransformer:
    """Scale the numbers and one-hot encode the station type."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ]
    )


class BikeDataset(Dataset):
    """Holds features and targets as tensors and hands out one row at a time"""

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


@dataclass
class Splits:
    """The three datasets plus the fitted preprocessor, kept together."""

    train: BikeDataset
    val: BikeDataset
    test: BikeDataset
    preprocessor: ColumnTransformer

    @property
    def n_features(self) -> int:
        """The number of features after preprocessing."""
        return self.train.x.shape[1]


def make_splits(seed: int = 42) -> Splits:
    """Split 60/20/20 first, then fit the preprocessor on training rows only."""
    frame = make_rentals()
    train_df, rest = train_test_split(frame, test_size=0.4, random_state=seed)
    val_df, test_df = train_test_split(rest, test_size=0.5, random_state=seed)

    prep = make_preprocessor()
    prep.fit(train_df[NUMERIC + CATEGORICAL])

    def build(df) -> BikeDataset:
        return BikeDataset(
            prep.transform(df[NUMERIC + CATEGORICAL]), df[TARGET].to_numpy()
        )

    return Splits(build(train_df), build(val_df), build(test_df), prep)


def make_loaders(splits: Splits, batch_size: int = 64, seed: int = 42):
    """Shuffle the training batches. Validation and test keep their order."""
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(
        splits.train, batch_size=batch_size, shuffle=True, generator=generator
    )
    val = DataLoader(splits.val, batch_size=256)
    test = DataLoader(splits.test, batch_size=256)
    return train, val, test
