import torch
from sklearn.model_selection import train_test_split

from bikes_data import make_rentals
from prepare import make_loaders


def test_dataset_has_right_length_and_float32_rows(splits):
    assert len(splits.train) == 3600
    x, y = splits.train[0]
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert x.shape == (splits.n_features,)
    assert y.shape == ()


def test_splits_add_up_and_do_not_overlap(splits):
    # Repeat the split from prepare.py, with the same seed, to see the row ids.
    frame = make_rentals()
    train_df, rest = train_test_split(frame, test_size=0.4, random_state=42)
    val_df, test_df = train_test_split(rest, test_size=0.5, random_state=42)

    # It is the same split prepare.py made: the targets match row for row.
    assert torch.equal(
        splits.train.y,
        torch.tensor(train_df["rentals"].to_numpy(), dtype=torch.float32),
    )
    assert torch.equal(
        splits.val.y, torch.tensor(val_df["rentals"].to_numpy(), dtype=torch.float32)
    )
    assert torch.equal(
        splits.test.y, torch.tensor(test_df["rentals"].to_numpy(), dtype=torch.float32)
    )

    assert (len(train_df), len(val_df), len(test_df)) == (3600, 1200, 1200)
    train_ids, val_ids, test_ids = (
        set(train_df.index),
        set(val_df.index),
        set(test_df.index),
    )
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert train_ids | val_ids | test_ids == set(frame.index)


def test_scaler_was_fitted_on_training_rows_only(splits):
    train_mean = splits.train.x[:, :5].mean(dim=0)
    val_mean = splits.val.x[:, :5].mean(dim=0)
    assert torch.allclose(train_mean, torch.zeros(5), atol=1e-5)
    assert not torch.allclose(val_mean, torch.zeros(5), atol=1e-3)


def test_training_loader_shuffles_but_is_repeatable(splits):
    first = next(iter(make_loaders(splits, seed=1)[0]))[1]
    again = next(iter(make_loaders(splits, seed=1)[0]))[1]
    in_order = splits.train.y[:64]
    assert torch.equal(first, again)
    assert not torch.equal(first, in_order)
