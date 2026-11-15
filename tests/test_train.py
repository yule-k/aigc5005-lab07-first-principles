import torch
from torch import nn

from model import BikeNet
from prepare import make_loaders
from train import run_epoch, train_model


def test_validation_epoch_leaves_weights_unchanged(splits):
    torch.manual_seed(0)
    _, val_loader, _ = make_loaders(splits)
    model = BikeNet(splits.n_features)
    before = [p.clone() for p in model.parameters()]
    run_epoch(model, val_loader, nn.MSELoss())
    after = list(model.parameters())
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))


def test_train_model_logs_every_epoch_and_keeps_best_weights(splits):
    torch.manual_seed(0)
    train_loader, val_loader, _ = make_loaders(splits)
    model = BikeNet(splits.n_features)
    history, best_epoch = train_model(model, train_loader, val_loader, 5, 3e-3)

    assert len(history) == 5
    for row in history:
        assert set(row) == {"epoch", "train_loss", "val_loss", "val_mae"}

    best_val_loss = min(row["val_loss"] for row in history)
    assert history[best_epoch - 1]["val_loss"] == best_val_loss
    val_loss_now, _ = run_epoch(model, val_loader, nn.MSELoss())
    assert abs(val_loss_now - best_val_loss) < 1e-4
