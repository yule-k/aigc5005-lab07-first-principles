"""Train BikeNet with a hand-written loop."""

import torch
from torch import nn

from model import BikeNet
from prepare import make_loaders, make_splits

SEED = 42
EPOCHS = 150
LEARNING_RATE = 3e-3
HIDDEN = 64
BATCH_SIZE = 64


def run_epoch(model, loader, loss_fn, optimizer=None):
    """One pass over a loader. Trains if given an optimizer, else only measures."""
    training = optimizer is not None
    model.train(training)
    total_loss, total_abs_error, count = 0.0, 0.0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            prediction = model(x)
            loss = loss_fn(prediction, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            total_abs_error += (prediction - y).abs().sum().item()
            count += len(y)
    return total_loss / count, total_abs_error / count


def train_model(model, train_loader, val_loader, epochs, learning_rate):
    """Train for many epochs, log every epoch, and put back the best weights."""
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    best_loss, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_mae = run_epoch(model, val_loader, loss_fn)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
            }
        )
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return history, best_epoch


def main() -> None:
    torch.manual_seed(SEED)
    splits = make_splits(SEED)
    train_loader, val_loader, _ = make_loaders(splits, BATCH_SIZE, SEED)
    model = BikeNet(splits.n_features, HIDDEN)
    history, best_epoch = train_model(
        model, train_loader, val_loader, EPOCHS, LEARNING_RATE
    )
    for row in history:
        if row["epoch"] == 1 or row["epoch"] % 20 == 0:
            print(
                f"epoch {row['epoch']:3d}  train loss {row['train_loss']:8.2f}  "
                f"val loss {row['val_loss']:8.2f}  val MAE {row['val_mae']:.2f}"
            )
    best = history[best_epoch - 1]
    print(f"Best epoch {best_epoch}: val MAE {best['val_mae']:.2f}")


if __name__ == "__main__":
    main()
