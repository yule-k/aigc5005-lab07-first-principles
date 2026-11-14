"""Train BikeNet with a hand-written loop, compare it with baselines, save reports."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import mean_absolute_error  # noqa: E402
from torch import nn  # noqa: E402

from model import BikeNet  # noqa: E402
from prepare import make_loaders, make_splits  # noqa: E402

EPOCHS = 150
LEARNING_RATE = 3e-3
HIDDEN = 64
BATCH_SIZE = 64
SEED = 42
REPORTS = Path("reports")

# Each experiment changes one setting from the main run.
EXPERIMENTS = [
    # name, what changed, hidden, learning rate
    ("lr_tiny", "learning rate 3e-3 -> 1e-5", HIDDEN, 1e-5),
    ("lr_huge", "learning rate 3e-3 -> 1e-1", HIDDEN, 1e-1),
    ("wide", "hidden size 64 -> 1024", 1024, LEARNING_RATE),
]


def run_epoch(model, loader, loss_fn, optimizer=None):
    """One pass over a loader. Trains if given an optimizer, else only measures.

    Returns the mean loss and the mean absolute error, in rentals.
    """
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


def plot_curves(history, best_epoch, title, path):
    """Training and validation loss for every epoch, with the best epoch marked."""
    frame = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(frame["epoch"], frame["train_loss"], label="train loss")
    ax.plot(frame["epoch"], frame["val_loss"], label="validation loss")
    ax.axvline(
        best_epoch, color="grey", linestyle="--", label=f"best epoch ({best_epoch})"
    )
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    splits = make_splits(SEED)
    train_loader, val_loader, test_loader = make_loaders(splits, BATCH_SIZE, SEED)

    # The main run
    model = BikeNet(splits.n_features, HIDDEN)
    history, best_epoch = train_model(
        model, train_loader, val_loader, EPOCHS, LEARNING_RATE
    )
    for row in history:
        if row["epoch"] == 1 or row["epoch"] % 10 == 0:
            print(
                f"epoch {row['epoch']:3d}  train loss {row['train_loss']:.2f}  "
                f"val loss {row['val_loss']:.2f}  val MAE {row['val_mae']:.2f}"
            )
    val_mae = history[best_epoch - 1]["val_mae"]
    print(f"Best epoch by validation loss: {best_epoch}")

    pd.DataFrame(history).to_csv(REPORTS / "history.csv", index=False)
    plot_curves(history, best_epoch, "Main run", REPORTS / "learning_curves.png")

    # The experiments: same seed, one setting changed each time
    rows = [
        {
            "run": "main",
            "change": "none",
            "best_epoch": best_epoch,
            "best_val_mae": val_mae,
            "final_train_loss": history[-1]["train_loss"],
            "final_val_loss": history[-1]["val_loss"],
        }
    ]
    for name, change, hidden, lr in EXPERIMENTS:
        torch.manual_seed(SEED)
        exp_train, exp_val, _ = make_loaders(splits, BATCH_SIZE, SEED)
        exp_model = BikeNet(splits.n_features, hidden)
        exp_history, exp_best = train_model(exp_model, exp_train, exp_val, EPOCHS, lr)
        rows.append(
            {
                "run": name,
                "change": change,
                "best_epoch": exp_best,
                "best_val_mae": exp_history[exp_best - 1]["val_mae"],
                "final_train_loss": exp_history[-1]["train_loss"],
                "final_val_loss": exp_history[-1]["val_loss"],
            }
        )
        plot_curves(exp_history, exp_best, name, REPORTS / f"curves_{name}.png")
        print(f"Experiment {name}: best epoch {exp_best}")
    pd.DataFrame(rows).round(3).to_csv(REPORTS / "experiments.csv", index=False)

    # The two baselines, on exactly the same rows
    y_train = splits.train.y.numpy()
    mean = y_train.mean()
    linear = LinearRegression().fit(splits.train.x.numpy(), y_train)
    baselines = {
        "train_mean": {
            "val_mae": mean_absolute_error(
                splits.val.y, np.full(len(splits.val), mean)
            ),
            "test_mae": mean_absolute_error(
                splits.test.y, np.full(len(splits.test), mean)
            ),
        },
        "linear_regression": {
            "val_mae": mean_absolute_error(splits.val.y, linear.predict(splits.val.x)),
            "test_mae": mean_absolute_error(
                splits.test.y, linear.predict(splits.test.x)
            ),
        },
    }

    # The test set, used once, on the main run's best weights
    _, test_mae = run_epoch(model, test_loader, nn.MSELoss())

    results = {
        "train_mean": baselines["train_mean"],
        "linear_regression": baselines["linear_regression"],
        "network": {"val_mae": val_mae, "test_mae": test_mae},
    }
    for name, r in results.items():
        print(f"{name:18s} val MAE {r['val_mae']:.2f}  test MAE {r['test_mae']:.2f}")

    with open(REPORTS / "metrics.json", "w") as f:
        json.dump(
            {
                "results": {
                    name: {k: round(float(v), 3) for k, v in r.items()}
                    for name, r in results.items()
                },
                "best_epoch": best_epoch,
                "settings": {
                    "epochs": EPOCHS,
                    "learning_rate": LEARNING_RATE,
                    "hidden": HIDDEN,
                    "batch_size": BATCH_SIZE,
                    "seed": SEED,
                },
                "torch": torch.__version__,
            },
            f,
            indent=2,
        )
    print("Saved history.csv, learning_curves.png, experiments.csv, metrics.json")


if __name__ == "__main__":
    main()
