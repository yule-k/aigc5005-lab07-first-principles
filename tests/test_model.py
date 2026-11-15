import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import BikeNet
from train import run_epoch


@pytest.mark.parametrize("hidden", [8, 64, 256])
def test_network_returns_one_value_per_row(hidden):
    model = BikeNet(n_features=8, hidden=hidden)
    out = model(torch.zeros(5, 8))
    assert out.shape == (5,)


def test_one_step_changes_every_layer():
    torch.manual_seed(0)
    model = BikeNet(8)
    before = [p.clone() for p in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = nn.MSELoss()(model(torch.randn(16, 8)), torch.rand(16) * 20)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    after = list(model.parameters())
    assert all(not torch.equal(a, b) for a, b in zip(before, after, strict=True))


def test_network_can_memorize_one_small_batch(splits):
    torch.manual_seed(0)
    x, y = splits.train.x[:16], splits.train.y[:16]
    loader = DataLoader(TensorDataset(x, y), batch_size=16)
    model = BikeNet(splits.n_features)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    start, _ = run_epoch(model, loader, loss_fn)
    for _ in range(300):
        run_epoch(model, loader, loss_fn, optimizer)
    end, _ = run_epoch(model, loader, loss_fn)
    assert end < 0.05 * start
