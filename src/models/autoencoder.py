"""Autoencoder MLP (PyTorch). Opcional: solo se importa si se usa."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.models.base import BaseAnomalyModel

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False


if _TORCH:
    class _MLPAE(nn.Module):
        def __init__(self, in_dim: int, hidden: list[int]):
            super().__init__()
            enc, prev = [], in_dim
            for h in hidden:
                enc += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            dec, rev = [], hidden[-1]
            for h in reversed(hidden[:-1]) + [in_dim]:
                dec += [nn.Linear(rev, h), nn.ReLU() if h != in_dim else nn.Identity()]
                rev = h
            self.net = nn.Sequential(*enc, *dec)

        def forward(self, x):
            return self.net(x)


class AutoencoderModel(BaseAnomalyModel):
    name = "autoencoder"

    def __init__(self, hidden: list[int] = None, epochs: int = 50,
                 batch_size: int = 64, lr: float = 1e-3, **params):
        if not _TORCH:
            raise ImportError("PyTorch no está instalado. `pip install torch`.")
        super().__init__(hidden=hidden or [32, 16, 8], epochs=epochs,
                         batch_size=batch_size, lr=lr, **params)
        self.net = None
        self.input_dim_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.input_dim_ = X.shape[1]
        self.net = _MLPAE(self.input_dim_, self.params["hidden"])
        opt = torch.optim.Adam(self.net.parameters(), lr=self.params["lr"])
        loss_fn = nn.MSELoss()
        ds = torch.utils.data.TensorDataset(torch.from_numpy(X))
        dl = torch.utils.data.DataLoader(ds, batch_size=self.params["batch_size"], shuffle=True)
        self.net.train()
        for _ in range(self.params["epochs"]):
            for (batch,) in dl:
                opt.zero_grad()
                out = self.net(batch)
                loss = loss_fn(out, batch)
                loss.backward()
                opt.step()
        self.net.eval()
        return self

    def score_samples(self, X):
        X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            rec = self.net(X)
            return ((rec - X) ** 2).mean(dim=1).cpu().numpy()

    def save(self, dir_path):
        torch.save(self.net.state_dict(), dir_path / "model.pt")
        (dir_path / "meta.json").write_text(json.dumps(
            {"type": self.name, "params": self.params, "threshold_": self.threshold_,
             "input_dim_": self.input_dim_}, indent=2))

    def load(self, dir_path):
        meta = json.loads((dir_path / "meta.json").read_text())
        self.params = meta["params"]
        self.threshold_ = meta["threshold_"]
        self.input_dim_ = meta["input_dim_"]
        self.net = _MLPAE(self.input_dim_, self.params["hidden"])
        self.net.load_state_dict(torch.load(dir_path / "model.pt"))
        self.net.eval()
        return self