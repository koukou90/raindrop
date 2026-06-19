import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    """MLP基线：拼接全部输入后用多层感知机回归未来 pred_len。"""

    def __init__(
        self,
        args=None,
        seq_len=10,
        conc_dim=32,
        vel_dim=32,
        phys_dim=5,
        pred_len=5,
        hidden_dim=256,
        dropout=0.2
    ):
        super().__init__()

        if args is not None:
            seq_len = getattr(args, 'seq_len', seq_len)
            conc_dim = getattr(args, 'conc_dim', conc_dim)
            vel_dim = getattr(args, 'vel_dim', vel_dim)
            phys_dim = getattr(args, 'phys_dim', phys_dim)
            pred_len = getattr(args, 'pred_len', pred_len)
            hidden_dim = getattr(args, 'mlp_hidden', hidden_dim)
            dropout = getattr(args, 'dropout', dropout)

        input_dim = seq_len * (conc_dim + vel_dim + phys_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, pred_len)
        )

    def forward(self, conc, vel, phys):
        x = torch.cat([conc, vel, phys], dim=-1)
        x = x.reshape(x.size(0), -1)
        return self.net(x)
