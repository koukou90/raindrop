import torch
import torch.nn as nn


class CNNTransformerBaseline(nn.Module):
    """CNN-Transformer 基线：时序卷积 + Transformer 时序编码。"""

    def __init__(
        self,
        args=None,
        pred_len=5,
        conc_dim=32,
        vel_dim=32,
        phys_dim=5,
        cnn_hidden=64,
        d_model=128,
        n_heads=4,
        e_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        if args is not None:
            pred_len = getattr(args, 'pred_len', pred_len)
            cnn_hidden = getattr(args, 'cnn_hidden', cnn_hidden)
            d_model = getattr(args, 'decoder_hidden', d_model)
            dropout = getattr(args, 'dropout', dropout)

        input_dim = conc_dim + vel_dim + phys_dim
        self.pred_len = pred_len

        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_hidden),
            nn.Conv1d(cnn_hidden, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(d_model),
            nn.Dropout(dropout),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=e_layers)
        self.head = nn.Linear(d_model * 2, pred_len)

    def forward(self, conc, vel, phys):
        x = torch.cat([conc, vel, phys], dim=-1)   # (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1)                     # (batch, input_dim, seq_len)
        x = self.cnn(x).permute(0, 2, 1)           # (batch, seq_len, d_model)
        x = self.encoder(x)

        last_feat = x[:, -1, :]
        mean_feat = x.mean(dim=1)
        fused = torch.cat([last_feat, mean_feat], dim=-1)
        return self.head(fused)
