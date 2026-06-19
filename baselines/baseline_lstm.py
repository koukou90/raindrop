import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """LSTM基线：使用conc+vel+phys全量输入做seq2seq多步回归。"""

    def __init__(
        self,
        args=None,
        seq_len=10,
        conc_dim=32,
        vel_dim=32,
        phys_dim=5,
        pred_len=5,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()

        if args is not None:
            seq_len = getattr(args, 'seq_len', seq_len)
            pred_len = getattr(args, 'pred_len', pred_len)
            hidden_dim = getattr(args, 'lstm_hidden', hidden_dim)
            dropout = getattr(args, 'dropout', dropout)

        self.seq_len = seq_len
        self.pred_len = pred_len
        input_dim = conc_dim + vel_dim + phys_dim

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, conc, vel, phys):
        x = torch.cat([conc, vel, phys], dim=-1)

        enc_out, (h_n, c_n) = self.encoder(x)
        dec_input = enc_out[:, -1:, :]
        hidden = (h_n[-1:].contiguous(), c_n[-1:].contiguous())

        preds = []
        for _ in range(self.pred_len):
            dec_out, hidden = self.decoder(dec_input, hidden)
            step_pred = self.head(dec_out.squeeze(1))
            preds.append(step_pred)
            dec_input = dec_out

        return torch.cat(preds, dim=1)
