import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""

    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class ResBlock(nn.Module):
    """TiDE residual block."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, bias=True):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.identity_proj = nn.Linear(input_dim, output_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.ln = LayerNorm(output_dim, bias=bias)

    def forward(self, x):
        identity = self.identity_proj(x)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + identity
        out = self.ln(out)
        return out


class TIDEBaseline(nn.Module):
    """
    TiDE 官方思路适配版（单目标 RainRate）。
    保留:
      - feature encoder
      - dense encoder/decoder blocks
      - temporal decoder
      - residual projection
    """

    def __init__(
        self,
        args=None,
        seq_len=10,
        pred_len=5,
        hidden_dim=256,
        dropout=0.2,
        time_feature_dim=8,
        feature_encode_dim=16,
        encoder_layers=2,
        decoder_layers=2,
        temporal_decoder_hidden=64,
    ):
        super().__init__()

        if args is not None:
            seq_len = getattr(args, 'seq_len', seq_len)
            pred_len = getattr(args, 'pred_len', pred_len)
            hidden_dim = getattr(args, 'mlp_hidden', hidden_dim)
            dropout = getattr(args, 'dropout', dropout)

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.time_feature_dim = time_feature_dim
        self.feature_encode_dim = feature_encode_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        # time-varying covariates encoder
        self.feature_encoder = ResBlock(
            input_dim=self.time_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=self.feature_encode_dim,
            dropout=dropout,
        )

        # main encoder
        flatten_dim = self.seq_len + (self.seq_len + self.pred_len) * self.feature_encode_dim
        enc_blocks = [ResBlock(flatten_dim, hidden_dim, hidden_dim, dropout)]
        for _ in range(self.encoder_layers - 1):
            enc_blocks.append(ResBlock(hidden_dim, hidden_dim, hidden_dim, dropout))
        self.encoders = nn.Sequential(*enc_blocks)

        # decoder
        dec_blocks = []
        for _ in range(self.decoder_layers - 1):
            dec_blocks.append(ResBlock(hidden_dim, hidden_dim, hidden_dim, dropout))
        dec_blocks.append(ResBlock(hidden_dim, hidden_dim, self.pred_len, dropout))
        self.decoders = nn.Sequential(*dec_blocks)

        # temporal decoder + residual
        self.temporal_decoder = ResBlock(
            input_dim=1 + self.feature_encode_dim,
            hidden_dim=temporal_decoder_hidden,
            output_dim=1,
            dropout=dropout,
        )
        self.residual_proj = nn.Linear(self.seq_len, self.pred_len)

    def _build_time_features(self, batch_size, device):
        # 无外部时间特征时，构造相对位置特征占位（与 refs 接口保持一致）
        total_len = self.seq_len + self.pred_len
        t = torch.linspace(0.0, 1.0, steps=total_len, device=device).unsqueeze(1)  # [T,1]
        feats = torch.cat(
            [
                t,
                t ** 2,
                torch.sin(2 * torch.pi * t),
                torch.cos(2 * torch.pi * t),
                torch.sin(4 * torch.pi * t),
                torch.cos(4 * torch.pi * t),
                (t > 0.5).float(),
                torch.ones_like(t),
            ],
            dim=1,
        )  # [T, 8]
        return feats.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, T, 8]

    def _forecast_univariate(self, x_enc, x_time):
        # x_enc: [B, L]
        b = x_enc.shape[0]

        means = x_enc.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc_norm = (x_enc - means) / stdev

        encoded_features = self.feature_encoder(x_time)  # [B, L+H, E]
        flat_features = encoded_features.reshape(b, -1)
        encoder_input = torch.cat([x_enc_norm, flat_features], dim=-1)
        hidden = self.encoders(encoder_input)
        decoded_global = self.decoders(hidden)  # [B, H]

        future_features = encoded_features[:, self.seq_len:, :]  # [B, H, E]
        temporal_input = torch.cat([decoded_global.unsqueeze(-1), future_features], dim=-1)  # [B,H,1+E]

        temporal_input_flat = temporal_input.reshape(b * self.pred_len, -1)
        decoded_temporal_flat = self.temporal_decoder(temporal_input_flat)
        decoded_temporal = decoded_temporal_flat.reshape(b, self.pred_len)

        residual = self.residual_proj(x_enc_norm)
        pred_norm = decoded_temporal + residual
        pred = pred_norm * stdev + means
        return pred

    def forward(self, conc, vel, phys):
        # TiDE 在本任务中按单目标 RainRate 预测
        rain_hist = phys[:, :, 0]  # [B, L]
        x_time = self._build_time_features(rain_hist.shape[0], rain_hist.device)
        return self._forecast_univariate(rain_hist, x_time)
