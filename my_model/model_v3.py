"""
TripleStreamFusionNetworkV3
===========================

在 v2 基础上的进一步增强版，核心目标是提升短时首步（T+1）精度并保持多步稳定性。

相对 v2 的主要变化：
1) 引入 Persistence 残差学习：
   - 输出不再直接回归雨强，而是预测“相对持续性预报的残差”。
   - 最终输出 = Persistence + gate * Residual。
   - 通过将残差头末层零初始化，模型初始行为接近 Persistence，训练更稳。

2) 继续保留 v2 的多任务学习：
   - 主任务：RainRate 多步预测
   - 辅助任务：Dm/LogNw/LWC/Z 多步预测

3) 保持物理先验编码与三流结构：
   - 数浓度/速度分支使用 BinAwareSpectrumEncoder（位置编码 + bin注意力）
   - 物理量分支使用 LSTM
"""

import torch
import torch.nn as nn

# 与数据处理脚本约定一致的 32 个粒径中心（mm）
DIAMETER_BINS = [
    0.062, 0.187, 0.312, 0.437, 0.562, 0.687, 0.812, 0.937,
    1.062, 1.17553125, 1.33515625, 1.54765625, 1.75078125, 1.94453125, 2.12890625, 2.384375,
    2.696875, 2.971875, 3.209375, 3.409375, 3.85, 4.55, 5.25, 5.95,
    6.65, 7.7, 9.1, 10.5, 11.9, 13.3, 15.05, 17.15
]


def _terminal_velocity_from_diameter(diameter_mm):
    """
    经验终端速度关系（Atlas 常见近似形式之一）：
        Vt = 9.65 - 10.3 * exp(-0.6 * D)
    D 单位为 mm，Vt 单位近似为 m/s。
    """
    return 9.65 - 10.3 * torch.exp(-0.6 * diameter_mm)


class BinAwareSpectrumEncoder(nn.Module):
    """基于物理位置编码和 bin 级注意力的谱特征编码器。"""

    def __init__(self, input_dim=32, hidden_dim=64, output_dim=32, num_heads=4, dropout=0.2):
        super().__init__()

        if input_dim != len(DIAMETER_BINS):
            raise ValueError(
                f"input_dim={input_dim} 与粒径 bins 数量 {len(DIAMETER_BINS)} 不一致。"
            )

        diameter = torch.tensor(DIAMETER_BINS, dtype=torch.float32)
        diameter_norm = (diameter - diameter.min()) / (diameter.max() - diameter.min() + 1e-6)

        vt = _terminal_velocity_from_diameter(diameter)
        vt_norm = (vt - vt.min()) / (vt.max() - vt.min() + 1e-6)

        # (1, 1, bins, 1)，便于按 batch/time 广播拼接
        self.register_buffer('diameter_pos', diameter_norm.view(1, 1, input_dim, 1))
        self.register_buffer('velocity_pos', vt_norm.view(1, 1, input_dim, 1))

        # 每个 bin token 的输入特征: [观测值, 粒径位置, 终端速度位置]
        self.token_proj = nn.Linear(3, hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 32)
        Returns:
            (batch, seq_len, output_dim)
        """
        batch_size, seq_len, n_bins = x.shape

        value = x.unsqueeze(-1)  # (batch, seq_len, 32, 1)
        d_pos = self.diameter_pos.expand(batch_size, seq_len, n_bins, 1)
        v_pos = self.velocity_pos.expand(batch_size, seq_len, n_bins, 1)
        token = torch.cat([value, d_pos, v_pos], dim=-1)  # (batch, seq_len, 32, 3)

        token = token.reshape(batch_size * seq_len, n_bins, 3)
        h = self.token_proj(token)

        attn_out, _ = self.self_attn(h, h, h, need_weights=False)
        h = self.norm1(h + attn_out)
        h = self.norm2(h + self.ffn(h))

        # 对 bin 维度做全局池化，得到每个时间步的谱表示
        pooled = h.mean(dim=1)  # (batch*seq_len, hidden_dim)
        features = self.out_proj(pooled)
        features = features.reshape(batch_size, seq_len, -1)
        return features


class TripleStreamFusionNetworkV3(nn.Module):
    """三流融合 v3：Persistence 残差建模 + 多任务辅助学习。"""

    supports_aux_task = True

    def __init__(
        self,
        args=None,
        conc_dim=32,
        vel_dim=32,
        phys_dim=5,
        cnn_hidden=64,
        cnn_output=32,
        lstm_hidden=64,
        decoder_hidden=128,
        pred_len=5,
        dropout=0.2,
        aux_dim=4
    ):
        super().__init__()

        if args is not None:
            conc_dim = getattr(args, 'conc_dim', conc_dim)
            vel_dim = getattr(args, 'vel_dim', vel_dim)
            phys_dim = getattr(args, 'phys_dim', phys_dim)
            cnn_hidden = getattr(args, 'cnn_hidden', cnn_hidden)
            cnn_output = getattr(args, 'cnn_output', cnn_output)
            lstm_hidden = getattr(args, 'lstm_hidden', lstm_hidden)
            decoder_hidden = getattr(args, 'decoder_hidden', decoder_hidden)
            pred_len = getattr(args, 'pred_len', pred_len)
            dropout = getattr(args, 'dropout', dropout)

        self.pred_len = pred_len
        self.aux_dim = aux_dim

        # 将标准化输入中的 RainRate 反变换回原始 mm/h 尺度
        self.register_buffer('rain_mean', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('rain_std', torch.tensor(1.0, dtype=torch.float32))

        self.conc_extractor = BinAwareSpectrumEncoder(
            input_dim=conc_dim,
            hidden_dim=cnn_hidden,
            output_dim=cnn_output,
            num_heads=4,
            dropout=dropout
        )
        self.vel_extractor = BinAwareSpectrumEncoder(
            input_dim=vel_dim,
            hidden_dim=cnn_hidden,
            output_dim=cnn_output,
            num_heads=4,
            dropout=dropout
        )
        self.phys_lstm = nn.LSTM(
            input_size=phys_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        fusion_dim = cnn_output + cnn_output + lstm_hidden
        self.stream_gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid()
        )

        self.decoder_lstm = nn.LSTM(
            input_size=fusion_dim,
            hidden_size=decoder_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        self.decoder_proj = nn.Linear(decoder_hidden, fusion_dim)

        # 残差头：预测相对 persistence 的校正量
        self.residual_head = nn.Sequential(
            nn.Linear(decoder_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        # 门控头：控制残差注入强度
        self.mix_gate_head = nn.Sequential(
            nn.Linear(decoder_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.aux_head = nn.Sequential(
            nn.Linear(decoder_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, aux_dim)
        )

        # 让模型初始行为更接近 Persistence
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def set_scaler_stats(self, phys_mean, phys_std):
        """设置 RainRate 的标准化统计量（phys 第 1 列）。"""
        rain_mean = float(phys_mean.reshape(-1)[0])
        rain_std = float(phys_std.reshape(-1)[0])
        if rain_std == 0.0:
            rain_std = 1.0
        self.rain_mean.fill_(rain_mean)
        self.rain_std.fill_(rain_std)

    def forward(self, conc, vel, phys, return_aux=False):
        h_conc = self.conc_extractor(conc)
        h_vel = self.vel_extractor(vel)
        h_phys, _ = self.phys_lstm(phys)

        h_fused = torch.cat([h_conc, h_vel, h_phys], dim=-1)
        h_fused = h_fused * self.stream_gate(h_fused)

        _, (h_n, c_n) = self.decoder_lstm(h_fused)
        decoder_input = h_fused[:, -1:, :]
        hidden = (h_n, c_n)

        # 持续性基准（原始尺度 mm/h）
        persistence_scaled = phys[:, -1, 0]
        persistence_raw = persistence_scaled * self.rain_std + self.rain_mean
        persistence_seq = persistence_raw.unsqueeze(1).repeat(1, self.pred_len)

        rain_preds = []
        aux_preds = []

        for step in range(self.pred_len):
            out, hidden = self.decoder_lstm(decoder_input, hidden)
            state = out.squeeze(1)

            residual = self.residual_head(state)        # (batch, 1)
            mix_gate = self.mix_gate_head(state)        # (batch, 1)
            base_step = persistence_seq[:, step:step + 1]
            rain_step = base_step + mix_gate * residual

            aux_step = self.aux_head(state)             # (batch, 4)

            rain_preds.append(rain_step)
            aux_preds.append(aux_step.unsqueeze(1))
            decoder_input = self.decoder_proj(out)

        rain_pred = torch.cat(rain_preds, dim=1)  # (batch, pred_len)
        if return_aux:
            aux_pred = torch.cat(aux_preds, dim=1)
            return rain_pred, aux_pred
        return rain_pred
