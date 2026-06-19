"""
TripleStreamFusionNetworkV2
===========================

本文件实现相对于 v1(`my_model/model_v1.py`) 的升级版三流融合模型。

与上一版(v1)相比的核心变化
------------------------
1) 想法A：引入“粒径 bin 物理位置编码 + bin 级注意力”
   - v1 使用 TimeDistributed1DCNN 对 32 个粒径 bin 做局部卷积汇聚。
   - v2 新增 BinAwareSpectrumEncoder：
     - 为每个 bin 显式注入两个物理位置先验：
       a. 粒径中心归一化位置
       b. 基于经验公式计算的终端速度归一化位置
     - 在每个时间步内，对 32 个 bin token 做 Multi-Head Self-Attention，
       让模型学习不同粒径 bin 之间的全局依赖关系。

2) 想法B：引入多任务学习（主任务 + 辅助任务）
   - 主任务：预测未来 pred_len 步 RainRate。
   - 辅助任务：同步预测未来 pred_len 步的 Dm/LogNw/LWC/Z（4个量）。
   - 训练时联合优化主任务与辅助任务，提升表征稳定性和泛化能力。

3) 输出约束一致性
   - v2 主头不再在模型内部硬编码 ReLU 非负约束；
   - 统一由训练/测试流程在模型外执行同一输出约束策略，确保跨模型公平对比。

模型功能介绍
-----------
- 输入：
  conc: (batch, seq_len, 32)  数浓度谱
  vel:  (batch, seq_len, 32)  速度谱
  phys: (batch, seq_len, 5)   物理量序列 [RainRate, Dm, LogNw, LWC, Z]

- 编码：
  1) conc 分支：BinAwareSpectrumEncoder（位置编码 + bin 注意力）
  2) vel  分支：BinAwareSpectrumEncoder（位置编码 + bin 注意力）
  3) phys 分支：两层 LSTM 提取宏观演变特征

- 融合与解码：
  - 三流特征拼接后，经门控融合层抑制冗余信息
  - 自回归 LSTM Decoder 逐步生成未来 pred_len 步隐藏状态
  - 每一步隐藏状态同时送入：
    a) 雨强主头（RainRate）
    b) 物理量辅助头（Dm/LogNw/LWC/Z）

- 输出：
  - 默认返回 rain_pred: (batch, pred_len)
  - 若 return_aux=True，返回 (rain_pred, aux_pred)
    其中 aux_pred 形状为 (batch, pred_len, 4)
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


class TripleStreamFusionNetworkV2(nn.Module):
    """三流融合升级版：位置感知谱编码 + 多任务辅助学习。"""

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

        # 分支1: 数浓度（bin 位置编码 + 注意力）
        self.conc_extractor = BinAwareSpectrumEncoder(
            input_dim=conc_dim,
            hidden_dim=cnn_hidden,
            output_dim=cnn_output,
            num_heads=4,
            dropout=dropout
        )

        # 分支2: 速度（bin 位置编码 + 注意力）
        self.vel_extractor = BinAwareSpectrumEncoder(
            input_dim=vel_dim,
            hidden_dim=cnn_hidden,
            output_dim=cnn_output,
            num_heads=4,
            dropout=dropout
        )

        # 分支3: 宏观物理量时序
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

        # 主任务头：RainRate
        self.rain_head = nn.Sequential(
            nn.Linear(decoder_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # 辅助任务头：Dm/LogNw/LWC/Z
        self.aux_head = nn.Sequential(
            nn.Linear(decoder_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, aux_dim)
        )

    def forward(self, conc, vel, phys, return_aux=False):
        """
        Args:
            conc: (batch, seq_len, 32)
            vel: (batch, seq_len, 32)
            phys: (batch, seq_len, 5)
            return_aux: 是否返回辅助任务输出
        Returns:
            rain_pred: (batch, pred_len)
            或 (rain_pred, aux_pred)，aux_pred: (batch, pred_len, 4)
        """
        h_conc = self.conc_extractor(conc)
        h_vel = self.vel_extractor(vel)
        h_phys, _ = self.phys_lstm(phys)

        h_fused = torch.cat([h_conc, h_vel, h_phys], dim=-1)
        h_fused = h_fused * self.stream_gate(h_fused)

        _, (h_n, c_n) = self.decoder_lstm(h_fused)

        decoder_input = h_fused[:, -1:, :]
        hidden = (h_n, c_n)

        rain_preds = []
        aux_preds = []

        for _ in range(self.pred_len):
            out, hidden = self.decoder_lstm(decoder_input, hidden)
            state = out.squeeze(1)

            rain_step = self.rain_head(state)      # (batch, 1)
            aux_step = self.aux_head(state)        # (batch, 4)

            rain_preds.append(rain_step)
            aux_preds.append(aux_step.unsqueeze(1))

            decoder_input = self.decoder_proj(out)

        rain_pred = torch.cat(rain_preds, dim=1)

        if return_aux:
            aux_pred = torch.cat(aux_preds, dim=1)
            return rain_pred, aux_pred

        return rain_pred
