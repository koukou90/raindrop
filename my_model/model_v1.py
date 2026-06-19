import torch
import torch.nn as nn


class TimeDistributed1DCNN(nn.Module):
    """Time-Distributed 1D CNN for spectrum feature extraction"""

    def __init__(self, input_dim=32, hidden_dim=64, output_dim=32):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            features: (batch, seq_len, output_dim)
        """
        batch_size, seq_len, input_dim = x.shape

        # 重塑为 (batch*seq_len, 1, input_dim) for Conv1d
        x = x.view(batch_size * seq_len, 1, input_dim)

        # 应用CNN
        features = self.conv_layers(x)  # (batch*seq_len, output_dim, 1)
        features = features.squeeze(-1)  # (batch*seq_len, output_dim)

        # 重塑回 (batch, seq_len, output_dim)
        features = features.view(batch_size, seq_len, -1)

        return features


class TripleStreamFusionNetwork(nn.Module):
    """三流融合网络：数浓度 + 速度 + 宏观物理量"""

    def __init__(self, args=None,
                 conc_dim=32,
                 vel_dim=32,
                 phys_dim=5,
                 cnn_hidden=64,
                 cnn_output=32,
                 lstm_hidden=64,
                 decoder_hidden=128,
                 pred_len=5,
                 dropout=0.2):
        super().__init__()

        # 如果提供了args，优先从args中读取参数
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

        # 分支1: 数浓度特征提取器 (Time-Distributed 1D-CNN)
        self.conc_extractor = TimeDistributed1DCNN(
            input_dim=conc_dim,
            hidden_dim=cnn_hidden,
            output_dim=cnn_output
        )

        # 分支2: 速度特征提取器 (Time-Distributed 1D-CNN)
        self.vel_extractor = TimeDistributed1DCNN(
            input_dim=vel_dim,
            hidden_dim=cnn_hidden,
            output_dim=cnn_output
        )

        # 分支3: 宏观演变提取器 (LSTM)
        self.phys_lstm = nn.LSTM(
            input_size=phys_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        # 特征融合维度
        fusion_dim = cnn_output + cnn_output + lstm_hidden

        # 时序解码器 (LSTM Decoder)
        self.decoder_lstm = nn.LSTM(
            input_size=fusion_dim,
            hidden_size=decoder_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        # 将decoder输出映射回fusion_dim，作为下一时刻输入（自回归）
        self.decoder_proj = nn.Linear(decoder_hidden, fusion_dim)

        # 输出头
        self.output_head = nn.Sequential(
            nn.Linear(decoder_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, conc, vel, phys):
        """
        Args:
            conc: (batch, seq_len, 32) 数浓度
            vel: (batch, seq_len, 32) 速度
            phys: (batch, seq_len, 5) 物理量

        Returns:
            predictions: (batch, pred_len) 未来5分钟雨强预测
        """
        batch_size = conc.size(0)

        # 分支1: 数浓度特征
        h_conc = self.conc_extractor(conc)  # (batch, seq_len, cnn_output)

        # 分支2: 速度特征
        h_vel = self.vel_extractor(vel)  # (batch, seq_len, cnn_output)

        # 分支3: 宏观物理量特征
        h_phys, _ = self.phys_lstm(phys)  # (batch, seq_len, lstm_hidden)

        # 特征融合
        h_fused = torch.cat([h_conc, h_vel, h_phys], dim=-1)  # (batch, seq_len, fusion_dim)

        # 解码器：获取最终隐藏状态
        _, (h_n, c_n) = self.decoder_lstm(h_fused)

        # 自回归解码：逐步预测未来 pred_len 个时间步
        predictions = []
        decoder_input = h_fused[:, -1:, :]  # 使用最后一个时间步作为初始输入
        hidden = (h_n, c_n)

        for _ in range(self.pred_len):
            out, hidden = self.decoder_lstm(decoder_input, hidden)
            pred = self.output_head(out.squeeze(1))  # (batch, 1)
            predictions.append(pred)

            # 使用当前解码器输出生成下一步输入，形成真正自回归链
            decoder_input = self.decoder_proj(out)

        predictions = torch.cat(predictions, dim=1)  # (batch, pred_len)

        return predictions


class WeightedMSELoss(nn.Module):
    """连续加权均方误差损失"""

    def __init__(self, threshold=10.0, alpha=0.2):
        super().__init__()
        self.threshold = threshold
        self.alpha = alpha

    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch, pred_len)
            targets: (batch, pred_len)
        """
        # 计算权重: w = 1 + alpha * max(0, y - threshold)
        weights = 1.0 + self.alpha * torch.clamp(targets - self.threshold, min=0)

        # 加权MSE
        mse = (predictions - targets) ** 2
        weighted_mse = weights * mse

        return weighted_mse.mean()


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
