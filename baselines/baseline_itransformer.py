import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ITransformerBaseline(nn.Module):
    """
    iTransformer 官方思路适配版。
    保留 inverted embedding + variate attention 主干，并适配当前接口。
    """

    def __init__(
        self,
        args=None,
        seq_len=10,
        pred_len=5,
        conc_dim=32,
        vel_dim=32,
        phys_dim=5,
        d_model=128,
        n_heads=4,
        e_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        if args is not None:
            seq_len = getattr(args, 'seq_len', seq_len)
            pred_len = getattr(args, 'pred_len', pred_len)
            d_model = getattr(args, 'decoder_hidden', d_model)
            dropout = getattr(args, 'dropout', dropout)

        self.pred_len = pred_len
        self.target_idx = conc_dim + vel_dim  # phys 第一个维度 RainRate

        self.enc_embedding = DataEmbeddingInverted(seq_len, d_model, dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(FullAttention(False, attention_dropout=dropout), d_model, n_heads),
                    d_model=d_model,
                    d_ff=d_model * 4,
                    dropout=dropout,
                    activation='gelu',
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.projection = nn.Linear(d_model, pred_len)

    def forward(self, conc, vel, phys):
        x = torch.cat([conc, vel, phys], dim=-1)  # [B, T, C]

        # instance normalization
        means = x.mean(1, keepdim=True).detach()
        x_norm = x - means
        stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_norm = x_norm / stdev

        # inverted embedding: [B, T, C] -> [B, C, D]
        enc_out = self.enc_embedding(x_norm)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        # [B, C, D] -> [B, C, H] -> [B, H, C]
        dec_out = self.projection(enc_out)
        pred_all = dec_out.permute(0, 2, 1)

        # de-normalization
        pred_all = pred_all * stdev + means
        return pred_all[:, :, self.target_idx]


class DataEmbeddingInverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, C] -> [B, C, T] -> [B, C, D]
        x = x.permute(0, 2, 1)
        x = self.value_embedding(x)
        return self.dropout(x)


class FullAttention(nn.Module):
    def __init__(self, mask_flag=False, scale=None, attention_dropout=0.1):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        b, l, h, e = queries.shape
        scale = self.scale or 1.0 / math.sqrt(e)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag and attn_mask is not None:
            scores.masked_fill_(attn_mask.mask, -torch.inf)
        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        out = torch.einsum("bhls,bshd->blhd", attn, values)
        return out.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        b, l, _ = queries.shape
        _, s, _ = keys.shape
        h = self.n_heads
        queries = self.query_projection(queries).view(b, l, h, -1)
        keys = self.key_projection(keys).view(b, s, h, -1)
        values = self.value_projection(values).view(b, s, h, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(b, l, -1)
        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns
