import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchTSTBaseline(nn.Module):
    """
    PatchTST 官方思路适配版。
    保留 patching + channel-independent transformer + flatten head 主干，
    并适配到当前工程输入输出接口。
    """

    def __init__(
        self,
        args=None,
        seq_len=10,
        pred_len=5,
        conc_dim=32,
        vel_dim=32,
        phys_dim=5,
        patch_len=4,
        patch_stride=2,
        d_model=128,
        n_heads=4,
        e_layers=2,
        dropout=0.2,
        d_ff=None,
    ):
        super().__init__()

        if args is not None:
            seq_len = getattr(args, 'seq_len', seq_len)
            pred_len = getattr(args, 'pred_len', pred_len)
            d_model = getattr(args, 'decoder_hidden', d_model)
            dropout = getattr(args, 'dropout', dropout)
            n_heads = max(1, min(n_heads, d_model))

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = conc_dim + vel_dim + phys_dim
        self.target_idx = conc_dim + vel_dim  # RainRate

        self.patch_len = max(2, min(patch_len, seq_len))
        self.patch_stride = max(1, patch_stride)
        self.patch_padding = self.patch_stride

        self.patch_embedding = PatchEmbedding(
            d_model=d_model,
            patch_len=self.patch_len,
            stride=self.patch_stride,
            padding=self.patch_padding,
            dropout=dropout,
        )

        d_ff = d_ff or d_model * 4
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(FullAttention(False, attention_dropout=dropout), d_model, n_heads),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation='gelu',
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)),
        )

        num_patch = int((self.seq_len - self.patch_len) / self.patch_stride + 2)
        head_nf = d_model * num_patch
        self.head = FlattenHead(head_nf, pred_len, head_dropout=dropout)

    def forward(self, conc, vel, phys):
        x = torch.cat([conc, vel, phys], dim=-1)  # (B, L, C)

        # instance normalization
        means = x.mean(1, keepdim=True).detach()
        x_norm = x - means
        stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_norm = x_norm / stdev

        # [B, L, C] -> [B, C, L]
        x_norm = x_norm.permute(0, 2, 1)

        # patch embedding
        enc_out, n_vars = self.patch_embedding(x_norm)  # [B*C, Np, D]
        enc_out, _ = self.encoder(enc_out)              # [B*C, Np, D]

        # reshape for head
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))  # [B, C, Np, D]
        enc_out = enc_out.permute(0, 1, 3, 2)  # [B, C, D, Np]

        # [B, C, H] -> [B, H, C]
        dec_out = self.head(enc_out)
        pred_all = dec_out.permute(0, 2, 1)

        # de-normalization
        pred_all = pred_all * stdev + means
        return pred_all[:, :, self.target_idx]


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


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


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
