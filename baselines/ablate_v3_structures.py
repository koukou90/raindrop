import torch
import torch.nn as nn

from my_model.model_v1 import TimeDistributed1DCNN
from my_model.model_v3 import TripleStreamFusionNetworkV3


class _AblateV3Base(TripleStreamFusionNetworkV3):
    """V3 结构消融公共基类。"""

    supports_aux_task = True
    use_persistence = True
    use_mix_gate = True
    use_stream_gate = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # no_persistence 情况下，不再使用“零初始化残差=Persistence起点”
        if not self.use_persistence:
            nn.init.xavier_uniform_(self.residual_head[-1].weight)
            nn.init.zeros_(self.residual_head[-1].bias)

    def _apply_stream_gate(self, fused_feature):
        if not self.use_stream_gate:
            return fused_feature
        return fused_feature * self.stream_gate(fused_feature)

    def _compose_rain_step(self, state, base_step):
        if not self.use_persistence:
            # 直接回归雨强（去掉 persistence 残差框架）
            return self.residual_head(state)

        residual = self.residual_head(state)
        if self.use_mix_gate:
            mix_gate = self.mix_gate_head(state)
        else:
            mix_gate = torch.ones_like(residual)
        return base_step + mix_gate * residual

    def forward(self, conc, vel, phys, return_aux=False):
        h_conc = self.conc_extractor(conc)
        h_vel = self.vel_extractor(vel)
        h_phys, _ = self.phys_lstm(phys)

        h_fused = torch.cat([h_conc, h_vel, h_phys], dim=-1)
        h_fused = self._apply_stream_gate(h_fused)

        _, (h_n, c_n) = self.decoder_lstm(h_fused)
        decoder_input = h_fused[:, -1:, :]
        hidden = (h_n, c_n)

        persistence_scaled = phys[:, -1, 0]
        persistence_raw = persistence_scaled * self.rain_std + self.rain_mean
        persistence_seq = persistence_raw.unsqueeze(1).repeat(1, self.pred_len)

        rain_preds = []
        aux_preds = []

        for step in range(self.pred_len):
            out, hidden = self.decoder_lstm(decoder_input, hidden)
            state = out.squeeze(1)

            base_step = persistence_seq[:, step:step + 1]
            rain_step = self._compose_rain_step(state, base_step)
            rain_preds.append(rain_step)

            if self.supports_aux_task:
                aux_step = self.aux_head(state)
                aux_preds.append(aux_step.unsqueeze(1))

            decoder_input = self.decoder_proj(out)

        rain_pred = torch.cat(rain_preds, dim=1)
        if return_aux and self.supports_aux_task:
            aux_pred = torch.cat(aux_preds, dim=1)
            return rain_pred, aux_pred
        return rain_pred


class AblateV3NoPersistence(_AblateV3Base):
    """A1: 去掉 persistence 残差学习。"""

    use_persistence = False


class AblateV3NoMixGate(_AblateV3Base):
    """A2: 固定 mix gate=1（仅保留残差）。"""

    use_mix_gate = False


class AblateV3NoBinAware(_AblateV3Base):
    """A3: 去掉 BinAware 编码，改回 TD-CNN。"""

    def __init__(
        self,
        args=None,
        conc_dim=32,
        vel_dim=32,
        cnn_hidden=64,
        cnn_output=32,
        *a,
        **kw
    ):
        super().__init__(args=args, *a, **kw)

        if args is not None:
            conc_dim = getattr(args, 'conc_dim', conc_dim)
            vel_dim = getattr(args, 'vel_dim', vel_dim)
            cnn_hidden = getattr(args, 'cnn_hidden', cnn_hidden)
            cnn_output = getattr(args, 'cnn_output', cnn_output)

        self.conc_extractor = TimeDistributed1DCNN(
            input_dim=conc_dim,
            hidden_dim=cnn_hidden,
            output_dim=cnn_output
        )
        self.vel_extractor = TimeDistributed1DCNN(
            input_dim=vel_dim,
            hidden_dim=cnn_hidden,
            output_dim=cnn_output
        )


class AblateV3NoAux(_AblateV3Base):
    """A4: 去掉辅助任务。"""

    supports_aux_task = False


class AblateV3NoStreamGate(_AblateV3Base):
    """A5: 去掉三流融合门控。"""

    use_stream_gate = False
