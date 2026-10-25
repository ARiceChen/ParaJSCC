from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from models.upsampling import Upsampling

class SynthesisResidualLayer(nn.Module):
    def __init__(self, input_ft: int, output_ft: int, kernel_size: int, non_linearity: nn.Module = nn.Identity()):
        super().__init__()
        assert input_ft == output_ft
        self.pad = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        self.conv_layer = nn.Conv2d(input_ft, output_ft, kernel_size)
        self.non_linearity = non_linearity

        with torch.no_grad():
            self.conv_layer.weight.zero_()
            self.conv_layer.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self.non_linearity(self.conv_layer(self.pad(x)) + x)

class SynthesisLayer(nn.Module):
    def __init__(self, input_ft: int, output_ft: int, kernel_size: int, non_linearity: nn.Module = nn.Identity()):
        super().__init__()
        self.pad = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        self.conv_layer = nn.Conv2d(input_ft, output_ft, kernel_size)
        self.non_linearity = non_linearity

        # Keep PyTorch's fan-in initialization. Dividing each layer by
        # output_ft**2 compounded across the synthesis trunk and drove the
        # decoder output and gradients close to zero (the pilot reconstructions
        # collapsed to a near-black constant image).
        with torch.no_grad():
            self.conv_layer.bias.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self.non_linearity(self.conv_layer(self.pad(x)))

class SharedDecoder(nn.Module):
    def __init__(self, in_channels: int, hid_channel: int):
        super().__init__()
        self.conv1_1 = SynthesisLayer(in_channels, hid_channel, 1, nn.GELU())
        self.conv1_2 = SynthesisResidualLayer(hid_channel, hid_channel, 1, nn.GELU())
        self.conv1_3 = SynthesisLayer(hid_channel, hid_channel, 1, nn.GELU())
        self.conv2_1 = SynthesisResidualLayer(hid_channel, hid_channel, 3, nn.GELU())
        self.conv2_2 = SynthesisResidualLayer(hid_channel, hid_channel, 3)

    def get_param(self) -> "OrderedDict[str, Tensor]":
        return OrderedDict({k: v.detach().clone() for k, v in self.named_parameters()})

    def set_param(self, param: "OrderedDict[str, Tensor]") -> None:
        self.load_state_dict(param, strict=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1_1(x)
        out = self.conv1_2(out)
        out = self.conv1_3(out)
        out = self.conv2_1(out)
        out = self.conv2_2(out)
        return out

class ImageReconstructionHead(nn.Module):
    def __init__(self, hid_channel: int):
        super().__init__()
        self.h1 = SynthesisLayer(hid_channel, hid_channel, 1, nn.GELU())
        self.h2 = SynthesisLayer(hid_channel, 3, 1)

    def get_param(self):
        return OrderedDict({k: v.detach().clone() for k, v in self.named_parameters()})

    def set_param(self, param):
        self.load_state_dict(param, strict=True)

    def forward(self, feat):
        return self.h2(self.h1(feat))

class SignalReconstructionHead1D(nn.Module):
    def __init__(self, hid_channel, out_len, mlp_hidden=512, refine_kernel=9):
        super().__init__()
        self.out_len = int(out_len)
        # Match the empirically validated LMT-108 decoder: pool the 2D feature
        # map, map the global descriptor to the target signal, then apply a
        # small identity-initialized 1D refinement layer.
        self.mlp = nn.Sequential(
            nn.Linear(hid_channel, int(mlp_hidden)),
            nn.GELU(),
            nn.Linear(int(mlp_hidden), int(mlp_hidden)),
            nn.GELU(),
            nn.Linear(int(mlp_hidden), self.out_len),
        )

        k = int(refine_kernel)
        self.refine = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2)
        with torch.no_grad():
            self.refine.weight.zero_()
            self.refine.bias.zero_()
            self.refine.weight[:, :, k // 2] = 1.0

    def get_param(self):
        return OrderedDict({k: v.detach().clone() for k, v in self.named_parameters()})

    def set_param(self, param):
        self.load_state_dict(param, strict=True)

    def forward(self, feat):
        pooled = feat.mean(dim=(-2, -1))
        signal = self.mlp(pooled).unsqueeze(1).contiguous()
        return self.refine(signal)

class PrivateCodeDecoder(nn.Module):
    def __init__(self, code_channels, hid_channel):
        super().__init__()
        self.to_gb = SynthesisLayer(code_channels, hid_channel * 2, 1, nn.GELU())

    def get_param(self):
        return OrderedDict({k: v.detach().clone() for k, v in self.named_parameters()})

    def set_param(self, param):
        self.load_state_dict(param, strict=True)

    def forward(self, code_map):
        return self.to_gb(code_map)

class PrivateModalityAdaptor(nn.Module):
    def __init__(self, hid_channel):
        super().__init__()
        self.refine = SynthesisResidualLayer(hid_channel, hid_channel, 3, nn.GELU())

    def forward(self, feat_shared, private_feature):
        gb = private_feature
        gamma, beta = torch.chunk(gb, 2, dim=1)
        gamma = torch.tanh(gamma)
        feat = feat_shared * (1.0 + gamma) + beta
        feat = self.refine(feat)
        return feat

class ParaJSCC(nn.Module):
    """Multimodal decoder with progressive shared and private code pyramids."""

    def __init__(self, args, sparsity, in_features, out_features, hidden_features, hidden_layers):
        super().__init__()

        self.enable_code_channel_noise = True
        self.enable_param_noise = False

        self.trans_time = getattr(args, "trans_time", 1)
        self.extra_upper = getattr(args, "extra_up_trans", 0)

        self.sparsity = sparsity
        self.channel_snr = float(getattr(args, "snr", 10.0))

        self.h = int(args.patch_h)
        self.w = int(args.patch_w)

        self.progressive_upsampler = Upsampling(
            args.upsampling_kernel_size,
            args.static_upsampling_kernel,
            args.highest_flag,
        )

        self.modulation_base_number = int(args.mod_base)
        self.channel_compression = int(getattr(args, "channel_compression", 1))
        if self.channel_compression not in (1, 2):
            raise ValueError("channel_compression must be 1 or 2")

        self.sig_len = int(getattr(args, "sig_len", 576))
        self.aud_len = int(getattr(args, "aud_len", 1024))

        hid_ch = int(getattr(args, "hid_channel", 32))
        self.cm_dim = int(getattr(args, "cm_dim", 64))

        # quantization / rate configs
        self.quant_step_shared = float(getattr(args, "quant_step_shared", 0.0))
        self.quant_step_img = float(getattr(args, "quant_step_img", 0.0))
        self.quant_step_tac = float(getattr(args, "quant_step_tac", 0.0))
        self.quant_step_aud = float(getattr(args, "quant_step_aud", 0.0))

        self.rate_mode = str(getattr(args, "rate_mode", "bandwidth"))
        self.lambda_rate = float(getattr(args, "lambda_rate", 0.0))

        shared_in_ch = self.modulation_base_number
        self.D_s = SharedDecoder(in_channels=shared_in_ch, hid_channel=hid_ch)

        code_ch = self.modulation_base_number
        self.D_m_p = nn.ModuleDict({
            "img": PrivateCodeDecoder(code_channels=code_ch, hid_channel=hid_ch),
            "tac": PrivateCodeDecoder(code_channels=code_ch, hid_channel=hid_ch),
            "aud": PrivateCodeDecoder(code_channels=code_ch, hid_channel=hid_ch),
        })
        self.h_m = nn.ModuleDict({
            modality: PrivateModalityAdaptor(hid_channel=hid_ch)
            for modality in ("img", "tac", "aud")
        })
        self.H_m = nn.ModuleDict({
            "img": ImageReconstructionHead(hid_channel=hid_ch),
            "tac": SignalReconstructionHead1D(hid_channel=hid_ch, out_len=self.sig_len),
            "aud": SignalReconstructionHead1D(hid_channel=hid_ch, out_len=self.aud_len),
        })
        self.P_m = nn.ModuleDict({
            modality: nn.Linear(hid_ch, self.cm_dim)
            for modality in ("img", "tac", "aud")
        })

        scales = [2 ** i for i in range(self.modulation_base_number)]
        scales.reverse()

        fact_shape = []
        for s in scales:
            Hk = max(1, self.h // s)
            Wk = max(1, math.ceil(self.w / (s * self.channel_compression)))
            fact_shape.append((Hk, Wk))
        self.fact_shape = fact_shape

        batch_size = int(getattr(args, "batch_size", 1))

        def _make_code():
            plist = nn.ParameterList()
            for layer_idx in range(self.modulation_base_number):
                Hk, Wk = self.fact_shape[layer_idx]
                plist.append(nn.Parameter(torch.randn(batch_size, 1, Hk, Wk)))
            return plist

        self.theta_shared = _make_code()
        self.theta_img = _make_code()
        self.theta_tac = _make_code()
        self.theta_aud = _make_code()

    def _normalize_mask(self, mask, n_level: int, device: torch.device) -> List[int]:
        if mask is None:
            return [1] * n_level
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().to("cpu").view(-1).tolist()
        mask = [int(x) for x in mask]
        if len(mask) < n_level:
            mask = mask + [0] * (n_level - len(mask))
        return mask[:n_level]

    def _quantize_ste(self, x: Tensor, step: float) -> Tensor:
        if step is None or step <= 0:
            return x
        q = torch.round(x / step) * step
        return x + (q - x).detach()

    def _shared_summary(self, feat_shared: Tensor, proj: nn.Module) -> Tensor:
        pooled = feat_shared.mean(dim=(-2, -1))   # (B, C)
        emb = proj(pooled)                        # (B, D)
        emb = F.normalize(emb, dim=-1, eps=1e-8)
        return emb

    def _get_active_source_symbols(self) -> Tensor:
        """Return the full source size.

        The request selects which parameter groups are transmitted; it does not
        change the source-size normalizer used for the bandwidth ratio.
        """
        n_src = max(3 * self.h * self.w + self.sig_len + self.aud_len, 1)
        return self.theta_shared[0].new_zeros((), dtype=torch.float32) + float(n_src)

    def _tensor_channel_uses(self, x: Tensor) -> float:
        if x.dim() == 0:
            num_real = 1
        else:
            num_real = int(x[0].numel())
        k = math.ceil(num_real / 2.0)
        return float(k)

    def _channel_uses_to_rate(self, k: Tensor, n_src: Tensor) -> Tensor:
        return k / (n_src + 1e-12)

    def channel(self, inputs, input_snr):
        snr = float(input_snr)
        in_shape = inputs.shape
        B = in_shape[0]

        z = inputs.float().contiguous().view(B, -1)
        orig_len = z.shape[1]

        if orig_len % 2 != 0:
            z = torch.cat([z, torch.zeros(B, 1, device=z.device, dtype=z.dtype)], dim=1)

        dim_z = z.shape[1] // 2
        real = z[:, :dim_z]
        imag = z[:, dim_z:]

        ave_sig_pwr = (real.square() + imag.square()).mean(dim=1, keepdim=True)
        power_scale = torch.sqrt(ave_sig_pwr) + 1e-12
        real = real / power_scale
        imag = imag / power_scale

        # Unit-power normalization is part of the transmitter and must remain
        # active during deterministic scoring. The flag controls additive AWGN
        # only; bypassing the whole channel changes the decoder's input scale.
        if self.enable_code_channel_noise:
            noise_std = np.sqrt((10 ** (-snr / 10)) / 2)
            real = real + torch.randn_like(real) * noise_std
            imag = imag + torch.randn_like(imag) * noise_std

        out = torch.cat([real, imag], dim=1)

        if orig_len % 2 != 0:
            out = out[:, :-1]

        return out.view(in_shape).to(dtype=torch.float32)

    def _encode_pyramid(
        self,
        code: nn.ParameterList,
        level_mask: Optional[List[int]] = None,
        quant_step: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        n_level = len(code)
        level_mask = self._normalize_mask(level_mask, n_level, code[0].device)

        channel_out_all = []
        k_total = code[0].new_tensor(0.0)

        for idx in range(n_level):
            z = code[idx]

            if level_mask[idx] == 0:
                channel_out_all.append(torch.zeros_like(z))
                k_level = 0.0
            else:
                z_used = self._quantize_ste(z, quant_step)
                k_level = self._tensor_channel_uses(z_used)
                channel_out_all.append(self.channel(z_used, self.channel_snr))

            if k_level > 0:
                k_total = k_total + z.new_tensor(k_level)

        q_upsample = self.progressive_upsampler(channel_out_all)
        if self.channel_compression > 1:
            q_upsample = F.interpolate(
                q_upsample, size=(self.h, self.w), mode="bilinear", align_corners=False
            )
        return q_upsample, k_total

    def _encode_joint_pyramids(
        self,
        codes: List[nn.ParameterList],
        masks: List[Optional[List[int]]],
        quant_steps: List[float],
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Batch independent code groups through the channel and upsampler."""
        n_groups = len(codes)
        n_level = len(codes[0])
        batch_size = codes[0][0].shape[0]
        masks = [
            self._normalize_mask(mask, n_level, code[0].device)
            for code, mask in zip(codes, masks)
        ]
        channel_levels = []
        channel_uses = [0.0] * n_groups

        for level in range(n_level):
            level_codes = []
            for group in range(n_groups):
                code = codes[group][level]
                if masks[group][level]:
                    level_codes.append(self._quantize_ste(code, quant_steps[group]))
                    channel_uses[group] += self._tensor_channel_uses(code)
                else:
                    level_codes.append(torch.zeros_like(code))
            stacked_codes = torch.cat(level_codes, dim=0)
            transmitted = self.channel(stacked_codes, self.channel_snr)
            if all(masks[group][level] for group in range(n_groups)):
                channel_levels.append(transmitted)
            else:
                active = stacked_codes.new_tensor(
                    [masks[group][level] for group in range(n_groups)]
                ).repeat_interleave(batch_size).view(-1, 1, 1, 1)
                channel_levels.append(transmitted * active)

        dense_codes = self.progressive_upsampler(channel_levels)
        if self.channel_compression > 1:
            dense_codes = F.interpolate(
                dense_codes, size=(self.h, self.w), mode="bilinear", align_corners=False
            )
        code_maps = list(dense_codes.split(batch_size, dim=0))
        uses = [codes[group][0].new_zeros(()) + channel_uses[group] for group in range(n_groups)]
        return code_maps, uses

    def _decode_shared_feat(
        self,
        level_mask_shared: Optional[List[int]] = None,
        quant_step: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        q_shared, k_shared = self._encode_pyramid(
            self.theta_shared,
            level_mask=level_mask_shared,
            quant_step=quant_step,
        )
        feat_shared = self.D_s(q_shared)
        return feat_shared, k_shared

    def _decode_code_map(
        self,
        code_priv: nn.ParameterList,
        level_mask: Optional[List[int]] = None,
        quant_step: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        q_code, k_code = self._encode_pyramid(
            code_priv,
            level_mask=level_mask,
            quant_step=quant_step,
        )
        return q_code, k_code

    def forward(
        self,
        level_mask_shared: Optional[List[int]] = None,
        level_mask_img: Optional[List[int]] = None,
        level_mask_tac: Optional[List[int]] = None,
        level_mask_aud: Optional[List[int]] = None,
        transmit_img: bool = True,
        transmit_tac: bool = True,
        transmit_aud: bool = True,
        quantize: bool = False,
        return_rate: bool = False,
        return_cm_embed: bool = False,
        rate_mode: Optional[str] = None,
    ):

        _ = rate_mode

        q_shared_step = self.quant_step_shared if quantize else 0.0
        q_img_step = self.quant_step_img if quantize else 0.0
        q_tac_step = self.quant_step_tac if quantize else 0.0
        q_aud_step = self.quant_step_aud if quantize else 0.0

        code_img = code_tac = code_aud = None
        if transmit_img and transmit_tac and transmit_aud:
            code_maps, channel_uses = self._encode_joint_pyramids(
                codes=[self.theta_shared, self.theta_img, self.theta_tac, self.theta_aud],
                masks=[level_mask_shared, level_mask_img, level_mask_tac, level_mask_aud],
                quant_steps=[q_shared_step, q_img_step, q_tac_step, q_aud_step],
            )
            feat_shared = self.D_s(code_maps[0])
            code_img, code_tac, code_aud = code_maps[1:]
            k_shared, k_img, k_tac, k_aud = channel_uses
        else:
            feat_shared, k_shared = self._decode_shared_feat(
                level_mask_shared=level_mask_shared,
                quant_step=q_shared_step,
            )
            k_img = feat_shared.new_tensor(0.0)
            k_tac = feat_shared.new_tensor(0.0)
            k_aud = feat_shared.new_tensor(0.0)

        B = feat_shared.shape[0]
        n_src = self._get_active_source_symbols()

        pred_img = None
        pred_tac = None
        pred_aud = None

        cm_embed = None
        if return_cm_embed:
            cm_embed = {
                "img": self._shared_summary(feat_shared, self.P_m["img"]) if transmit_img else None,
                "tac": self._shared_summary(feat_shared, self.P_m["tac"]) if transmit_tac else None,
                "aud": self._shared_summary(feat_shared, self.P_m["aud"]) if transmit_aud else None,
            }

        if transmit_img:
            if code_img is None:
                code_img, k_img = self._decode_code_map(
                    self.theta_img,
                    level_mask=level_mask_img,
                    quant_step=q_img_step,
                )
            private_img = self.D_m_p["img"](code_img)
            feat_img = self.h_m["img"](feat_shared, private_img)
            img = self.H_m["img"](feat_img)
            pred_img = img.view(B, 3, -1).permute(0, 2, 1).contiguous()

        if transmit_tac:
            if code_tac is None:
                code_tac, k_tac = self._decode_code_map(
                    self.theta_tac,
                    level_mask=level_mask_tac,
                    quant_step=q_tac_step,
                )
            private_tac = self.D_m_p["tac"](code_tac)
            feat_tac = self.h_m["tac"](feat_shared, private_tac)
            tac_1d = self.H_m["tac"](feat_tac)
            pred_tac = tac_1d.permute(0, 2, 1).contiguous()

        if transmit_aud:
            if code_aud is None:
                code_aud, k_aud = self._decode_code_map(
                    self.theta_aud,
                    level_mask=level_mask_aud,
                    quant_step=q_aud_step,
                )
            private_aud = self.D_m_p["aud"](code_aud)
            feat_aud = self.h_m["aud"](feat_shared, private_aud)
            aud_1d = self.H_m["aud"](feat_aud)
            pred_aud = aud_1d.permute(0, 2, 1).contiguous()

        k_total = k_shared + k_img + k_tac + k_aud

        r_shared = self._channel_uses_to_rate(k_shared, n_src)
        r_img = self._channel_uses_to_rate(k_img, n_src)
        r_tac = self._channel_uses_to_rate(k_tac, n_src)
        r_aud = self._channel_uses_to_rate(k_aud, n_src)
        r_total = self._channel_uses_to_rate(k_total, n_src)

        if return_rate and return_cm_embed:
            rate_pack = {
                "total": r_total,
                "shared": r_shared,
                "img": r_img,
                "tac": r_tac,
                "aud": r_aud,
                "k_total": k_total,
                "k_shared": k_shared,
                "k_img": k_img,
                "k_tac": k_tac,
                "k_aud": k_aud,
                "n_src": n_src,
            }
            return pred_img, pred_tac, pred_aud, rate_pack, cm_embed

        if return_rate:
            rate_pack = {
                "total": r_total,
                "shared": r_shared,
                "img": r_img,
                "tac": r_tac,
                "aud": r_aud,
                "k_total": k_total,
                "k_shared": k_shared,
                "k_img": k_img,
                "k_tac": k_tac,
                "k_aud": k_aud,
                "n_src": n_src,
            }
            return pred_img, pred_tac, pred_aud, rate_pack

        if return_cm_embed:
            return pred_img, pred_tac, pred_aud, cm_embed

        return pred_img, pred_tac, pred_aud
