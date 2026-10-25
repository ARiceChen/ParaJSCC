import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


# ============================================================
# Small utils
# ============================================================

def _safe_makedirs_for_file(file_path: str) -> None:
    d = os.path.dirname(file_path)
    if d:
        os.makedirs(d, exist_ok=True)

def split_model_parameters(model):
    latent_params = []

    latent_params += list(getattr(model, "modulation_sf", []))

    latent_params += list(getattr(model, "theta_shared", []))
    latent_params += list(getattr(model, "theta_img", []))
    latent_params += list(getattr(model, "theta_tac", []))
    latent_params += list(getattr(model, "theta_aud", []))

    latent_ids = {id(p) for p in latent_params}
    base_params = [p for _, p in model.named_parameters() if id(p) not in latent_ids]
    return latent_params, base_params

def _try_prepare_for_loading(model) -> None:
    try:
        if getattr(model, "enable_param_noise", False) and hasattr(model, "register_param_noise"):
            model.register_param_noise(1.0, 1.0, force_reset=True)
    except Exception:
        pass

def _clone_state_dict_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}

def _zero_like(ref: torch.Tensor) -> torch.Tensor:
    return ref.new_zeros(())

def _scalar_to_float(x, default: float = 0.0) -> float:
    if x is None:
        return float(default)
    try:
        if isinstance(x, (float, int)):
            return float(x)
        if torch.is_tensor(x):
            return float(x.detach().item())
        return float(x)
    except Exception:
        return float(default)

def _rate_pack_ref(rate_pack: Optional[Dict[str, Any]], ref: torch.Tensor) -> Dict[str, torch.Tensor]:
    if rate_pack is None:
        rate_pack = {}

    def _get_tensor(key: str) -> torch.Tensor:
        v = rate_pack.get(key, None)
        if v is None:
            return _zero_like(ref)
        if torch.is_tensor(v):
            return v
        try:
            return ref.new_tensor(float(v))
        except Exception:
            return _zero_like(ref)

    return {
        "total": _get_tensor("total"),
        "shared": _get_tensor("shared"),
        "img": _get_tensor("img"),
        "tac": _get_tensor("tac"),
        "aud": _get_tensor("aud"),
        "k_total": _get_tensor("k_total"),
        "k_shared": _get_tensor("k_shared"),
        "k_img": _get_tensor("k_img"),
        "k_tac": _get_tensor("k_tac"),
        "k_aud": _get_tensor("k_aud"),
        "n_src": _get_tensor("n_src"),
    }

# ============================================================
# Resume payload
# ============================================================

@dataclass
class ResumeState:
    start_step: int = 0
    best_score: float = 1e18
    best_state: Optional[Dict[str, torch.Tensor]] = None

    resumed: bool = False

def _load_resume(
    saved_path: str,
    model,
    optimizer,
    scheduler,
    device: torch.device,
) -> ResumeState:
    if not os.path.exists(saved_path):
        return ResumeState()

    try:
        ckpt = torch.load(saved_path, map_location=device)
    except Exception:
        return ResumeState()

    _try_prepare_for_loading(model)

    if isinstance(ckpt, dict) and "resume" in ckpt:
        resume = ckpt.get("resume", {}) or {}

        model_sd = ckpt.get("model_state_dict", None)
        if model_sd is not None:
            model.load_state_dict(model_sd, strict=True)

        opt_sd = ckpt.get("optimizer_state_dict", None)
        sch_sd = ckpt.get("scheduler_state_dict", None)
        if opt_sd is not None:
            optimizer.load_state_dict(opt_sd)
        if sch_sd is not None:
            scheduler.load_state_dict(sch_sd)

        return ResumeState(
            start_step=int(resume.get("step", 0)),
            best_score=float(resume.get("best_score", 1e18)),
            best_state=ckpt.get("best_model_state_dict", None),

            resumed=True,
        )

    best_state = ckpt.get("model_state_dict", None) if isinstance(ckpt, dict) else None
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    return ResumeState(start_step=0, best_state=best_state, resumed=False)

def _save_resume(
    saved_path: str,
    model,
    optimizer,
    scheduler,
    step: int,
    best_score: float,
    best_state: Optional[Dict[str, torch.Tensor]],
) -> None:
    _safe_makedirs_for_file(saved_path)

    payload = {
        "model_state_dict": _clone_state_dict_cpu(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_model_state_dict": best_state,
        "resume": {
            "step": int(step),
            "best_score": float(best_score),
        },
    }

    tmp = saved_path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, saved_path)

def _save_best_only(saved_path: str, best_state: Optional[Dict[str, torch.Tensor]]) -> None:
    if best_state is None:
        return
    _safe_makedirs_for_file(saved_path)
    tmp = saved_path + ".besttmp"
    torch.save({"model_state_dict": best_state}, tmp)
    os.replace(tmp, saved_path)

# ============================================================
# Core helpers
# ============================================================

def _prepare_targets(
    img_tensor: torch.Tensor,
    tac_tensor: torch.Tensor,
    aud_tensor: torch.Tensor,
    tac_len_1d: int = 576,
    aud_len_1d: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    b, _, h, w = img_tensor.shape
    target_img = img_tensor.permute(0, 2, 3, 1).contiguous().view(b, -1, 3)

    if tac_tensor.ndim == 2:
        if tac_tensor.shape[1] != int(tac_len_1d):
            raise ValueError(
                f"tactile 1D shape mismatch, expect (B,{tac_len_1d}), got {tuple(tac_tensor.shape)}"
            )
        tac_len = int(tac_len_1d)
        target_tac = tac_tensor.unsqueeze(2).contiguous()
    elif tac_tensor.ndim == 3:
        if tac_tensor.shape[1] == 1 and tac_tensor.shape[2] == int(tac_len_1d):
            target_tac = tac_tensor.permute(0, 2, 1).contiguous()
        elif tac_tensor.shape[1] == int(tac_len_1d) and tac_tensor.shape[2] == 1:
            target_tac = tac_tensor.contiguous()
        else:
            raise ValueError(
                f"tactile 1D shape mismatch, expect (B,1,{tac_len_1d}) or (B,{tac_len_1d},1), got {tuple(tac_tensor.shape)}"
            )
        tac_len = int(tac_len_1d)
    else:
        raise ValueError(f"Unsupported tac_tensor ndim={tac_tensor.ndim}, shape={tuple(tac_tensor.shape)}")

    if aud_tensor.ndim == 2:
        if aud_tensor.shape[1] != int(aud_len_1d):
            raise ValueError(
                f"audio 1D shape mismatch, expect (B,{aud_len_1d}), got {tuple(aud_tensor.shape)}"
            )
        aud_len = int(aud_len_1d)
        target_aud = aud_tensor.unsqueeze(2).contiguous()
        return target_img, target_tac, target_aud, tac_len, aud_len

    if aud_tensor.ndim == 3:
        if aud_tensor.shape[1] == 1 and aud_tensor.shape[2] == int(aud_len_1d):
            target_aud = aud_tensor.permute(0, 2, 1).contiguous()
        elif aud_tensor.shape[1] == int(aud_len_1d) and aud_tensor.shape[2] == 1:
            target_aud = aud_tensor.contiguous()
        else:
            raise ValueError(
                f"audio 1D shape mismatch, expect (B,1,{aud_len_1d}) or (B,{aud_len_1d},1) or (B,{aud_len_1d}), got {tuple(aud_tensor.shape)}"
            )
        aud_len = int(aud_len_1d)
        return target_img, target_tac, target_aud, tac_len, aud_len

    raise ValueError(f"Unsupported aud_tensor ndim={aud_tensor.ndim}, shape={tuple(aud_tensor.shape)}")

def _validate_pred_shapes(
    pred_img: Optional[torch.Tensor],
    pred_tac: Optional[torch.Tensor],
    pred_aud: Optional[torch.Tensor],
    target_img: torch.Tensor,
    tac_len: int,
    aud_len: int,
) -> None:
    if pred_img is not None and pred_img.shape != target_img.shape:
        raise RuntimeError(f"Image shape mismatch: pred={tuple(pred_img.shape)} target={tuple(target_img.shape)}")

    if pred_tac is not None:
        if pred_tac.ndim != 3 or pred_tac.shape[1] != tac_len or pred_tac.shape[2] != 1:
            raise RuntimeError(f"Tactile shape mismatch: pred={tuple(pred_tac.shape)} expect (B,{tac_len},1)")

    if pred_aud is not None:
        if pred_aud.ndim != 3 or pred_aud.shape[1] != aud_len or pred_aud.shape[2] != 1:
            raise RuntimeError(f"Audio shape mismatch: pred={tuple(pred_aud.shape)} expect (B,{aud_len},1)")

@dataclass
class LossPack:
    total: torch.Tensor
    img: torch.Tensor
    tac: torch.Tensor
    aud: torch.Tensor
    stft: torch.Tensor
    rate: torch.Tensor
    cm: torch.Tensor

def _audio_logmag(
    audio: torch.Tensor,
    window: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 128,
    win_length: int = 512,
) -> torch.Tensor:
    waveform = audio.squeeze(2)
    waveform = waveform - waveform.mean(dim=1, keepdim=True)
    spectrum = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    return torch.log1p(spectrum.abs())


def _compute_stft_logmag_l1(
    pred_aud: torch.Tensor,
    target_aud: torch.Tensor,
    window: torch.Tensor,
    target_logmag: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if target_logmag is None:
        target_logmag = _audio_logmag(target_aud, window)
    return F.l1_loss(_audio_logmag(pred_aud, window), target_logmag)

def _compute_cm_loss(
    cm_embed: Optional[Dict[str, Optional[torch.Tensor]]],
    ref: torch.Tensor,
) -> torch.Tensor:
    """
    Shared-feature embedding consistency loss.
    cm_embed:
        {
            "img": (B, D) or None,
            "tac": (B, D) or None,
            "aud": (B, D) or None,
        }
    """
    if cm_embed is None:
        return ref.new_zeros(())

    vecs = [
        cm_embed.get("img", None),
        cm_embed.get("tac", None),
        cm_embed.get("aud", None),
    ]
    vecs = [v for v in vecs if v is not None]

    if len(vecs) < 2:
        return ref.new_zeros(())

    loss = ref.new_zeros(())
    cnt = 0
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            loss = loss + F.mse_loss(vecs[i], vecs[j])
            cnt += 1

    return loss / max(cnt, 1)

def _compute_losses(
    criterion: nn.Module,
    pred_img: Optional[torch.Tensor],
    pred_tac: Optional[torch.Tensor],
    pred_aud: Optional[torch.Tensor],
    target_img: torch.Tensor,
    target_tac: torch.Tensor,
    target_aud: torch.Tensor,
    stft_window: torch.Tensor,
    target_stft_logmag: Optional[torch.Tensor],
    lambda_img: float,
    lambda_tac: float,
    lambda_aud: float,
    lambda_stft: float,
    rate_term: Optional[torch.Tensor] = None,
    lambda_rate: float = 0.0,
    cm_term: Optional[torch.Tensor] = None,
    lambda_cm: float = 0.1,
) -> LossPack:
    ref = target_img

    if pred_img is not None:
        loss_img = criterion(pred_img, target_img)
    else:
        loss_img = _zero_like(ref)

    if pred_tac is not None:
        loss_tac = criterion(pred_tac, target_tac)
    else:
        loss_tac = _zero_like(ref)

    if pred_aud is not None:
        loss_aud = criterion(pred_aud, target_aud)
        loss_stft = _compute_stft_logmag_l1(
            pred_aud, target_aud, window=stft_window, target_logmag=target_stft_logmag
        )
    else:
        loss_aud = _zero_like(ref)
        loss_stft = _zero_like(ref)

    if rate_term is None:
        rate_term = _zero_like(ref)

    if cm_term is None:
        cm_term = _zero_like(ref)

    total = (
        lambda_img * loss_img
        + lambda_tac * loss_tac
        + lambda_aud * loss_aud
        + lambda_stft * loss_stft
        + lambda_rate * rate_term
        + lambda_cm * cm_term
    )

    return LossPack(
        total=total,
        img=loss_img,
        tac=loss_tac,
        aud=loss_aud,
        stft=loss_stft,
        rate=rate_term,
        cm=cm_term,
    )

def _tb_log_step(
    writer,
    gstep: int,
    prefix: str,
    losses: LossPack,
    lr_val: float,
    tb_prefix_loss: str,
    rate_pack: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    if writer is None:
        return

    try:
        lroot = f"{prefix}{tb_prefix_loss}/"
        writer.add_scalar(lroot + "Total", float(losses.total.item()), gstep)
        writer.add_scalar(lroot + "Image_MSE", float(losses.img.item()), gstep)
        writer.add_scalar(lroot + "Tactile_MSE", float(losses.tac.item()), gstep)
        writer.add_scalar(lroot + "Audio_MSE", float(losses.aud.item()), gstep)
        writer.add_scalar(lroot + "Audio_STFT_LogMag_L1", float(losses.stft.item()), gstep)
        writer.add_scalar(lroot + "Rate", float(losses.rate.item()), gstep)
        writer.add_scalar(lroot + "CrossModal", float(losses.cm.item()), gstep)

        if rate_pack is not None:
            rroot = f"{prefix}RateDetail/"
            writer.add_scalar(rroot + "R_total", float(rate_pack["total"].item()), gstep)
            writer.add_scalar(rroot + "R_shared", float(rate_pack["shared"].item()), gstep)
            writer.add_scalar(rroot + "R_img", float(rate_pack["img"].item()), gstep)
            writer.add_scalar(rroot + "R_tac", float(rate_pack["tac"].item()), gstep)
            writer.add_scalar(rroot + "R_aud", float(rate_pack["aud"].item()), gstep)

            writer.add_scalar(rroot + "K_total", float(rate_pack["k_total"].item()), gstep)
            writer.add_scalar(rroot + "K_shared", float(rate_pack["k_shared"].item()), gstep)
            writer.add_scalar(rroot + "K_img", float(rate_pack["k_img"].item()), gstep)
            writer.add_scalar(rroot + "K_tac", float(rate_pack["k_tac"].item()), gstep)
            writer.add_scalar(rroot + "K_aud", float(rate_pack["k_aud"].item()), gstep)
            writer.add_scalar(rroot + "N_src", float(rate_pack["n_src"].item()), gstep)

        writer.add_scalar(f"{prefix}LR/Main", float(lr_val), gstep)
    except Exception:
        pass

def _maybe_flush(writer, step: int, steps_til_summary: int, save_every: int) -> None:
    if writer is None:
        return
    if (steps_til_summary > 0 and step % steps_til_summary == 0) or (save_every > 0 and step % save_every == 0):
        try:
            writer.flush()
        except Exception:
            pass

def _sample_level_mask(n_level: int, min_active: int = 1):
    k = np.random.randint(min_active, n_level + 1)
    return [1 if i < k else 0 for i in range(n_level)]

def _make_progressive_cfg(model, args):
    """Sample a random modality request for this training iteration."""
    L = int(getattr(model, "modulation_base_number", 1))

    fixed_shared = getattr(args, "level_mask_shared", None)
    fixed_img = getattr(args, "level_mask_img", None)
    fixed_tac = getattr(args, "level_mask_tac", None)
    fixed_aud = getattr(args, "level_mask_aud", None)

    fixed_tx_img = bool(getattr(args, "transmit_img", True))
    fixed_tx_tac = bool(getattr(args, "transmit_tac", True))
    fixed_tx_aud = bool(getattr(args, "transmit_aud", True))

    use_prog = bool(getattr(args, "apply_progressive_in_forward", True))
    p = float(getattr(args, "train_progressive_prob", 1.0))

    if (not use_prog) or (np.random.rand() > p):
        return {
            "level_mask_shared": fixed_shared if fixed_shared is not None else [1] * L,
            "level_mask_img": fixed_img if fixed_img is not None else [1] * L,
            "level_mask_tac": fixed_tac if fixed_tac is not None else [1] * L,
            "level_mask_aud": fixed_aud if fixed_aud is not None else [1] * L,
            "transmit_img": fixed_tx_img,
            "transmit_tac": fixed_tx_tac,
            "transmit_aud": fixed_tx_aud,
        }

    # Sample uniformly from all non-empty modality subsets.
    request_id = int(np.random.randint(1, 8))
    transmit_img = bool(request_id & 0b001)
    transmit_tac = bool(request_id & 0b010)
    transmit_aud = bool(request_id & 0b100)

    # The same progressive level l applies to shared and requested private codes.
    active_levels = int(np.random.randint(1, L + 1))
    prefix = [1 if i < active_levels else 0 for i in range(L)]

    return {
        "level_mask_shared": prefix,
        "level_mask_img": prefix if transmit_img else [0] * L,
        "level_mask_tac": prefix if transmit_tac else [0] * L,
        "level_mask_aud": prefix if transmit_aud else [0] * L,
        "transmit_img": transmit_img,
        "transmit_tac": transmit_tac,
        "transmit_aud": transmit_aud,
    }

# ============================================================
# Main API
# ============================================================

def train_one_image(
    model,
    img_tensor,
    tac_tensor,
    aud_tensor,
    total_steps: int,
    steps_til_summary: int,
    lr: float,
    saved_path: str,
    args=None,
    lambda_img: float = 1.0,
    lambda_tac: float = 1.0,
    lambda_aud: float = 1.0,
    lambda_stft: float = 0.1,
    tac_len_1d: int = 576,
    aud_len_1d: int = 1024,
    aud_fs: int = 16000,
    save_every: int = 200,
    resume: bool = True,
    writer=None,
    tb_tag: str = "",
    tb_every: int = 10,
    global_step_offset: int = 0,
    tb_prefix_loss: str = "Loss",
    optimize_network: bool = True,
    checkpoint_interval: int = 100,
    scheduler_step_size: int = 100,
    optimizer=None,
    scheduler=None,
    write_checkpoint: bool = True,
    release_cuda_cache: bool = False,
    latent_step_fn=None,
    cuda_graph: bool = False,
):
    device = next(model.parameters()).device
    criterion = nn.MSELoss().to(device)
    total_steps = min(max(int(total_steps), 1), 2000)
    checkpoint_interval = max(int(checkpoint_interval), 1)
    if total_steps < 1:
        raise ValueError(f"total_steps must be at least 1, got {total_steps}")

    latent_params, base_params = split_model_parameters(model)
    if optimize_network:
        optimizer_params = [{"params": base_params}, {"params": latent_params}]
    else:
        for param in base_params:
            param.requires_grad_(False)
        optimizer_params = [{"params": latent_params}]
    if optimizer is None:
        if optimize_network:
            # Optimize this sample's network parameters and parameter groups.
            optimizer = torch.optim.AdamW(optimizer_params, lr=lr)
        else:
            # Offline package-code fitting uses Adam without weight decay,
            # matching the previous LMT-108 implementation.
            optimizer = torch.optim.Adam(optimizer_params, lr=lr)
    if scheduler is None:
        if optimize_network:
            scheduler = StepLR(optimizer, step_size=max(int(scheduler_step_size), 1), gamma=0.5)
        else:
            # Anneal the per-sample code learning rate smoothly.
            scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0.0)

    model.train()

    target_img, target_tac, target_aud, tac_len_used, aud_len_used = _prepare_targets(
        img_tensor=img_tensor,
        tac_tensor=tac_tensor,
        aud_tensor=aud_tensor,
        tac_len_1d=int(tac_len_1d),
        aud_len_1d=int(aud_len_1d),
    )

    win_length = 512
    stft_window = torch.hann_window(win_length, device=device)
    with torch.no_grad():
        target_stft_logmag = _audio_logmag(target_aud, stft_window)

    best_score = 1e18
    best_state = None

    start_step = 0

    if resume and write_checkpoint:
        rs = _load_resume(saved_path, model, optimizer, scheduler, device)
        start_step = int(rs.start_step)
        best_score = float(rs.best_score)
        best_state = rs.best_state

    prefix = (tb_tag.strip() + "/") if tb_tag else ""

    lambda_rate = float(getattr(model, "lambda_rate", 0.0))
    rate_mode = str(getattr(model, "rate_mode", "bandwidth"))
    lambda_cm = float(getattr(args, "lambda_cm", 0.1)) if args is not None else 0.1

    def forward_losses(prog_cfg):
        out = model(
            level_mask_shared=prog_cfg["level_mask_shared"],
            level_mask_img=prog_cfg["level_mask_img"],
            level_mask_tac=prog_cfg["level_mask_tac"],
            level_mask_aud=prog_cfg["level_mask_aud"],
            transmit_img=prog_cfg["transmit_img"],
            transmit_tac=prog_cfg["transmit_tac"],
            transmit_aud=prog_cfg["transmit_aud"],
            quantize=True,
            return_rate=True,
            return_cm_embed=True,
            rate_mode=rate_mode,
        )

        cm_embed = None
        if isinstance(out, (tuple, list)) and len(out) == 5:
            pred_img, pred_tac, pred_aud, rate_pack_raw, cm_embed = out
        elif isinstance(out, (tuple, list)) and len(out) == 4:
            pred_img, pred_tac, pred_aud, rate_pack_raw = out
        else:
            pred_img, pred_tac, pred_aud = out
            rate_pack_raw = {"total": _zero_like(target_img)}

        rate_pack = _rate_pack_ref(rate_pack_raw, ref=target_img)

        _validate_pred_shapes(pred_img, pred_tac, pred_aud, target_img, tac_len_used, aud_len_used)

        loss_img_w = float(lambda_img) if pred_img is not None else 0.0
        loss_tac_w = float(lambda_tac) if pred_tac is not None else 0.0
        loss_aud_w = float(lambda_aud) if pred_aud is not None else 0.0
        loss_stft_w = float(lambda_stft) if pred_aud is not None else 0.0

        cm_term = _compute_cm_loss(cm_embed, ref=target_img)

        losses = _compute_losses(
            criterion=criterion,
            pred_img=pred_img,
            pred_tac=pred_tac,
            pred_aud=pred_aud,
            target_img=target_img,
            target_tac=target_tac,
            target_aud=target_aud,
            stft_window=stft_window,
            target_stft_logmag=target_stft_logmag,
            lambda_img=loss_img_w,
            lambda_tac=loss_tac_w,
            lambda_aud=loss_aud_w,
            lambda_stft=loss_stft_w,
            rate_term=rate_pack["total"],
            lambda_rate=lambda_rate,
            cm_term=cm_term,
            lambda_cm=lambda_cm,
        )
        return losses, rate_pack

    training_graph = None
    if cuda_graph:
        if device.type != "cuda" or latent_step_fn is not None:
            raise ValueError("CUDA graph fitting requires CUDA and no latent step callback.")
        if args is not None and float(getattr(args, "train_progressive_prob", 0.0)) != 0.0:
            raise ValueError("CUDA graph fitting requires a fixed training request.")
        fixed_cfg = _make_progressive_cfg(model, args)
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                model.zero_grad(set_to_none=False)
                warmup_losses, _ = forward_losses(fixed_cfg)
                warmup_losses.total.backward()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        grad_tensors = [param.grad for param in model.parameters() if param.grad is not None]
        training_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(training_graph):
            torch._foreach_zero_(grad_tensors)
            graph_losses, graph_rate_pack = forward_losses(fixed_cfg)
            graph_losses.total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)

    for step in range(int(start_step), total_steps):
        if training_graph is None:
            prog_cfg = _make_progressive_cfg(model, args)
            losses, rate_pack = forward_losses(prog_cfg)
        else:
            training_graph.replay()
            losses, rate_pack = graph_losses, graph_rate_pack

        score_candidate = (
            step == 0 or (step + 1) % checkpoint_interval == 0 or step == total_steps - 1
        )

        if score_candidate:
            # Score the complete training request periodically to retain the best checkpoint.
            was_training = model.training
            model.eval()
            with torch.no_grad():
                full_out = model(
                    level_mask_shared=[1] * int(getattr(model, "modulation_base_number", 1)),
                    level_mask_img=[1] * int(getattr(model, "modulation_base_number", 1)),
                    level_mask_tac=[1] * int(getattr(model, "modulation_base_number", 1)),
                    level_mask_aud=[1] * int(getattr(model, "modulation_base_number", 1)),
                    transmit_img=True,
                    transmit_tac=True,
                    transmit_aud=True,
                    quantize=True,
                    return_rate=True,
                    return_cm_embed=True,
                    rate_mode=rate_mode,
                )
                if isinstance(full_out, (tuple, list)) and len(full_out) == 5:
                    full_img, full_tac, full_aud, full_rate_raw, full_cm = full_out
                elif isinstance(full_out, (tuple, list)) and len(full_out) == 4:
                    full_img, full_tac, full_aud, full_rate_raw = full_out
                    full_cm = None
                else:
                    full_img, full_tac, full_aud = full_out
                    full_rate_raw, full_cm = {"total": _zero_like(target_img)}, None
                full_rate = _rate_pack_ref(full_rate_raw, ref=target_img)
                full_losses = _compute_losses(
                    criterion=criterion,
                    pred_img=full_img,
                    pred_tac=full_tac,
                    pred_aud=full_aud,
                    target_img=target_img,
                    target_tac=target_tac,
                    target_aud=target_aud,
                    stft_window=stft_window,
                    target_stft_logmag=target_stft_logmag,
                    lambda_img=float(lambda_img),
                    lambda_tac=float(lambda_tac),
                    lambda_aud=float(lambda_aud),
                    lambda_stft=float(lambda_stft),
                    rate_term=full_rate["total"],
                    lambda_rate=lambda_rate,
                    cm_term=_compute_cm_loss(full_cm, ref=target_img),
                    lambda_cm=lambda_cm,
                )
            model.train(was_training)

            cur_total = float(full_losses.total.item())
            if cur_total < best_score:
                best_score = cur_total
                best_state = _clone_state_dict_cpu(model.state_dict())

        if writer is not None and tb_every > 0 and (step % tb_every == 0):
            gstep = int(global_step_offset) + int(step)
            lr_val = float(optimizer.param_groups[0]["lr"])
            _tb_log_step(
                writer=writer,
                gstep=gstep,
                prefix=prefix,
                losses=losses,
                lr_val=lr_val,
                tb_prefix_loss=tb_prefix_loss,
                rate_pack=rate_pack,
            )

        if steps_til_summary > 0 and (step % steps_til_summary == 0):
            print(
                f"Step {step} | "
                f"Loss={losses.total.item():.4e} | "
                f"R_total={rate_pack['total'].item():.4e} | "
                f"R_sh={rate_pack['shared'].item():.4e} | "
                f"R_img={rate_pack['img'].item():.4e} | "
                f"R_tac={rate_pack['tac'].item():.4e} | "
                f"R_aud={rate_pack['aud'].item():.4e} | "
                f"K_total={rate_pack['k_total'].item():.4e} | "
                f"N_src={rate_pack['n_src'].item():.4e} | "
                f"CM={losses.cm.item():.4e} | "
                f"TacLen={tac_len_used} AudLen={aud_len_used}"
            )

        # Clear gradients for the current sample model before its joint update.
        if training_graph is None:
            model.zero_grad(set_to_none=True)
            losses.total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if latent_step_fn is not None:
            latent_step_fn()
        scheduler.step()

        if write_checkpoint and save_every > 0 and (step % save_every == 0) and (step != total_steps - 1):
            _save_resume(
                saved_path=saved_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=int(step + 1),
                best_score=best_score,
                best_state=best_state,
            )

        _maybe_flush(writer, step, steps_til_summary, save_every)

    if write_checkpoint:
        _save_resume(
            saved_path=saved_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=int(total_steps),
            best_score=best_score,
            best_state=best_state,
        )
        _save_best_only(saved_path + ".best.pth", best_state)

    if writer is not None:
        try:
            writer.flush()
        except Exception:
            pass

    if release_cuda_cache:
        torch.cuda.empty_cache()
    return float(best_score)
