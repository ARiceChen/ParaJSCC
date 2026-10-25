"""Serialization helpers for sample-specific ParaJSCC model parameters."""

from pathlib import Path
from typing import Any, Dict
import copy
import zlib

import torch

_GROUPS = ("shared", "img", "tac", "aud")
_STATE_PREFIX = {
    "shared": "theta_shared.",
    "img": "theta_img.",
    "tac": "theta_tac.",
    "aud": "theta_aud.",
}
_MODEL_ATTR = {
    "shared": "theta_shared",
    "img": "theta_img",
    "tac": "theta_tac",
    "aud": "theta_aud",
}
_FORMAT = "ParaJSCC-sample-parameters-v2"
_ARCHITECTURE = "sample_network_v2"


def sample_package_path(sample_checkpoint: Path, args) -> Path:
    """Return a cache path for one sample's complete fitted parameter package."""
    steps = (
        float(getattr(args, "quant_step_shared", 0.0)),
        float(getattr(args, "quant_step_img", 0.0)),
        float(getattr(args, "quant_step_tac", 0.0)),
        float(getattr(args, "quant_step_aud", 0.0)),
    )
    tag = "_".join(f"{step:g}" for step in steps)
    model_tag = (
        f"L{int(getattr(args, 'mod_base', 4))}_"
        f"hid{int(getattr(args, 'hid_channel', 32))}_"
        f"H{int(getattr(args, 'patch_h', 0))}_W{int(getattr(args, 'patch_w', 0))}_"
        f"snr{float(getattr(args, 'snr', 10.0)):g}"
    )
    if int(getattr(args, "channel_compression", 1)) > 1:
        model_tag += f"_c{int(args.channel_compression)}"
    checkpoint = Path(sample_checkpoint)
    return (
        checkpoint.parent
        / "parameter_packages"
        / _ARCHITECTURE
        / model_tag
        / f"q_{tag}"
        / f"{checkpoint.stem}.package.pth"
    )


def _fit_signature(args, steps: int) -> Dict[str, Any]:
    return {
        "protocol": _ARCHITECTURE,
        "dataset": str(getattr(args, "dataset", "LMT-108")),
        "snr": float(getattr(args, "snr", 10.0)),
        "hid_channel": int(getattr(args, "hid_channel", 32)),
        "progressive_levels": int(getattr(args, "mod_base", 4)),
        "channel_compression": int(getattr(args, "channel_compression", 1)),
        "patch_height": int(getattr(args, "patch_h", 0)),
        "patch_width": int(getattr(args, "patch_w", 0)),
        "tactile_length": int(getattr(args, "sig_len", 576)),
        "audio_length": int(getattr(args, "aud_len", 1024)),
        "fit_steps": int(steps),
        "code_lr": float(getattr(args, "code_lr", 1e-2)),
        "sample_progressive": bool(getattr(args, "sample_progressive", True)),
        "train_progressive_prob": float(getattr(args, "train_progressive_prob", 1.0)),
        "quantization_steps": {
            group: float(getattr(args, f"quant_step_{group}", 0.0))
            for group in _GROUPS
        },
        "loss_weights": {
            name: float(getattr(args, name, default))
            for name, default in (
                ("lambda_img", 1.0),
                ("lambda_tac", 1.0),
                ("lambda_aud", 1.0),
                ("lambda_stft", 0.1),
                ("lambda_rate", 0.0),
                ("lambda_cm", 0.1),
            )
        },
    }


def _checkpoint_state(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no model state dictionary: {checkpoint_path}")
    return state


def export_parameter_package(
    checkpoint_path: Path,
    package_path: Path,
    args,
    model,
    sample_id: str = "",
    model_signature: Any = None,
) -> None:
    """Store all fitted network weights and quantized sample parameter groups."""
    checkpoint_path = Path(checkpoint_path)
    state = _checkpoint_state(checkpoint_path)
    checkpoint_stat = checkpoint_path.stat()
    steps = {
        group: float(getattr(args, f"quant_step_{group}", 0.0))
        for group in _GROUPS
    }

    codes: Dict[str, Dict[str, Any]] = {}
    for group in _GROUPS:
        prefix = _STATE_PREFIX[group]
        entries = sorted(
            ((key, value) for key, value in state.items() if key.startswith(prefix)),
            key=lambda pair: int(pair[0].rsplit(".", 1)[1]),
        )
        if len(entries) != int(getattr(model, "modulation_base_number", 0)):
            raise ValueError(f"Expected a full {group} pyramid, found {len(entries)} tensors")
        for key, value in entries:
            level = int(key.rsplit(".", 1)[1])
            step = steps[group]
            symbols = (
                torch.round(value / step).to(torch.int32)
                if step > 0
                else value.to(torch.float32)
            )
            symbols = symbols.contiguous().cpu()
            codes[f"{group}.{level}"] = {
                "shape": list(symbols.shape),
                "dtype": "int32" if step > 0 else "float32",
                "data": zlib.compress(symbols.numpy().tobytes(), level=6),
            }

    latent_prefixes = tuple(_STATE_PREFIX[group] for group in _GROUPS)
    network_state = {
        key: value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for key, value in state.items()
        if not key.startswith(latent_prefixes)
    }
    expected_network_keys = {
        key for key in model.state_dict() if not key.startswith(latent_prefixes)
    }
    if set(network_state) != expected_network_keys:
        missing = sorted(expected_network_keys - set(network_state))
        extra = sorted(set(network_state) - expected_network_keys)
        raise ValueError(
            f"Sample network state does not match the current architecture; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    package = {
        "format": _FORMAT,
        "architecture": _ARCHITECTURE,
        "sample_id": str(sample_id),
        "model_signature": model_signature,
        "source_checkpoint": {
            "size": int(checkpoint_stat.st_size),
            "mtime_ns": int(checkpoint_stat.st_mtime_ns),
        },
        "network_state_dict": network_state,
        "network_parameter_bytes": sum(
            int(value.numel() * value.element_size()) for value in network_state.values()
        ),
        "codes": codes,
        "quantization_steps": steps,
        "progressive_levels": int(getattr(model, "modulation_base_number", 0)),
        "channel_compression": int(getattr(model, "channel_compression", 1)),
        "modalities": ["img", "tac", "aud"],
        "shape": {
            "height": int(getattr(model, "h", 0)),
            "width": int(getattr(model, "w", 0)),
            "tactile_length": int(getattr(model, "sig_len", 0)),
            "audio_length": int(getattr(model, "aud_len", 0)),
        },
    }
    package_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(package_path) + ".tmp")
    torch.save(package, tmp_path)
    tmp_path.replace(package_path)


def load_parameter_package(model, package_path: Path) -> Dict[str, Any]:
    """Load one sample's network weights and progressive parameter groups."""
    package = torch.load(package_path, map_location="cpu")
    if package.get("format") != _FORMAT or package.get("architecture") != _ARCHITECTURE:
        raise ValueError(f"Unsupported sample parameter package: {package.get('format')!r}")
    if int(package.get("progressive_levels", -1)) != int(model.modulation_base_number):
        raise ValueError("Parameter package progressive level count does not match the model")
    if int(package.get("channel_compression", 1)) != int(model.channel_compression):
        raise ValueError("Parameter package channel compression does not match the model")

    network_state = package.get("network_state_dict")
    if not isinstance(network_state, dict):
        raise ValueError("Sample parameter package has no per-sample network weights")
    result = model.load_state_dict(network_state, strict=False)
    expected_missing = {
        f"{_STATE_PREFIX[group]}{level}"
        for group in _GROUPS
        for level in range(int(model.modulation_base_number))
    }
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise ValueError(
            f"Sample network state mismatch: missing={result.missing_keys[:5]}, "
            f"unexpected={result.unexpected_keys[:5]}"
        )
    model._sample_network_parameter_bytes = int(package.get("network_parameter_bytes", 0))
    model._sample_package_file_bytes = int(Path(package_path).stat().st_size)

    steps = package.get("quantization_steps", {})
    codes = package.get("codes", {})
    with torch.no_grad():
        for group in _GROUPS:
            params = getattr(model, _MODEL_ATTR[group])
            step = float(steps[group])
            for level, param in enumerate(params):
                key = f"{group}.{level}"
                if key not in codes:
                    raise KeyError(f"Parameter package is missing code tensor {key}")
                encoded = codes[key]
                dtype = torch.int32 if encoded["dtype"] == "int32" else torch.float32
                raw = bytearray(zlib.decompress(encoded["data"]))
                value = torch.frombuffer(raw, dtype=dtype).reshape(encoded["shape"])
                value = value.to(device=param.device, dtype=param.dtype)
                if step > 0:
                    value = value * step
                if value.shape != param.shape:
                    raise ValueError(
                        f"Package tensor {key} has shape {tuple(value.shape)}, "
                        f"expected {tuple(param.shape)}"
                    )
                param.copy_(value)
    return package


def prepare_sample_package(
    model,
    package_path: Path,
    sample_checkpoint: Path,
    args,
    img: torch.Tensor,
    tactile: torch.Tensor,
    audio: torch.Tensor,
    sample_id: str,
) -> Dict[str, Any]:
    """Load a sample package or fit a fresh complete model for this sample."""
    sample_checkpoint = Path(sample_checkpoint)
    best_checkpoint = Path(str(sample_checkpoint) + ".best.pth")
    fit_steps = min(
        max(int(getattr(args, "total_steps", 2000)), 1),
        2000,
    )
    fit_signature = _fit_signature(args, fit_steps)

    if package_path.exists():
        cached = torch.load(package_path, map_location="cpu")
        is_checkpoint_package = bool(cached.get("source_checkpoint"))
        checkpoint_matches = False
        if best_checkpoint.exists() and is_checkpoint_package:
            stat = best_checkpoint.stat()
            checkpoint_matches = cached.get("source_checkpoint") == {
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        fit_matches = (
            not best_checkpoint.exists()
            and cached.get("fit_signature") == fit_signature
        )
        if (
            cached.get("format") == _FORMAT
            and cached.get("architecture") == _ARCHITECTURE
            and cached.get("sample_id") == str(sample_id)
            and (checkpoint_matches or fit_matches)
        ):
            return load_parameter_package(model, package_path)
        package_path.unlink()

    if best_checkpoint.exists():
        export_parameter_package(
            checkpoint_path=best_checkpoint,
            package_path=package_path,
            args=args,
            model=model,
            sample_id=sample_id,
            model_signature={"source": "sample_checkpoint"},
        )
        package = torch.load(package_path, map_location="cpu")
        stat = best_checkpoint.stat()
        package["source_checkpoint"] = {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
        tmp_path = Path(str(package_path) + ".tmp")
        torch.save(package, tmp_path)
        tmp_path.replace(package_path)
        return load_parameter_package(model, package_path)

    from trainer.per_sample_optimizer import train_one_image

    fit_dir = package_path.parent / "sample_fits"
    fit_dir.mkdir(parents=True, exist_ok=True)
    fit_path = fit_dir / f"{sample_id}.pth"
    fit_args = copy.copy(args)
    fit_args.apply_progressive_in_forward = bool(
        getattr(args, "sample_progressive", True)
    )
    fit_args.transmit_img = True
    fit_args.transmit_tac = True
    fit_args.transmit_aud = True
    from trainer.per_sample_optimizer import split_model_parameters

    latent_params, network_params = split_model_parameters(model)
    optimizer = torch.optim.Adam(
        [{"params": network_params}, {"params": latent_params}],
        lr=float(getattr(args, "code_lr", 1e-2)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(fit_steps, 1)
    )
    train_one_image(
        model=model,
        img_tensor=img,
        tac_tensor=tactile,
        aud_tensor=audio,
        total_steps=fit_steps,
        steps_til_summary=0,
        saved_path=str(fit_path),
        lr=float(getattr(args, "code_lr", 1e-2)),
        args=fit_args,
        lambda_img=float(getattr(args, "lambda_img", 1.0)),
        lambda_tac=float(getattr(args, "lambda_tac", 1.0)),
        lambda_aud=float(getattr(args, "lambda_aud", 1.0)),
        lambda_stft=float(getattr(args, "lambda_stft", 0.1)),
        tac_len_1d=int(getattr(args, "sig_len", 576)),
        aud_len_1d=int(getattr(args, "aud_len", 1024)),
        aud_fs=int(getattr(args, "aud_fs", 16000)),
        save_every=0,
        resume=False,
        writer=None,
        optimize_network=True,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_step_size=fit_steps,
    )
    fit_best = Path(str(fit_path) + ".best.pth")
    export_parameter_package(
        checkpoint_path=fit_best,
        package_path=package_path,
        args=args,
        model=model,
        sample_id=sample_id,
        model_signature=fit_signature,
    )
    package = torch.load(package_path, map_location="cpu")
    package["fit_signature"] = fit_signature
    tmp_path = Path(str(package_path) + ".tmp")
    torch.save(package, tmp_path)
    tmp_path.replace(package_path)
    fit_path.unlink(missing_ok=True)
    fit_best.unlink(missing_ok=True)
    return load_parameter_package(model, package_path)
