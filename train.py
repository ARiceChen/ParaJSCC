import os
import json
import csv
import time
import random
import logging
import copy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:  # TensorBoard is optional for model training.
        def __init__(self, *args, **kwargs):
            logging.getLogger("train_logger").warning(
                "tensorboard is unavailable; scalar and text summaries are disabled."
            )

        def add_scalar(self, *args, **kwargs):
            pass

        def add_text(self, *args, **kwargs):
            pass

        def flush(self):
            pass

        def close(self):
            pass

from dataset.dataset import get_data
from trainer.per_sample_optimizer import split_model_parameters, train_one_image
from config.args import get_args
from models.parajscc import ParaJSCC
from models.parameter_package import export_parameter_package, sample_package_path

# ============================================================
# Config / Constants
# ============================================================

@dataclass(frozen=True)
class ShapeCfg:
    tac_len_1d: int = 576
    aud_len_1d: int = 1024

CSV_FIELDS = [
    "idx", "stem", "ckpt_path", "elapsed_s"
]

SUMMARY_FIELDS = [
    "date",
    "num_finished",
    "trained_now",
    "skipped_existing",
    "failed_count",
    "mean_elapsed_s",
]

# ============================================================
# Utils
# ============================================================

def seed_everything(seed: int = 1029) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

def safe_stem(fname) -> str:
    if isinstance(fname, (list, tuple)):
        return Path(fname[0]).stem
    return Path(fname).stem

def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

def load_progress(progress_path: Path) -> Dict[str, Any]:
    if not progress_path.exists():
        return {"done": {}}
    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if "done" not in obj or not isinstance(obj["done"], dict):
            obj["done"] = {}
        return obj
    except Exception:
        return {"done": {}}

def append_sample_csv(csv_path: Path, row: Dict[str, Any]) -> None:
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})

def write_summary_csv(summary_csv: Path, row: Dict[str, Any]) -> None:
    file_exists = summary_csv.exists()
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_run_name(args) -> str:
    run_tag = getattr(args, "run_tag", "")
    base = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_snr{args.snr}_hid{args.hid_channel}"
    return f"{run_tag}_{base}" if run_tag else base

def write_hparams_as_scalars(writer: SummaryWriter, args) -> None:
    writer.add_scalar("HParams/snr", float(getattr(args, "snr", 0.0)), 0)
    writer.add_scalar("HParams/code_lr", float(getattr(args, "code_lr", 0.0)), 0)
    writer.add_scalar("HParams/lambda_img", float(getattr(args, "lambda_img", 1.0)), 0)
    writer.add_scalar("HParams/lambda_tac", float(getattr(args, "lambda_tac", 1.0)), 0)
    writer.add_scalar("HParams/lambda_aud", float(getattr(args, "lambda_aud", 1.0)), 0)
    writer.add_scalar("HParams/lambda_stft", float(getattr(args, "lambda_stft", 0.0)), 0)
    writer.add_scalar("HParams/lambda_rate", float(getattr(args, "lambda_rate", 0.0)), 0)
    writer.add_scalar("HParams/lambda_cm", float(getattr(args, "lambda_cm", 0.1)), 0)

    writer.add_scalar("HParams/quant_step_shared", float(getattr(args, "quant_step_shared", 0.0)), 0)
    writer.add_scalar("HParams/quant_step_img", float(getattr(args, "quant_step_img", 0.0)), 0)
    writer.add_scalar("HParams/quant_step_tac", float(getattr(args, "quant_step_tac", 0.0)), 0)
    writer.add_scalar("HParams/quant_step_aud", float(getattr(args, "quant_step_aud", 0.0)), 0)

    writer.add_text("HParams/rate_mode", str(getattr(args, "rate_mode", "bandwidth")), 0)
    writer.add_scalar("HParams/train_progressive_prob", float(getattr(args, "train_progressive_prob", 0.0)), 0)
    writer.flush()

def validate_and_prepare_batch(
    batch,
    device: torch.device,
    shape: ShapeCfg,
    allow_batch: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    tactile, audio, image, fname = batch
    stem = safe_stem(fname)

    img = image.to(device)
    tac = tactile.to(device)
    aud = audio.to(device)

    if tac.ndim != 2 or tac.shape[1] != int(shape.tac_len_1d):
        raise ValueError(f"tactile must be (B,{shape.tac_len_1d}), got {tuple(tac.shape)}")

    if aud.ndim != 2 or aud.shape[1] != int(shape.aud_len_1d):
        raise ValueError(f"audio must be (B,{shape.aud_len_1d}), got {tuple(aud.shape)}")

    B = int(tac.shape[0])
    if B != 1 and not allow_batch:
        raise ValueError(f"per-sample overfitting expects batch_size=1, got {B}")

    return img, tac, aud, stem

def compute_mean_from_done(done_map: Dict[str, Any]) -> Dict[str, float]:
    rows = list(done_map.values()) if isinstance(done_map, dict) else []

    def _collect(key: str) -> List[float]:
        vals = []
        for r in rows:
            try:
                v = float(r[key])
                if np.isfinite(v):
                    vals.append(v)
            except Exception:
                pass
        return vals

    def _mean(key: str) -> float:
        vals = _collect(key)
        return float(np.mean(vals)) if len(vals) > 0 else float("nan")

    return {"num_finished": int(len(rows)), "mean_elapsed_s": _mean("elapsed_s")}

# ============================================================
# Model Builder
# ============================================================

def build_model(args, device: torch.device) -> torch.nn.Module:
    model = ParaJSCC(
        args,
        sparsity=args.sparsity,
        in_features=2,
        out_features=3,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layer,
    ).to(device)

    if hasattr(model, "enable_code_channel_noise"):
        model.enable_code_channel_noise = True
    else:
        model.enable_latent_channel_noise = True

    if hasattr(model, "enable_param_noise"):
        model.enable_param_noise = False

    model.lambda_rate = float(getattr(args, "lambda_rate", 0.0))
    model.rate_mode = str(getattr(args, "rate_mode", "bandwidth"))

    model.quant_step_shared = float(getattr(args, "quant_step_shared", 0.0))
    model.quant_step_img = float(getattr(args, "quant_step_img", 0.0))
    model.quant_step_tac = float(getattr(args, "quant_step_tac", 0.0))
    model.quant_step_aud = float(getattr(args, "quant_step_aud", 0.0))

    return model

def _run_signature(args, shape: ShapeCfg) -> Dict[str, Any]:
    keys = (
        "dataset", "seed", "hid_channel", "hidden_features", "hidden_layer",
        "mod_base", "channel_compression", "snr", "total_steps", "code_lr", "lambda_img",
        "lambda_tac", "lambda_aud", "lambda_stft", "lambda_rate", "lambda_cm",
        "quant_step_shared", "quant_step_img", "quant_step_tac", "quant_step_aud",
        "sample_progressive", "train_progressive_prob", "apply_progressive_in_forward",
        "max_train_samples", "code_split", "code_trial", "code_sample",
        "sample_shard_index", "sample_shard_count", "checkpoint_interval",
    )
    signature = {key: getattr(args, key, None) for key in keys}
    if bool(getattr(args, "cuda_graph", False)):
        signature["cuda_graph"] = True
    signature.update({"tactile_length": shape.tac_len_1d, "audio_length": shape.aud_len_1d})
    signature["training_protocol"] = "lmt108_per_sample_full_network_v2"
    return signature

def update_args_with_shapes(args, img: torch.Tensor, B: int, shape: ShapeCfg) -> None:
    args.patch_h = int(img.shape[2])
    args.patch_w = int(img.shape[3])
    args.batch_size = int(B)

    setattr(args, "sig_len", int(shape.tac_len_1d))
    setattr(args, "aud_len_1d", int(shape.aud_len_1d))
    setattr(args, "aud_len", int(shape.aud_len_1d))


# ============================================================
# Main
# ============================================================

def main():
    args = get_args()

    requested_steps = int(args.total_steps)
    args.total_steps = min(max(requested_steps, 1), 2000)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed_everything(args.seed)

    device = get_device()

    shape = ShapeCfg(
        tac_len_1d=int(getattr(args, "sig_len", 576)),
        aud_len_1d=int(getattr(args, "aud_len_1d", getattr(args, "aud_len", 1024))),
    )

    code_sample = getattr(args, "code_sample", None)
    code_split = "all" if code_sample else str(getattr(args, "code_split", "train"))
    dataloader = get_data(
        args.dataset,
        batch_size=1,
        n_worker=1,
        train=(code_split == "train"),
        c=1,
        tactile_len=int(shape.tac_len_1d),
        audio_len=int(shape.aud_len_1d),
        audio_target_sr=getattr(args, "aud_fs", None),
        only_trial=getattr(args, "code_trial", None),
        split=code_split,
        split_seed=int(getattr(args, "seed", 100)),
        train_ratio=0.7,
    )
    if code_sample:
        if code_sample not in dataloader.dataset.sample_ids:
            raise ValueError(
                f"Requested code sample {code_sample!r} is absent from the selected LMT-108 trial."
            )
        dataloader.dataset.sample_ids = [code_sample]

    sample_ids_before_sharding = list(dataloader.dataset.sample_ids)
    sample_position = {sample_id: index for index, sample_id in enumerate(sample_ids_before_sharding)}
    shard_count = int(getattr(args, "sample_shard_count", 1))
    shard_index = int(getattr(args, "sample_shard_index", 0))
    if shard_count < 1 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"Invalid sample shard: index={shard_index}, count={shard_count}")
    if shard_count > 1:
        dataloader.dataset.sample_ids = sample_ids_before_sharding[shard_index::shard_count]
        if not dataloader.dataset.sample_ids:
            raise ValueError(f"Sample shard {shard_index}/{shard_count} is empty")

    ckpt_dir = Path(args.save_dir) / (
        f"LMT108_Tac1D{shape.tac_len_1d}_Aud1D{shape.aud_len_1d}_MM_ImgTacAud_"
        f"Split70Seed{int(getattr(args, 'seed', 100))}_PerSampleNetwork"
    )
    ensure_dir(ckpt_dir)

    shard_tag = f"_shard{shard_index}of{shard_count}" if shard_count > 1 else ""
    logger = setup_logger(ckpt_dir / f"train{shard_tag}.log")
    progress_path = ckpt_dir / f"progress{shard_tag}.json"
    samples_csv = ckpt_dir / f"samples{shard_tag}.csv"
    summary_json = ckpt_dir / f"summary{shard_tag}.json"
    summary_csv = ckpt_dir / f"summary{shard_tag}.csv"

    progress = load_progress(progress_path)
    run_signature = _run_signature(args, shape)
    previous_signature = progress.get("run_signature")
    if previous_signature != run_signature and (previous_signature is not None or progress.get("done")):
        logger.warning("Training progress belongs to a different config; starting a fresh sample run.")
        progress = {"done": {}}
    progress["run_signature"] = run_signature
    done_map: Dict[str, Any] = progress.get("done", {})
    atomic_write_json(progress_path, progress)

    logger.info(f"device={device}")
    if requested_steps != args.total_steps:
        logger.warning(
            f"Requested {requested_steps} updates per sample; capped at {args.total_steps}."
        )
    logger.info(
        "Training protocol: each sample gets an independent ParaJSCC model instance; "
        "all network weights and parameter groups are optimized together."
    )
    logger.info(
        f"Offline code-fit split={code_split}, trial={getattr(args, 'code_trial', None)}, "
        f"sample={code_sample or 'all'}"
    )
    logger.info(
        f"Sample worker shard={shard_index + 1}/{shard_count}, "
        f"assigned_samples={len(dataloader.dataset.sample_ids)}"
    )
    logger.info(f"Tactile 1D len={shape.tac_len_1d}")
    logger.info(f"Audio   1D len={shape.aud_len_1d}")
    logger.info(
        f"Quant steps: shared={getattr(args, 'quant_step_shared', 0.0)}, "
        f"img={getattr(args, 'quant_step_img', 0.0)}, "
        f"tac={getattr(args, 'quant_step_tac', 0.0)}, "
        f"aud={getattr(args, 'quant_step_aud', 0.0)}"
    )
    logger.info(
        f"Rate mode={getattr(args, 'rate_mode', 'bandwidth')}, "
        f"lambda_rate={getattr(args, 'lambda_rate', 0.0)}"
    )
    logger.info(
        f"Progressive: enabled={bool(getattr(args, 'apply_progressive_in_forward', False))}, "
        f"prob={float(getattr(args, 'train_progressive_prob', 0.0))}"
    )

    run_name = make_run_name(args) + shard_tag
    tb_dir = ckpt_dir / "tb_runs" / run_name
    ensure_dir(tb_dir)

    writer = SummaryWriter(log_dir=str(tb_dir))
    write_hparams_as_scalars(writer, args)

    tb_every = int(getattr(args, "tb_every", 100))
    global_step_offset = len(done_map) * int(args.total_steps)

    trained_now = 0
    skipped_existing = 0
    failed_count = 0

    try:
        for idx, batch in enumerate(dataloader):
            sample_limit = getattr(args, "max_train_samples", None)
            if sample_limit is not None and idx >= int(sample_limit):
                break
            try:
                img, tac, aud, stem = validate_and_prepare_batch(batch, device=device, shape=shape)
            except Exception as e:
                failed_count += 1
                logger.exception(f"[Skip Invalid Batch] idx={idx} | error={repr(e)}")
                continue

            saved_path = ckpt_dir / (
                f"inr_hid{args.hid_channel}_{stem}_snr{args.snr}_ImgTacAud1D{shape.aud_len_1d}.pth"
            )

            B = int(tac.shape[0])
            update_args_with_shapes(args, img=img, B=B, shape=shape)

            if stem in done_map:
                package_path = sample_package_path(saved_path, args)
                best_checkpoint = Path(str(saved_path) + ".best.pth")
                if not package_path.exists() and best_checkpoint.exists():
                    try:
                        package_model = build_model(args, device=device)
                        export_parameter_package(
                            model=package_model,
                            checkpoint_path=best_checkpoint,
                            package_path=package_path,
                            args=args,
                            sample_id=stem,
                            model_signature=run_signature,
                        )
                        logger.info(f"Exported reusable parameter package: {package_path}")
                        del package_model
                    except Exception:
                        logger.exception(f"[Package Export Failed] stem={stem}")
                skipped_existing += 1
                continue

            # Keep each sample's model initialization and channel noise reproducible.
            sample_idx = int(sample_position.get(stem, idx))
            seed_everything(int(args.seed) + sample_idx)
            model = build_model(args, device=device)
            sample_args = copy.copy(args)
            sample_args.apply_progressive_in_forward = bool(
                getattr(args, "sample_progressive", True)
            )
            sample_args.transmit_img = True
            sample_args.transmit_tac = True
            sample_args.transmit_aud = True
            latent_params, network_params = split_model_parameters(model)
            sample_optimizer = torch.optim.Adam(
                [{"params": network_params}, {"params": latent_params}],
                lr=float(getattr(args, "code_lr", 1e-2)),
                fused=(device.type == "cuda"),
            )
            sample_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                sample_optimizer,
                T_max=max(int(args.total_steps), 1),
            )

            t0 = time.time()

            try:
                train_one_image(
                    model=model,
                    img_tensor=img,
                    tac_tensor=tac,
                    aud_tensor=aud,

                    total_steps=args.total_steps,
                    steps_til_summary=args.steps_til_summary,
                    saved_path=str(saved_path),
                    lr=float(getattr(args, "code_lr", 1e-2)),

                    args=sample_args,

                    lambda_img=float(getattr(args, "lambda_img", 1.0)),
                    lambda_tac=float(getattr(args, "lambda_tac", 1.0)),
                    lambda_aud=float(getattr(args, "lambda_aud", 1.0)),
                    lambda_stft=float(getattr(args, "lambda_stft", 0.1)),

                    tac_len_1d=int(shape.tac_len_1d),

                    aud_len_1d=int(shape.aud_len_1d),

                    aud_fs=int(getattr(args, "aud_fs", 16000)),
                    save_every=int(getattr(args, "save_every", 200)),
                    resume=bool(getattr(args, "resume", False)),

                    writer=writer,
                    tb_tag="Train",
                    tb_every=tb_every,
                    global_step_offset=global_step_offset,
                    tb_prefix_loss="Loss",
                    optimize_network=True,
                    checkpoint_interval=int(getattr(args, "checkpoint_interval", 100)),
                    cuda_graph=bool(getattr(args, "cuda_graph", False)),
                    optimizer=sample_optimizer,
                    scheduler=sample_scheduler,
                    scheduler_step_size=int(args.total_steps),
                )
            except Exception as e:
                failed_count += 1
                logger.exception(f"[Train Failed] idx={idx} | stem={stem} | error={repr(e)}")
                torch.cuda.empty_cache()
                continue

            package_path = sample_package_path(saved_path, args)
            try:
                export_parameter_package(
                    model=model,
                    checkpoint_path=Path(str(saved_path) + ".best.pth"),
                    package_path=package_path,
                    args=args,
                    sample_id=stem,
                    model_signature=run_signature,
                )
                logger.info(f"Exported reusable parameter package: {package_path}")
            except Exception:
                # Keep the reconstruction checkpoint usable even if package
                # serialization fails; make the incomplete export visible.
                logger.exception(f"[Package Export Failed] stem={stem}")

            elapsed = time.time() - t0

            progress["done"][stem] = {
                "idx": sample_idx,
                "stem": str(stem),
                "ckpt_path": str(saved_path),
                "elapsed_s": float(elapsed),
                "time": datetime.now().isoformat(),
                "rate_mode": str(getattr(args, "rate_mode", "bandwidth")),
                "lambda_rate": float(getattr(args, "lambda_rate", 0.0)),
            }
            atomic_write_json(progress_path, progress)
            done_map = progress["done"]

            append_sample_csv(samples_csv, progress["done"][stem])

            sample_step = len(done_map)
            writer.add_scalar("Samples/Mean/Elapsed_s", float(compute_mean_from_done(done_map)["mean_elapsed_s"]), sample_step)
            writer.flush()

            logger.info(
                f"[{idx}] {stem} | "
                f"elapsed={elapsed:.2f}s | "
                f"mean_elapsed={compute_mean_from_done(done_map)['mean_elapsed_s']:.2f}s"
            )

            trained_now += 1
            global_step_offset += int(args.total_steps)

    finally:
        writer.close()

    final_means = compute_mean_from_done(done_map)
    summary = {
        "date": datetime.now().isoformat(),
        "num_finished": int(final_means["num_finished"]),
        "trained_now": int(trained_now),
        "skipped_existing": int(skipped_existing),
        "failed_count": int(failed_count),
        "mean_elapsed_s": float(final_means["mean_elapsed_s"]),
    }

    atomic_write_json(summary_json, summary)
    write_summary_csv(summary_csv, summary)

    logger.info("========== Training Summary ==========")
    logger.info(f"Finished samples : {summary['num_finished']}")
    logger.info(f"Trained now      : {summary['trained_now']}")
    logger.info(f"Skipped existing : {summary['skipped_existing']}")
    logger.info(f"Failed count     : {summary['failed_count']}")
    logger.info(f"Mean elapsed(s)  : {summary['mean_elapsed_s']:.4f}")
    logger.info(f"Summary JSON     : {summary_json}")
    logger.info(f"Summary CSV      : {summary_csv}")
    logger.info("======================================")

if __name__ == "__main__":
    main()
