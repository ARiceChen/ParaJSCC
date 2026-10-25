
"""
Training argument parser.
"""

import argparse
from typing import Optional, List

def _parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    """
    Parse comma-separated int list: "1,1,0,0,1" -> [1,1,0,0,1]
    Return None if s is None.
    """
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]

def get_args():
    """
    Build and return parsed arguments.

    Returns
    -------
    argparse.Namespace
        Parsed runtime configuration
    """
    parser = argparse.ArgumentParser(
        description="Train the LMT-108 multimodal JSCC model"
    )

    # ============================================================
    # System / Environment Settings
    # ============================================================

    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU id to use (sets CUDA_VISIBLE_DEVICES)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Optional experiment tag appended to run name"
    )

    # ============================================================
    # Dataset Settings
    # ============================================================

    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="./data/LMT_108",
        help="LMT-108 dataset root (this implementation is scoped to LMT-108)"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save trained checkpoints"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (per-image overfitting typically = 1)"
    )

    # ============================================================
    # Signal Resolution (tactile/audio) - 1D preferred
    # ============================================================

    parser.add_argument(
        "--sig_len",
        type=int,
        default=576,
        help="Length of 1D tactile signal (default 576)."
    )

    parser.add_argument(
        "--aud_len",
        type=int,
        default=1024,
        help="Length of 1D audio waveform (default 1024)."
    )

    parser.add_argument(
        "--audio_len",
        type=int,
        default=None,
        help="(Alias of --aud_len) Length of 1D audio waveform."
    )

    parser.add_argument(
        "--aud_fs",
        type=int,
        default=16000,
        help="Audio sampling rate used by preprocessing."
    )

    # ============================================================
    # Training Hyperparameters
    # ============================================================

    parser.add_argument(
        "--code_lr",
        type=float,
        default=1e-2,
        help="Learning rate for per-sample network and parameter optimization."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Deprecated alias for --code_lr."
    )

    parser.add_argument(
        "--total_steps",
        type=int,
        default=2000,
        help="Optimization updates per sample; values above 2000 are capped."
    )

    parser.add_argument(
        "--cuda_graph",
        action="store_true",
        help="Capture the fixed-request CUDA forward and backward step for each sample.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Optional limit on the number of independently fitted samples."
    )

    parser.add_argument(
        "--code_split",
        choices=("train", "val", "all"),
        default="train",
        help="Dataset split whose samples each receive an independently fitted model."
    )
    parser.add_argument(
        "--code_trial",
        type=str,
        default=None,
        help="Optional LMT-108 trial filter, e.g. test1."
    )
    parser.add_argument(
        "--code_sample",
        type=str,
        default=None,
        help="Optional exact LMT-108 sample ID, e.g. MaterialName__test1."
    )
    parser.add_argument("--sample_shard_index", type=int, default=0)
    parser.add_argument(
        "--sample_shard_count",
        type=int,
        default=1,
        help="Number of independent workers used to fit disjoint sample subsets.",
    )

    parser.add_argument(
        "--sample_progressive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Sample modality requests and progressive levels while fitting each sample model."
        ),
    )

    parser.add_argument(
        "--steps_til_summary",
        type=int,
        default=500,
        help="Logging interval (console summary)"
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=500,
        help="Checkpoint save interval"
    )

    parser.add_argument(
        "--tb_every",
        type=int,
        default=100,
        help="TensorBoard logging interval"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=100,
        help="Interval for scoring checkpoints during per-sample fitting.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from saved checkpoint if exists"
    )

    # ============================================================
    # Model / JSCC Parameters
    # ============================================================

    parser.add_argument(
        "--trans_time",
        type=int,
        default=1,
        help="Repetition factor for parameter transmission"
    )

    parser.add_argument(
        "--extra_up_trans",
        type=int,
        default=1,
        help="Additional repetition factor for upsampling module"
    )

    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.0,
        help="Pruning sparsity rate"
    )

    parser.add_argument(
        "--upsampling_kernel_size",
        type=int,
        default=8,
        help="Kernel size for upsampling module"
    )

    parser.add_argument(
        "--static_upsampling_kernel",
        action="store_true",
        help="If set, use fixed upsampling kernel"
    )

    parser.add_argument(
        "--latent_factor",
        type=int,
        default=1,
        help="Latent resolution scaling factor"
    )

    parser.add_argument(
        "--mod_base",
        type=int,
        default=6,
        help="Number of multi-scale latent bases (pyramid levels L)"
    )
    parser.add_argument(
        "--channel_compression",
        type=int,
        choices=(1, 2),
        default=2,
        help="Reduce transmitted code width by this factor before channel mapping.",
    )

    parser.add_argument(
        "--highest_flag",
        type=int,
        default=1,
        help="Whether to include highest resolution latent"
    )

    parser.add_argument(
        "--hid_channel",
        type=int,
        default=20,
        help="Hidden channel width of synthesis network"
    )

    parser.add_argument(
        "--hidden_features",
        type=int,
        default=64,
        help="Hidden width of the per-sample MLP"
    )

    parser.add_argument(
        "--hidden_layer",
        type=int,
        default=2,
        help="Number of hidden layers in the per-sample MLP"
    )

    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help="Pixel prediction scale factor"
    )

    parser.add_argument(
        "--snr",
        type=float,
        default=10.0,
        help="Channel SNR in dB"
    )

    # ============================================================
    # Reconstruction / distortion weights
    # ============================================================

    # Reconstruction and rate weights.
    parser.add_argument("--lambda_img", type=float, default=1.0)
    parser.add_argument("--lambda_tac", type=float, default=1.0)
    parser.add_argument("--lambda_aud", type=float, default=1.0)
    parser.add_argument("--lambda_stft", type=float, default=0.1)
    parser.add_argument("--lambda_rate", type=float, default=1e-6)

    # ============================================================
    # Quantization / rate settings
    # ============================================================

    parser.add_argument(
        "--quant_step_shared",
        type=float,
        default=0.01,
        help="Quantization step for shared latent pyramid. 0 means no quantization."
    )

    parser.add_argument(
        "--quant_step_img",
        type=float,
        default=0.02,
        help="Quantization step for image-private latent pyramid. 0 means no quantization."
    )

    parser.add_argument(
        "--quant_step_tac",
        type=float,
        default=0.02,
        help="Quantization step for tactile-private latent pyramid. 0 means no quantization."
    )

    parser.add_argument(
        "--quant_step_aud",
        type=float,
        default=0.02,
        help="Quantization step for audio-private latent pyramid. 0 means no quantization."
    )

    parser.add_argument(
        "--rate_mode",
        type=str,
        choices=["bandwidth"],
        default="l1",
        help="Rate estimate used during training; bandwidth counts real channel dimensions."
    )

    # ============================================================
    # Progressive / modality-selective transmission
    # ============================================================

    parser.add_argument(
        "--apply_progressive_in_forward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample request-aware modality subsets and progressive levels during training."
    )

    parser.add_argument(
        "--train_progressive_prob",
        type=float,
        default=0.0,
        help=(
            "Probability of sampling a modality subset or truncated level prefix. "
            "The default 0.0 trains all modalities with all levels on every update."
        )
    )

    parser.add_argument(
        "--level_mask_shared",
        type=str,
        default=None,
        help='Optional fixed level mask for shared pyramid, e.g. "1,1,1,0,0,0".'
    )

    parser.add_argument(
        "--level_mask_img",
        type=str,
        default=None,
        help='Optional fixed level mask for image-private pyramid, e.g. "1,1,0,0,0,0".'
    )

    parser.add_argument(
        "--level_mask_tac",
        type=str,
        default=None,
        help='Optional fixed level mask for tactile-private pyramid.'
    )

    parser.add_argument(
        "--level_mask_aud",
        type=str,
        default=None,
        help='Optional fixed level mask for audio-private pyramid.'
    )

    parser.add_argument(
        "--transmit_img",
        action="store_true",
        help="Enable image branch transmission when using fixed progressive settings."
    )

    parser.add_argument(
        "--transmit_tac",
        action="store_true",
        help="Enable tactile branch transmission when using fixed progressive settings."
    )

    parser.add_argument(
        "--transmit_aud",
        action="store_true",
        help="Enable audio branch transmission when using fixed progressive settings."
    )
    parser.add_argument("--lambda_cm", type=float, default=0.1)
    parser.add_argument("--cm_dim", type=int, default=64)

    args = parser.parse_args()

    # ============================================================
    # Post-processing Logic
    # ============================================================

    # Alias: audio_len -> aud_len (if user passes --audio_len)
    if args.audio_len is not None:
        args.aud_len = int(args.audio_len)

    if args.lr is not None:
        args.code_lr = float(args.lr)

    # Parse masks into list[int]
    args.level_mask_shared = _parse_int_list(args.level_mask_shared)
    args.level_mask_img = _parse_int_list(args.level_mask_img)
    args.level_mask_tac = _parse_int_list(args.level_mask_tac)
    args.level_mask_aud = _parse_int_list(args.level_mask_aud)

    # If user did not explicitly set any transmit flag, default to all True.
    if not (args.transmit_img or args.transmit_tac or args.transmit_aud):
        args.transmit_img = True
        args.transmit_tac = True
        args.transmit_aud = True

    return args
