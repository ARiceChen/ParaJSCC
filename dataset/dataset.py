import os
import random
from typing import Optional, Tuple
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path

root = "./data/LMT_108/"

# ============================================================

# ============================================================

def load_1d_signal(path: str) -> Tuple[torch.Tensor, Optional[int]]:
    path = str(path)
    ext = Path(path).suffix.lower()

    if ext in [".txt", ".csv"]:
        data = np.loadtxt(path)
        return torch.tensor(np.asarray(data).reshape(-1), dtype=torch.float32), None

    if ext == ".npy":
        data = np.load(path)
        return torch.tensor(np.asarray(data).reshape(-1), dtype=torch.float32), None

    if ext == ".wav":

        try:
            import soundfile as sf
            data, sr = sf.read(path, dtype="float32", always_2d=False)
            if data.ndim == 2:
                data = data[:, 0]
            return torch.tensor(data.reshape(-1), dtype=torch.float32), int(sr)
        except Exception:
            from scipy.io import wavfile
            sr, data = wavfile.read(path)
            if data.ndim == 2:
                data = data[:, 0]
            if np.issubdtype(data.dtype, np.integer):
                data = data.astype(np.float32) / np.iinfo(data.dtype).max
            else:
                data = data.astype(np.float32)
            return torch.tensor(data.reshape(-1), dtype=torch.float32), int(sr)

    raise ValueError(f"Unsupported extension: {ext}")

# ============================================================

# ============================================================

def _fix_length_1d(x: torch.Tensor, target_len: int) -> torch.Tensor:
    x = x.reshape(-1).to(dtype=torch.float32)
    L = int(x.numel())
    T = int(target_len)
    if L == T:
        return x
    if L > T:
        return x[:T]
    pad = torch.zeros(T - L, dtype=x.dtype)
    return torch.cat([x, pad], dim=0)

def _rms_normalize(x: torch.Tensor, target_rms: float = 0.2, eps: float = 1e-8) -> torch.Tensor:
    x = x.reshape(-1).to(dtype=torch.float32)

    if x.numel() == 0:
        return x

    x = x - x.mean()
    rms = torch.sqrt(torch.mean(x * x) + eps)
    x = x / (rms + eps)
    x = x * float(target_rms)
    x = torch.clamp(x, -1.0, 1.0)
    return x

def _extract_onset_aligned_window(
    x: torch.Tensor,
    target_len: int,
    peak_pos: int,
    jitter: int = 0,
) -> torch.Tensor:
    x = x.reshape(-1).to(dtype=torch.float32)
    L = int(x.numel())
    T = int(target_len)
    if T <= 0:
        raise ValueError(f"target_len must be >0, got {T}")

    peak_pos = int(max(0, min(int(peak_pos), T - 1)))

    if L <= 0:
        return torch.zeros(T, dtype=torch.float32)

    peak = int(torch.argmax(torch.abs(x)).item())
    start = peak - peak_pos + int(jitter)

    pad_left = max(0, -start)
    start = max(0, start)

    end = start + T
    pad_right = max(0, end - L)
    end = min(L, end)

    seg = x[start:end]
    if pad_left > 0:
        seg = torch.cat([torch.zeros(pad_left, dtype=seg.dtype), seg], dim=0)
    if pad_right > 0:
        seg = torch.cat([seg, torch.zeros(pad_right, dtype=seg.dtype)], dim=0)

    if seg.numel() != T:
        seg = _fix_length_1d(seg, T)
    return seg

def _resample_if_needed(x: torch.Tensor, sr: Optional[int], target_sr: Optional[int]) -> torch.Tensor:
    if target_sr is None or sr is None or int(sr) == int(target_sr):
        return x

    try:
        from math import gcd
        from scipy.signal import resample_poly
        sr = int(sr)
        target_sr = int(target_sr)
        g = gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        y = resample_poly(x.detach().cpu().numpy(), up=up, down=down).astype(np.float32)
        return torch.tensor(y, dtype=torch.float32)
    except Exception:

        sr = int(sr)
        target_sr = int(target_sr)
        L = int(x.numel())
        if L <= 1:
            return x
        t = torch.linspace(0, 1, steps=L, dtype=torch.float32)
        new_L = int(round(L * (target_sr / sr)))
        t2 = torch.linspace(0, 1, steps=max(1, new_L), dtype=torch.float32)
        y = torch.interp(t2, t, x.cpu()).to(dtype=torch.float32)
        return y

# ============================================================
# 2) Filename parsing + Trial filter helpers
# ============================================================

def get_trial_from_filename(filename: str) -> str:
    return Path(filename).stem.split("_")[-1]

def parse_sample_id(filename: str, modality: str) -> str:
    stem = Path(filename).stem

    if modality == "image":
        material = stem.split("_Image_")[0]
        trial = stem.split("_")[-1]
        return f"{material}__{trial}"

    if modality == "tactile":
        material = stem.split("_Tapping_")[0]
        trial = stem.split("_")[-1]
        return f"{material}__{trial}"

    if modality == "audio":
        if "_Sound_Tapping_" in stem:
            material = stem.split("_Sound_Tapping_")[0]
        else:
            material = stem.split("_Sound_")[0]
        trial = stem.split("_")[-1]
        return f"{material}__{trial}"

    raise ValueError("Unknown modality")

def build_index(folder, modality: str, allowed_exts: set, only_trial: Optional[str] = None) -> dict:
    folders = [folder] if isinstance(folder, (str, os.PathLike)) else list(folder)
    mapping = {}
    for current_folder in folders:
        if not os.path.isdir(current_folder):
            raise FileNotFoundError(f"LMT-108 modality directory not found: {current_folder}")
        for fn in os.listdir(current_folder):
            p = Path(current_folder) / fn
            if p.suffix.lower() not in allowed_exts:
                continue

            if only_trial is not None and get_trial_from_filename(fn) != only_trial:
                continue

            sid = parse_sample_id(fn, modality)
            if sid in mapping:
                raise ValueError(f"Duplicate LMT-108 sample for {modality}: {sid}")
            mapping[sid] = str(p)
    return mapping

# ============================================================

# ============================================================

class TriModalDataset(Dataset):
    def __init__(
        self,
        tactile_dir: str,
        image_dir: str,
        audio_dir: str,
        transform=None,
        C: int = 1,
        tactile_len: int = 576,
        audio_len: int = 1024,
        # ---- audio processing knobs ----
        audio_target_rms: float = 0.2,
        audio_peak_pos: Optional[int] = None,
        audio_jitter_max: int = 0,
        audio_target_sr: Optional[int] = None,
        # ---- trial filter ----
        only_trial: Optional[str] = None,
        split: str = "all",
        split_seed: int = 100,
        train_ratio: float = 0.7,
    ):
        self.tactile_dir = tactile_dir
        self.image_dir = image_dir
        self.audio_dir = audio_dir
        self.transform = transform
        self.C = int(C)

        self.tactile_len = int(tactile_len)
        self.audio_len = int(audio_len)

        # audio knobs
        self.audio_target_rms = float(audio_target_rms)
        self.audio_jitter_max = int(audio_jitter_max)
        self.audio_target_sr = audio_target_sr

        if audio_peak_pos is None:
            audio_peak_pos = self.audio_len // 4
        self.audio_peak_pos = int(audio_peak_pos)

        # Combine both LMT-108 partitions before making the seeded validation split.
        self.tac_map = build_index(self.tactile_dir, "tactile", {".txt", ".csv", ".npy"}, only_trial=only_trial)
        self.img_map = build_index(self.image_dir, "image", {".jpg", ".jpeg", ".png"}, only_trial=only_trial)
        self.aud_map = build_index(self.audio_dir, "audio", {".wav", ".npy", ".txt", ".csv"}, only_trial=only_trial)

        common = sorted(set(self.tac_map) & set(self.img_map) & set(self.aud_map))
        if len(common) == 0:
            raise RuntimeError(f"No common samples across modalities (only_trial={only_trial}).")

        if split not in {"all", "train", "val"}:
            raise ValueError(f"split must be one of all/train/val, got {split!r}")
        if not 0.0 < float(train_ratio) < 1.0:
            raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

        if split == "all":
            self.sample_ids = common
        else:
            by_material = {}
            for sample_id in common:
                material = sample_id.rsplit("__", 1)[0]
                by_material.setdefault(material, []).append(sample_id)
            rng = random.Random(int(split_seed))
            train_ids, val_ids = [], []
            for material in sorted(by_material):
                ids = sorted(by_material[material])
                rng.shuffle(ids)
                n_train = min(max(int(round(len(ids) * train_ratio)), 1), len(ids) - 1)
                train_ids.extend(ids[:n_train])
                val_ids.extend(ids[n_train:])
            self.sample_ids = sorted(train_ids if split == "train" else val_ids)

    def __len__(self):
        return len(self.sample_ids) * self.C

    def __getitem__(self, idx):
        base_idx = idx // self.C
        aug_idx = idx % self.C
        sid = self.sample_ids[base_idx]

        # -------------------------

        # -------------------------
        tac, _ = load_1d_signal(self.tac_map[sid])
        Lt = int(tac.numel())
        if Lt < self.tactile_len:
            raise ValueError(f"Tactile too short: {Lt} < {self.tactile_len}")

        start_t = int(aug_idx)
        if start_t + self.tactile_len > Lt:
            start_t = Lt - self.tactile_len
        tactile_seg = tac[start_t:start_t + self.tactile_len].to(dtype=torch.float32)  # (576,)

        # -------------------------

        # -------------------------
        aud, sr = load_1d_signal(self.aud_map[sid])
        aud = aud.to(dtype=torch.float32)

        aud = _resample_if_needed(aud, sr=sr, target_sr=self.audio_target_sr)

        # 1) DC + RMS normalize + clip
        aud = _rms_normalize(aud, target_rms=self.audio_target_rms)

        if self.audio_jitter_max > 0 and self.C > 1:
            # deterministic jitter: aug_idx -> [-jitter_max, +jitter_max]
            if self.C == 1:
                jitter = 0
            else:
                frac = aug_idx / float(self.C - 1)
                jitter = int(round((2.0 * frac - 1.0) * self.audio_jitter_max))
        else:
            jitter = 0

        peak_pos = int(max(0, min(self.audio_peak_pos, self.audio_len - 1)))
        audio_seg = _extract_onset_aligned_window(
            aud,
            target_len=self.audio_len,
            peak_pos=peak_pos,
            jitter=jitter,
        )  # (1024,)

        # -------------------------
        # image
        # -------------------------
        image = Image.open(self.img_map[sid]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return tactile_seg, audio_seg, image, sid

# ============================================================
# 4) Dataloader builder
# ============================================================

def get_data(
    dataset_path: str,
    batch_size: int,
    n_worker: int,
    c: int = 1,
    train: bool = True,
    tactile_len: int = 576,
    audio_len: int = 1024,
    audio_target_sr: Optional[int] = None,
    only_trial: Optional[str] = None,
    split: Optional[str] = None,
    split_seed: int = 100,
    train_ratio: float = 0.7,
    shuffle: Optional[bool] = None,
):
    if split is None:
        split = "train" if train else "val"
    tactile_dir = [
        os.path.join(dataset_path, "AccelScansComponents/Tapping/Training"),
        os.path.join(dataset_path, "AccelScansComponents/Tapping/Testing"),
    ]
    image_dir = [
        os.path.join(dataset_path, "ImageScans/NoFlash/Training"),
        os.path.join(dataset_path, "ImageScans/NoFlash/Testing"),
    ]
    audio_dir = [
        os.path.join(dataset_path, "SoundScans/Tapping/Training"),
        os.path.join(dataset_path, "SoundScans/Tapping/Testing"),
    ]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True),
    ])

    dataset = TriModalDataset(
        tactile_dir=tactile_dir,
        image_dir=image_dir,
        audio_dir=audio_dir,
        transform=transform,
        C=c,
        tactile_len=int(tactile_len),
        audio_len=int(audio_len),

        audio_target_rms=0.2,
        audio_peak_pos=None,        # None -> audio_len//4（1024->256）
        audio_jitter_max=0,
        audio_target_sr=audio_target_sr,

        only_trial=only_trial,
        split=split,
        split_seed=split_seed,
        train_ratio=train_ratio,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(split_seed))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train") if shuffle is None else bool(shuffle),
        num_workers=n_worker,
        drop_last=False,
        pin_memory=True,
        generator=loader_generator,
    )

# ============================================================
