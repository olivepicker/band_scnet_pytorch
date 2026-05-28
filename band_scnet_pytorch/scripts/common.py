import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from band_scnet_pytorch import BandSCNet


def strip_prefix_if_present(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k
        for prefix in ("module.", "model.", "net."):
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix):]
        cleaned[new_k] = v
    return cleaned


def load_state_dict_from_checkpoint(path: str) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return strip_prefix_if_present(checkpoint[key])

        if all(isinstance(k, str) for k in checkpoint.keys()):
            return strip_prefix_if_present(checkpoint)

    raise ValueError(f"Could not parse checkpoint format from: {path}")


def make_dummy_wave(seconds: float, sample_rate: int, batch_size: int = 1, channels: int = 2) -> torch.Tensor:
    num_samples = int(seconds * sample_rate)
    return torch.randn(batch_size, 1, channels, num_samples, dtype=torch.float32)


def build_model(weight_path: str, seconds: float, sample_rate: int, dim_hidden: int = 128,
                enc_in_channels: int = 2, dec_out_channels: int = 8,
                device: str = "cpu", strict: bool = False) -> nn.Module:
    model = BandSCNet(dim_hidden, enc_in_channels=enc_in_channels, dec_out_channels=dec_out_channels)
    model.eval()

    with torch.no_grad():
        dummy = make_dummy_wave(seconds=seconds, sample_rate=sample_rate, channels=enc_in_channels)
        _ = model(dummy)

    state_dict = load_state_dict_from_checkpoint(weight_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    if missing:
        print(f"[Warning] Missing keys: {len(missing)}")
        for k in missing[:20]:
            print(f"  - {k}")
    if unexpected:
        print(f"[Warning] Unexpected keys: {len(unexpected)}")
        for k in unexpected[:20]:
            print(f"  - {k}")

    model.to(device)
    model.eval()
    return model


def model_size_mb(model: nn.Module) -> float:
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / (1024 ** 2)


class BandSCNetCore(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.encoder = model.encoder
        self.separation = model.separation
        self.decoder = model.decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e, skips, sd_lengths_list, orig_lengths_list = self.encoder(x)
        e = self.separation(e)
        y = self.decoder(e, skips, sd_lengths_list, orig_lengths_list)

        return y


def infer_stft_shape(seconds: float, sample_rate: int, n_fft: int = 4096, hop_length: int = 1024) -> Tuple[int, int]:
    L = int(seconds * sample_rate)
    padding = (hop_length - (L % hop_length)) % hop_length
    L_pad = L + padding
    freq = n_fft // 2 + 1
    frames = (L_pad - n_fft) // hop_length + 1
    if frames <= 0:
        raise ValueError("Input is too short for the configured n_fft/hop_length.")
    return frames, freq
