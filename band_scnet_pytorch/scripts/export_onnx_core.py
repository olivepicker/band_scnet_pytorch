import argparse
from pathlib import Path

import torch

from common import build_model, BandSCNetCore, infer_stft_shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, required=True)
    parser.add_argument("--output", type=str, default="artifacts/band_scnet_core.onnx")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    model = build_model(args.weight, args.seconds, args.sample_rate, device="cpu")
    core = BandSCNetCore(model).eval()

    T, F = infer_stft_shape(args.seconds, args.sample_rate, model.n_fft, model.hop_length)
    dummy = torch.randn(1, 4, T, F, dtype=torch.float32)

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "stft_features": {0: "batch", 2: "frames", 3: "freq"},
            "estimated_sources_stft": {0: "batch", 2: "frames", 3: "freq"},
        }

    print("==== Export ONNX Core Network ====")
    print(f"input shape: {tuple(dummy.shape)}  # [B, 4, T, F]")
    print(f"output:      {out}")
    print("note: STFT/ISTFT are excluded and treated as pre/post-processing.")

    # with torch.no_grad():
    #     y = core(dummy)
    #     print("forward ok:", y.shape)

    with torch.no_grad():
        torch.onnx.export(
            core,
            dummy,
            str(out),
            input_names=["stft_features"],
            output_names=["estimated_sources_stft"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
        )

    print(f"Exported: {out}")


if __name__ == "__main__":
    main()
