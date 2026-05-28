import argparse
import time

import torch

from common import build_model, make_dummy_wave, model_size_mb


@torch.no_grad()
def summarize_output(y):
    if torch.is_tensor(y):
        return tuple(y.shape)
    if isinstance(y, (list, tuple)):
        return [tuple(t.shape) if torch.is_tensor(t) else type(t).__name__ for t in y]
    if isinstance(y, dict):
        return {k: tuple(v.shape) for k, v in y.items() if torch.is_tensor(v)}
    return type(y).__name__


@torch.no_grad()
def benchmark(model, x, warmup: int, repeat: int):
    for _ in range(warmup):
        _ = model(x)
    if x.device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        _ = model(x)
    if x.device.type == "cuda":
        torch.cuda.synchronize()

    return (time.perf_counter() - start) * 1000.0 / repeat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--seconds", type=float, default=11.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA is not available. Falling back to CPU.")
        args.device = "cpu"

    model = build_model(args.weight, args.seconds, args.sample_rate, device=args.device)

    if args.precision == "fp16":
        if args.device != "cuda":
            raise ValueError("fp16 benchmark requires CUDA.")
        model.half()

    x = make_dummy_wave(args.seconds, args.sample_rate).to(args.device)
    if args.precision == "fp16":
        x = x.half()

    with torch.no_grad():
        y = model(x)

    latency_ms = benchmark(model, x, args.warmup, args.repeat)
    rtf = latency_ms / (args.seconds * 1000.0)

    print("==== PyTorch Full Model Benchmark ====")
    print(f"input shape:  {tuple(x.shape)}  # [B, S, C, L]")
    print(f"output shape: {summarize_output(y)}")
    print(f"model size:   {model_size_mb(model):.2f} MB")
    print(f"latency:      {latency_ms:.3f} ms")
    print(f"RTF:          {rtf:.6f}")


if __name__ == "__main__":
    main()
