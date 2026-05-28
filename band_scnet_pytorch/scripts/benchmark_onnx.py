import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def benchmark(sess, x, warmup, repeat):
    name = sess.get_inputs()[0].name
    for _ in range(warmup):
        _ = sess.run(None, {name: x})

    start = time.perf_counter()
    for _ in range(repeat):
        _ = sess.run(None, {name: x})
    return (time.perf_counter() - start) * 1000.0 / repeat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--frames", type=int, required=True, help="STFT time frames T used for core input.")
    parser.add_argument("--freq", type=int, default=2049)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    args = parser.parse_args()

    providers = ["CPUExecutionProvider"]
    if not args.cpu and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    sess = ort.InferenceSession(args.onnx, providers=providers)
    x = np.random.randn(1, 4, args.frames, args.freq).astype(np.float32)

    outputs = sess.run(None, {sess.get_inputs()[0].name: x})
    latency_ms = benchmark(sess, x, args.warmup, args.repeat)
    size_mb = Path(args.onnx).stat().st_size / (1024 ** 2)

    print("==== ONNX Runtime Core Benchmark ====")
    print(f"providers:    {sess.get_providers()}")
    print(f"input shape:  {tuple(x.shape)}")
    print(f"output shape: {[tuple(o.shape) for o in outputs]}")
    print(f"model size:   {size_mb:.2f} MB")
    print(f"latency:      {latency_ms:.3f} ms")


if __name__ == "__main__":
    main()
