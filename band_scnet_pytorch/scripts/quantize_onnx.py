import argparse
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    CalibrationMethod,
    quantize_static,
)


class RandomCalibReader(CalibrationDataReader):
    def __init__(self, input_name: str, shape, n: int = 16):
        self.input_name = input_name
        self.shape = tuple(shape)
        self.n = n
        self._iter = None

    def get_next(self):
        if self._iter is None:
            data = [
                {self.input_name: np.random.randn(*self.shape).astype(np.float32)}
                for _ in range(self.n)
            ]
            self._iter = iter(data)

        return next(self._iter, None)


def get_onnx_input_name(path: str) -> str:
    model = onnx.load(path)
    initializer_names = {init.name for init in model.graph.initializer}
    graph_inputs = [i.name for i in model.graph.input if i.name not in initializer_names]
    if not graph_inputs:
        raise RuntimeError("No graph input found in ONNX model.")
    return graph_inputs[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/band_scnet_core_int8_static_qdq.onnx",
    )
    parser.add_argument("--input-name", type=str, default=None)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--freq", type=int, default=2049)
    parser.add_argument("--calib-samples", type=int, default=16)
    parser.add_argument("--per-channel", action="store_true")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    input_name = args.input_name or get_onnx_input_name(args.onnx)
    input_shape = (1, 4, args.frames, args.freq)

    before = Path(args.onnx).stat().st_size / (1024 ** 2)

    print("==== Static INT8 QDQ Quantization ====")
    print(f"input model:    {args.onnx}")
    print(f"output model:   {out}")
    print(f"input name:     {input_name}")
    print(f"calib shape:    {input_shape}")
    print(f"calib samples:  {args.calib_samples}")
    print(f"per channel:    {args.per_channel}")

    reader = RandomCalibReader(
        input_name=input_name,
        shape=input_shape,
        n=args.calib_samples,
    )

    quantize_static(
        model_input=args.onnx,
        model_output=str(out),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        per_channel=args.per_channel,
        reduce_range=False,
    )

    after = out.stat().st_size / (1024 ** 2)

    print("==== Result ====")
    print(f"FP32 size:      {before:.2f} MB")
    print(f"INT8 QDQ size:  {after:.2f} MB")
    print(f"size reduction: {100.0 * (1.0 - after / before):.2f}%")

    # optional sanity check
    model = onnx.load(str(out))
    onnx.checker.check_model(model)
    print("ONNX check:     OK")


if __name__ == "__main__":
    main()