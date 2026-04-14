import torch
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.utils import (
    generate_mx_scales,
)


def show_mode(name: str, block_max: torch.Tensor):
    args = QuantizationArgs(
        num_bits=4,
        type="float",
        strategy="group",
        group_size=32,
        scale_dtype=torch.uint8,
        scale_calculation_mode=name,
    )

    scale_exp = generate_mx_scales(
        block_max,
        num_bits=args.num_bits,
        scale_calculation_mode=args.scale_calculation_mode,
    ).to(torch.uint8)
    scale = 2.0 ** (scale_exp.to(torch.int32) - 127).to(torch.float32)

    print(f"\n{name.upper()} scales")
    print("block_max      :", block_max.tolist())
    print("e8m0 exponent  :", scale_exp.tolist())
    print("float scale    :", [float(value) for value in scale])


if __name__ == "__main__":
    # Values chosen to make the FLOOR and RCEIL difference obvious.
    block_max = torch.tensor([0.10, 0.40, 0.90, 6.00, 7.00], dtype=torch.float32)
    print("MXFP4 scale calculation demo")
    show_mode("floor", block_max)
    show_mode("rceil", block_max)
