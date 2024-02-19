import argparse
import os
import pathlib

import numpy as np
import shark_turbine.aot as aot
import torch


class Conv2DModule(torch.nn.Module):
    def __init__(self, padding: tuple[int, int] = (1, 1)):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            1, 1, kernel_size=(3, 3), stride=(1, 1), padding=padding, bias=False
        )

    def forward(self, x):
        return self.conv(x)


def generate_module(
    padding: tuple[int, int], module_input: np.ndarray, work_dir: pathlib.Path
):
    os.makedirs(name=work_dir, exist_ok=True)

    torch_module = Conv2DModule(padding=padding).to(torch.device("cpu"))

    # Infer shape
    torch_module.eval()
    torch_output = torch_module(torch.tensor(module_input))
    np.save(file=work_dir / "expected_output", arr=torch_output.detach().numpy())

    # Compile
    mlir_module_path = work_dir / "module.mlir"
    export_output = aot.export(torch_module, torch.tensor(module_input))
    with open(mlir_module_path, "w", encoding="utf-8") as f:
        f.write(str(export_output.mlir_module))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pad_x", type=int, required=True)
    parser.add_argument("--pad_y", type=int, required=True)

    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(0)

    work_dir = pathlib.Path(__file__).parent / "output" / "turbine"
    os.makedirs(name=work_dir, exist_ok=True)

    input_shape = (1, 1, 16, 16)
    module_input = np.random.rand(*input_shape).astype(np.float32)
    np.save(file=work_dir.parent / "module_input", arr=module_input)

    generate_module(
        padding=(args.pad_x, args.pad_y),
        module_input=module_input,
        work_dir=work_dir / f"padding_{args.pad_x}_{args.pad_y}",
    )


if __name__ == "__main__":
    main()
