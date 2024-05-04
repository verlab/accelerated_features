import types

import argparse
import torch
import torch.nn.functional as F
import onnx
import onnxsim

from modules.xfeat import XFeat


def preprocess_tensor(self, x):
    return x, 1.0, 1.0 # Assuming the width and height are multiples of 32, bypass preprocessing.
    """ Guarantee that image is divisible by 32 to avoid aliasing artifacts. """
    H, W = x.shape[-2:]
    _H, _W = (H//32) * 32, (W//32) * 32
    rh, rw = H/_H, W/_W

    x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=False)
    return x, rh, rw


def parse_args():
    parser = argparse.ArgumentParser(description="Export XFeat model to ONNX.")
    parser.add_argument(
        "--height",
        type=int,
        default=640,
        help="Input image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Input image width.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4800,
        help="Keep best k features.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic axes.",
    )
    parser.add_argument(
        "--export_path",
        type=str,
        default="./model.onnx",
        help="Path to export ONNX model.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=16,
        help="ONNX opset version.",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.dynamic:
        args.height = 640
        args.width = 640
    else:
        assert args.height % 32 == 0 and args.width % 32 == 0, "Height and width must be multiples of 32."

    if args.top_k > 4800:
        print("Warning: The current maximum supported value for TopK in TensorRT is 3840, which coincidentally equals 4800 * 0.8. Please ignore this warning if TensorRT will not be used in the future.")

    xfeat = XFeat()
    xfeat.top_k = args.top_k
    xfeat = xfeat.cpu().eval()
    xfeat.forward = xfeat.match_xfeat_star
    # Bypass preprocess_tensor
    xfeat.preprocess_tensor = types.MethodType(preprocess_tensor, xfeat)

    x1 = torch.randn(1, 3, args.height, args.width, dtype=torch.float32, device='cpu')
    x2 = torch.randn(1, 3, args.height, args.width, dtype=torch.float32, device='cpu')

    dynamic_axes = {"image0": {2: "height", 3: "width"}, "image1": {2: "height", 3: "width"}}

    torch.onnx.export(
        xfeat,
        (x1, x2),
        args.export_path,
        verbose=False,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["image0", "image1"],
        output_names=["mkpts_0", "mkpts_1"],
        dynamic_axes=dynamic_axes if args.dynamic else None,
    )

    model_onnx = onnx.load(args.export_path)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, "assert check failed"
    onnx.save(model_onnx, args.export_path)

    print(f"Model exported to {args.export_path}")
