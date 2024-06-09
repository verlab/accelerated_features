import os
import torch
import tqdm
import argparse
import subprocess
import numpy as np
from modules.model import *

def build_onnx_engine(weights: str,
                    onnx_weight: str,
                    imgsz: tuple = (480,640),
                    use_dynamic_axis: bool = True,
                    onnx_opset: int = 17) -> None:
    if onnx_weight is None:
        raise Exception("Onnx file path cannot be None.")
    dev = 'cpu'
    net = XFeatModel().to(dev).eval()
    net.load_state_dict(torch.load(weights, map_location=dev))
    #Random input
    x = torch.randn(1,3,*imgsz).to(dev)
    if onnx_weight is None:
        raise Exception("Onnx file path cannot be None.")
    if use_dynamic_axis:
        dynamic_axis = {
            "image": {0: "batch"},
            }
    else:
        dynamic_axis = {}
    # net = TempModule(net)
    torch.onnx.export(
        net,
        x,
        onnx_weight,
	    input_names=XFeatModel.get_xfeat_input_names(),
        output_names=XFeatModel.get_xfeat_output_names(),
        dynamic_axes=dynamic_axis,
        opset_version=onnx_opset,
    )

def build_tensorrt_engine(weights: str,
        imgsz: tuple = (480,640),
        fp16_mode: bool = True,
        use_dynamic_axis: bool = True,
        onnx_opset: int = 17,
        workspace: int = 4096) -> None: 

    if weights.endswith(".pt"):
        # Replace ".pt" with ".onnx"
        onnx_weight = weights[:-3] + ".onnx"
    else:
        raise Exception("File path does not end with '.pt'.")
    
    build_onnx_engine(weights, onnx_weight, imgsz, use_dynamic_axis, onnx_opset)

    if not os.path.exists(onnx_weight):
        raise Exception("ONNX export does not exist")

    if onnx_weight.endswith(".onnx"):
        # Replace ".pt" with ".onnx"
        engine_weight = onnx_weight[:-5] + ".engine"
    else:
        raise Exception("File path does not end with '.onnx'.")

    args = ["/usr/src/tensorrt/bin/trtexec"]
    args.append(f"--onnx={onnx_weight}")
    args.append(f"--saveEngine={engine_weight}")
    args.append(f"--workspace={workspace}")

    if fp16_mode:
        args += ["--fp16"]
    
    args += [f"--shapes=image:1x3x{imgsz[0]}x{imgsz[1]}"]

    subprocess.call(args)
    print(f"Finished TensorRT engine export to {engine_weight}.")

def main():
    parser = argparse.ArgumentParser(description='Create ONNX and TensorRT export for XFeat.')
    parser.add_argument('--weights', type=str, default=f'{os.path.abspath(os.path.dirname(__file__))}/weights/xfeat.pt', help='Path to the weights pt file to process')
    parser.add_argument('--imgsz', nargs=2, type=int, default=[480,640], help='Input image size')
    parser.add_argument("--fp16_mode", type=bool, default=True)
    parser.add_argument("--use_dynamic_axis", type=bool, default=True)
    parser.add_argument("--onnx_opset", type=int, default=17)
    parser.add_argument("--workspace", type=int, default=4096)
    args = parser.parse_args()
    weights = args.weights
    imgsz = args.imgsz
    fp16_mode = args.fp16_mode
    onnx_opset = args.onnx_opset
    use_dynamic_axis = args.use_dynamic_axis
    workspace = args.workspace
    build_tensorrt_engine(weights, imgsz, fp16_mode, use_dynamic_axis, onnx_opset, workspace)

if __name__ == '__main__':
    main()




        

