"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    Minimal example of how to use XFeat.
"""

import numpy as np
import os
import torch
import tqdm

from modules.xfeat import XFeat

os.environ['CUDA_VISIBLE_DEVICES'] = '' #Force CPU, comment for GPU

xfeat = XFeat()

#Random input
x = torch.randn(1,3,480,640)

#Simple inference with batch = 1
output = xfeat.detectAndCompute(x, top_k = 4096)[0]
print("----------------")
print("keypoints: ", output['keypoints'].shape)
print("descriptors: ", output['descriptors'].shape)
print("scores: ", output['scores'].shape)
print("----------------\n")

x = torch.randn(1,3,480,640)
# Stress test
for i in tqdm.tqdm(range(100), desc="Stress test on VGA resolution"):
	output = xfeat.detectAndCompute(x, top_k = 4096)

# Batched mode
x = torch.randn(4,3,480,640)
outputs = xfeat.detectAndCompute(x, top_k = 4096)
print("# detected features on each batch item:", [len(o['keypoints']) for o in outputs])

# Match two images with sparse features
x1 = torch.randn(1,3,480,640)
x2 = torch.randn(1,3,480,640)
mkpts_0, mkpts_1 = xfeat.match_xfeat(x1, x2)

# Match two images with semi-dense approach -- batched mode with batch size 4
x1 = torch.randn(4,3,480,640)
x2 = torch.randn(4,3,480,640)
matches_list = xfeat.match_xfeat_star(x1, x2)
print(matches_list[0].shape)
