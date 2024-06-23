"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""


import sys
import os

ALIKE_PATH = (os.path.abspath(os.path.dirname(__file__) + '/ALIKE'))
sys.path.append(ALIKE_PATH)

import torch
from alike import ALike
import cv2
import numpy as np

import pdb

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

configs = {
    'alike-t': {'c1': 8, 'c2': 16, 'c3': 32, 'c4': 64, 'dim': 64, 'single_head': True, 'radius': 2,
                'model_path': os.path.join(ALIKE_PATH, 'models', 'alike-t.pth')},
    'alike-s': {'c1': 8, 'c2': 16, 'c3': 48, 'c4': 96, 'dim': 96, 'single_head': True, 'radius': 2,
                'model_path': os.path.join(ALIKE_PATH, 'models', 'alike-s.pth')},
    'alike-n': {'c1': 16, 'c2': 32, 'c3': 64, 'c4': 128, 'dim': 128, 'single_head': True, 'radius': 2,
                'model_path': os.path.join(ALIKE_PATH, 'models', 'alike-n.pth')},
    'alike-l': {'c1': 32, 'c2': 64, 'c3': 128, 'c4': 128, 'dim': 128, 'single_head': False, 'radius': 2,
                'model_path': os.path.join(ALIKE_PATH, 'models', 'alike-l.pth')},
}

model = ALike(**configs['alike-t'],
                device=dev,
                top_k=4096,
                scores_th=0.1,
                n_limit=8000)

def extract_alike_kpts(img):
    pred0 = model(img, sub_pixel=True)
    return pred0['keypoints']

def detectAndCompute(img, top_k = 4096):

    img = (img[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

    pred0 = model(img, sub_pixel=True)
    return (torch.tensor(pred0['keypoints'], dtype=torch.float32), 
            torch.tensor(pred0['scores'], dtype=torch.float32), 
            torch.tensor(pred0['descriptors'], dtype = torch.float32)
            )

def match_alike(img1, img2):
    from kornia.feature import DescriptorMatcher
    kornia_matcher = DescriptorMatcher('mnn')
    with torch.inference_mode():
        pred0 = model(img1, sub_pixel=True)
        pred1 = model(img2, sub_pixel=True)

        kp1 = pred0['keypoints']
        kp2 = pred1['keypoints']
        des1 = pred0['descriptors']
        des2 = pred1['descriptors']

        des1 = torch.tensor(des1, device=dev)
        des2 = torch.tensor(des2, device=dev)
        dists, matches = kornia_matcher(des1, des2)

        mkpts0 = kp1[matches[:,0].cpu().numpy()]
        mkpts1 = kp2[matches[:,1].cpu().numpy()]

    return mkpts0, mkpts1
 

def create_xy(h, w, dev):
    y, x = torch.meshgrid(torch.arange(h, device = dev), 
                            torch.arange(w, device = dev))
    xy = torch.cat([x[..., None],y[..., None]], -1).reshape(-1,2)
    return xy

def match_alike_customkp(img1, img2, kp_img1):
    from kornia.feature import DescriptorMatcher
    kornia_matcher = DescriptorMatcher('nn')
    with torch.inference_mode():
        kp_img1 = kp_img1.astype(np.int32)
        pred0 = model(img1, sub_pixel=True, return_dense=True)
        pred1 = model(img2, sub_pixel=True, return_dense=True)

        print(pred0.keys())

        des1 = pred0['desc_map'].cpu()
        des2 = pred1['desc_map'].cpu()

        B, C, H, W = des1.shape
        kp2  = create_xy(H, W, des1.device).numpy()
        des2 = des2.reshape(C, -1).permute(1,0)
        des1 = des1[0, :, kp_img1[:,1], kp_img1[:,0]].permute(1,0)
        kp1 = kp_img1

        print(des1.shape, des2.shape, kp1.shape, kp2.shape)
        dists, matches = kornia_matcher(des1, des2)
        #print('time: ', pred1['time'])


    # Extract matching keypoints' positions
    mkpts0 = kp1[matches[:,0]]
    mkpts1 = kp2[matches[:,1]]

    return mkpts0.astype(np.float32), mkpts1.astype(np.float32)

