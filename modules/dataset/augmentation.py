"""
    "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
    https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    This script implements color + geometric transformations using Kornia.
    Given a dataset of random real unlabeled images, we apply photometric transformations, 
    homography warps and also TPS warps. It also handles black borders by
    pasting a random background image.
"""

import torch
from torch import nn
from torch.utils.data import Dataset
import torch.utils.data as data
from torchvision import transforms
import torch.nn.functional as F

import cv2
import kornia
import kornia.augmentation as K
from kornia.geometry.transform import get_tps_transform as findTPS
from kornia.geometry.transform import warp_points_tps, warp_image_tps

import glob
import random
import tqdm

import numpy as np
import pdb
import time

random.seed(0)
torch.manual_seed(0)

def generateRandomTPS(shape, grid = (8, 6), GLOBAL_MULTIPLIER = 0.3, prob = 0.5):

    h, w = shape
    sh, sw = h/grid[0], w/grid[1]
    src = torch.dstack(torch.meshgrid(torch.arange(0, h + sh , sh),
                         torch.arange(0, w + sw , sw), indexing='ij'))

    offsets = torch.rand(grid[0]+1, grid[1]+1, 2) - 0.5
    offsets *= torch.tensor([ sh/2, sw/2 ]).view(1, 1, 2)  * min(0.97, 2.0 * GLOBAL_MULTIPLIER)
    dst = src + offsets if np.random.uniform() < prob else src
    
    src, dst = src.view(1, -1, 2), dst.view(1, -1, 2)
    src = (src / torch.tensor([h,w]).view(1,1,2) ) * 2 - 1.
    dst = (dst / torch.tensor([h,w]).view(1,1,2) ) * 2 - 1.
    weights, A = findTPS(dst, src)

    return src, weights, A


def generateRandomHomography(shape, GLOBAL_MULTIPLIER = 0.3):
    #Generate random in-plane rotation [-theta,+theta]
    theta = np.radians(np.random.uniform(-30, 30))

    #Generate random scale in both x and y
    scale_x, scale_y = np.random.uniform(0.35, 1.2, 2)

    #Generate random translation shift
    tx , ty = -shape[1]/2.0 , -shape[0]/2.0 
    txn, tyn = np.random.normal(0, 120.0*GLOBAL_MULTIPLIER, 2) 

    c, s = np.cos(theta), np.sin(theta)

    #Affine coeffs
    sx , sy = np.random.normal(0,0.6*GLOBAL_MULTIPLIER,2)

    #Projective coeffs
    p1 , p2 = np.random.normal(0,0.006*GLOBAL_MULTIPLIER,2)


    # Build Homography from parmeterizations
    H_t = np.array(((1,0, tx), (0, 1, ty), (0,0,1))) #t

    H_r = np.array(((c,-s, 0), (s, c, 0), (0,0,1))) #rotation,
    H_a = np.array(((1,sy, 0), (sx, 1, 0), (0,0,1))) # affine
    H_p = np.array(((1, 0, 0), (0 , 1, 0), (p1,p2,1))) # projective

    H_s = np.array(((scale_x,0, 0), (0, scale_y, 0), (0,0,1))) #scale
    H_b = np.array(((1.0,0,-tx +txn), (0, 1, -ty + tyn), (0,0,1))) #t_back,

    #H = H_e * H_s * H_a * H_p
    H = np.dot(np.dot(np.dot(np.dot(np.dot(H_b,H_s),H_p),H_a),H_r),H_t)

    return H


class AugmentationPipe(nn.Module):
    def __init__(
                self, device, load_dataset = True,
                img_dir = "/homeLocal/guipotje/sshfs/datasets/coco_20k",
                warp_resolution = (1200, 900),
                out_resolution = (400, 300),
                sides_crop = 0.2,
                max_num_imgs = 50,
                num_test_imgs = 10,
                batch_size = 1,
                photometric = True,
                geometric = True,
                reload_step = 1_000
                ):
        super(AugmentationPipe, self).__init__()
        self.half = 16
        self.device = device

        self.dims = warp_resolution
        self.batch_size = batch_size
        self.out_resolution = out_resolution 
        self.sides_crop = sides_crop
        self.max_num_imgs = max_num_imgs
        self.num_test_imgs = num_test_imgs
        self.dims_t = torch.tensor([int(self.dims[0]*(1. - self.sides_crop)) - int(self.dims[0]*self.sides_crop) -1,
                                    int(self.dims[1]*(1. - self.sides_crop)) - int(self.dims[1]*self.sides_crop) -1]).float().to(device).view(1,1,2)
        self.dims_s = torch.tensor([ self.dims_t[0,0,0] / out_resolution[0],
                                     self.dims_t[0,0,1] / out_resolution[1]]).float().to(device).view(1,1,2) 

        self.all_imgs = glob.glob(img_dir + '/*.jpg') + glob.glob(img_dir + '/*.png')
        
        self.photometric = photometric
        self.geometric = geometric
        self.cnt = 1
        self.reload_step = reload_step

        list_augmentation = [
                        #kornia.augmentation.RandomChannelShuffle(p=0.5),
                        kornia.augmentation.ColorJitter(0.15, 0.15, 0.15, 0.15, p=1.),
                        kornia.augmentation.RandomEqualize(p = 0.4),
                        kornia.augmentation.RandomGaussianBlur(p = 0.3, sigma = (2.0, 2.0), kernel_size = (7,7))
                        ]
        if photometric is False:
            list_augmentation = []

        self.aug_list = kornia.augmentation.ImageSequential(*list_augmentation)
        
        if len(self.all_imgs) < 10:
            raise RuntimeError('Couldnt find enough images to train. Please check the path: ', img_dir)

        if load_dataset:
            print('[Synthetic] Found a total of ', len(self.all_imgs), ' images for training..')

            if len(self.all_imgs) - num_test_imgs < max_num_imgs:
                raise RuntimeError('Error: test set overlaps with training set! Decrease number of test imgs')

            self.load_imgs()

            self.TPS = True


    def load_imgs(self):

            random.shuffle(self.all_imgs)

            train = []
            fast = cv2.FastFeatureDetector_create(30)
            for p in tqdm.tqdm(self.all_imgs[:self.max_num_imgs], desc='loading train'):
                im = cv2.imread(p)
                halfH, halfW = im.shape[0]//2, im.shape[1]//2
                if halfH > halfW:
                    im = np.rot90(im)
                    halfH, halfW = halfW, halfH

                if im.shape[0] != self.dims[1] or im.shape[1] != self.dims[0]:
                    im = cv2.resize(im, self.dims)

                train.append(np.copy(im))

            self.train = train
            
            self.test = [
                        cv2.resize(cv2.imread(p), self.dims)                 
                        for p in tqdm.tqdm(self.all_imgs[-self.num_test_imgs:],
                                           desc='loading test')
                        ] 

    def norm_pts_grid(self, x):
        if len(x.size()) == 2:
            return (x.view(1,-1,2) * self.dims_s / self.dims_t) * 2. - 1 
        return (x * self.dims_s / self.dims_t) * 2. - 1

    def denorm_pts_grid(self, x):
        if len(x.size()) == 2:
            return ((x.view(1,-1,2) + 1) / 2.) / self.dims_s * self.dims_t
        return ((x+1) / 2.) / self.dims_s * self.dims_t

    def rnd_kps(self, shape, n = 256):
        h, w = shape
        kps = torch.rand(size = (3,n)).to(self.device)
        kps[0,:]*=w
        kps[1,:]*=h
        kps[2,:] = 1.0

        return kps

    def warp_points(self, H, pts):
      scale = self.dims_s.view(-1,2)
      offset = torch.tensor([int(self.dims[0]*self.sides_crop), int(self.dims[1]*self.sides_crop)], device = pts.device).float()
      pts = pts*scale + offset
      pts = torch.vstack( [pts.t(), torch.ones(1, pts.shape[0], device = pts.device)])
      warped = torch.matmul(H, pts)
      warped = warped / warped[2,...] 
      warped = warped.t()[:, :2]
      return (warped - offset) / scale

    @torch.inference_mode()
    def forward(self, x, difficulty = 0.3, TPS = False, prob_deformation = 0.5, test = False):
        """
            Perform augmentation to a batch of images.

            input:
                x -> torch.Tensor(B, C, H, W): rgb images
                difficulty -> float: level of difficulty, 0.1 is medium, 0.3 is already pretty hard
                tps -> bool: Wether to apply non-rigid deformations in images
                prob_deformation -> float: probability to apply a deformation

            return:
                'output'    ->   torch.Tensor(B, C, H, W): rgb images
                Tuple:
                    'H'       ->   torch.Tensor(3,3): homography matrix 
                    'mask'  ->     torch.Tensor(B, H, W): mask of valid pixels after warp
                    (deformation only)
                    src, weights, A are parameters from a TPS warp (all torch.Tensors)

        """

        if self.cnt % self.reload_step == 0:
            self.load_imgs()

        if self.geometric is False:
            difficulty = 0.

        with torch.no_grad():
            x = (x/255.).to(self.device)
            b, c, h, w = x.shape
            shape = (h, w)
              
            ######## Geometric Transformations

            H = torch.tensor(np.array([generateRandomHomography(shape, difficulty) for b in range(self.batch_size)]),
                               dtype = torch.float32).to(self.device)
            
            output = kornia.geometry.transform.warp_perspective(x, H,
                            dsize = shape, padding_mode = 'zeros')

            #crop % of image boundaries each side to reduce invalid pixels after warps
            low_h = int(h * self.sides_crop); low_w = int(w*self.sides_crop)
            high_h = int(h*(1. - self.sides_crop)); high_w= int(w * (1. - self.sides_crop))
            output = output[..., low_h:high_h, low_w:high_w]
            x = x[..., low_h:high_h, low_w:high_w]

            #apply TPS if desired:
            if TPS:
                src, weights, A = None, None, None
                for b in range(self.batch_size):
                    b_src, b_weights, b_A = generateRandomTPS(shape, (8,6), difficulty, prob = prob_deformation)
                    b_src, b_weights, b_A = b_src.to(self.device), b_weights.to(self.device), b_A.to(self.device)

                    if src is None:
                        src, weights, A = b_src, b_weights, b_A
                    else:
                        src = torch.cat((b_src, src))
                        weights = torch.cat((b_weights, weights))
                        A = torch.cat((b_A, A))

                output = warp_image_tps(output, src, weights, A)

            output = F.interpolate(output, self.out_resolution[::-1], mode = 'nearest')
            x = F.interpolate(x, self.out_resolution[::-1], mode = 'nearest')

            mask = ~torch.all(output == 0, dim=1, keepdim=True)
            mask = mask.expand(-1,3,-1,-1)

            # Make-up invalid regions with texture from the batch
            rv = 1 if not TPS else 2
            output_shifted = torch.roll(x, rv, 0)
            output[~mask] = output_shifted[~mask]
            mask = mask[:, 0, :, :]

            ######## Photometric Transformations
            output = self.aug_list(output)

            b, c, h, w = output.shape
            #Correlated Gaussian Noise
            if np.random.uniform() > 0.5 and self.photometric:
                noise = F.interpolate(torch.randn_like(output)*(10/255), (h//2, w//2))
                noise = F.interpolate(noise, (h, w), mode = 'bicubic')
                output = torch.clip( output + noise, 0., 1.)

            #Random shadows
            if np.random.uniform() > 0.6 and self.photometric:
                noise = torch.rand((b, 1, h//64, w//64), device = self.device) * 1.3
                noise = torch.clip(noise, 0.25, 1.0)
                noise = F.interpolate(noise, (h, w), mode = 'bicubic')
                noise = noise.expand(-1, 3, -1, -1)
                output *= noise
                output = torch.clip( output, 0., 1.)

            self.cnt+=1

        if TPS:
            return output, (H, src, weights, A, mask)
        else:
            return output, (H, mask)

    def get_correspondences(self, kps_target, T):
        H, H2, src, W, A = T
        undeformed  = self.denorm_pts_grid(   
                                        warp_points_tps(self.norm_pts_grid(kps_target),
                                        src, W, A) ).view(-1,2)

        warped_to_src = self.warp_points(H@torch.inverse(H2), undeformed)

        return warped_to_src