
"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""

import numpy as np
import os
import torch
import torch.nn.functional as F

import tqdm

from modules.model import *
from modules.interpolator import InterpolateSparse2d

class XFeat(nn.Module):
	""" 
		Implements the inference module for XFeat. 
		It supports inference for both sparse and semi-dense feature extraction & matching.
	"""

	def __init__(self, weights = os.path.abspath(os.path.dirname(__file__)) + '/../weights/xfeat.pt', top_k = 4096, detection_threshold=0.05):
		super().__init__()
		self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.net = XFeatModel().to(self.dev).eval()
		self.top_k = top_k
		self.detection_threshold = detection_threshold

		if weights is not None:
			if isinstance(weights, str):
				print('loading weights from: ' + weights)
				self.net.load_state_dict(torch.load(weights, map_location=self.dev))
			else:
				self.net.load_state_dict(weights)

		self.interpolator = InterpolateSparse2d('bicubic')

		#Try to import LightGlue from Kornia
		self.kornia_available = False
		self.lighterglue = None
		try:
			import kornia
			self.kornia_available=True
		except:
			pass


	@torch.inference_mode()
	def detectAndCompute(self, x, top_k = None, detection_threshold = None):
		"""
			Compute sparse keypoints & descriptors. Supports batched mode.

			input:
				x -> torch.Tensor(B, C, H, W): grayscale or rgb image
				top_k -> int: keep best k features
			return:
				List[Dict]: 
					'keypoints'    ->   torch.Tensor(N, 2): keypoints (x,y)
					'scores'       ->   torch.Tensor(N,): keypoint scores
					'descriptors'  ->   torch.Tensor(N, 64): local features
		"""
		if top_k is None: top_k = self.top_k
		if detection_threshold is None: detection_threshold = self.detection_threshold
		x, rh1, rw1 = self.preprocess_tensor(x)

		B, _, _H1, _W1 = x.shape
        
		M1, K1, H1 = self.net(x)
		M1 = F.normalize(M1, dim=1)

		#Convert logits to heatmap and extract kpts
		K1h = self.get_kpts_heatmap(K1)
		mkpts = self.NMS(K1h, threshold=detection_threshold, kernel_size=5)

		#Compute reliability scores
		_nearest = InterpolateSparse2d('nearest')
		_bilinear = InterpolateSparse2d('bilinear')
		scores = (_nearest(K1h, mkpts, _H1, _W1) * _bilinear(H1, mkpts, _H1, _W1))[..., 0]
		scores[torch.all(mkpts == 0, dim=-1)] = -1

		#Select top-k features
		idxs = torch.argsort(-scores)
		mkpts_x  = torch.gather(mkpts[...,0], -1, idxs)[:, :top_k]
		mkpts_y  = torch.gather(mkpts[...,1], -1, idxs)[:, :top_k]
		mkpts = torch.cat([mkpts_x[...,None], mkpts_y[...,None]], dim=-1)
		scores = torch.gather(scores, -1, idxs)[:, :top_k]

		#Interpolate descriptors at kpts positions
		if torch.onnx.is_in_onnx_export() and torch.onnx._globals.GLOBALS.export_onnx_opset_version < 16:
			# The bicubic grid_sample is currently not implemented. 
			# When the opset is less than 16, bilinear_grid_sample will be used as a replacement, which may introduce accuracy errors.
			self.interpolator = InterpolateSparse2d('bilinear')
		feats = self.interpolator(M1, mkpts, H = _H1, W = _W1)

		#L2-Normalize
		feats = F.normalize(feats, dim=-1)

		if torch.onnx.is_in_onnx_export():
			# Avoid warning of torch.tensor being treated as a constant when exporting to ONNX
			mkpts[..., 0] = mkpts[..., 0] * rw1
			mkpts[..., 1] = mkpts[..., 1] * rh1

			return [{'keypoints': mkpts, 'scores': scores, 'descriptors': feats}]

		#Correct kpt scale
		mkpts = mkpts * torch.tensor([rw1,rh1], device=mkpts.device).view(1, 1, -1)

		valid = scores > 0
		return [  
				   {'keypoints': mkpts[b][valid[b]],
					'scores': scores[b][valid[b]],
					'descriptors': feats[b][valid[b]]} for b in range(B) 
			   ]

	@torch.inference_mode()
	def detectAndComputeDense(self, x, top_k = None, multiscale = True):
		"""
			Compute dense *and coarse* descriptors. Supports batched mode.

			input:
				x -> torch.Tensor(B, C, H, W): grayscale or rgb image
				top_k -> int: keep best k features
			return: features sorted by their reliability score -- from most to least
				List[Dict]: 
					'keypoints'    ->   torch.Tensor(top_k, 2): coarse keypoints
					'scales'       ->   torch.Tensor(top_k,): extraction scale
					'descriptors'  ->   torch.Tensor(top_k, 64): coarse local features
		"""
		if top_k is None: top_k = self.top_k
		if multiscale:
			mkpts, sc, feats = self.extract_dualscale(x, top_k)
		else:
			mkpts, feats = self.extractDense(x, top_k)
			sc = torch.ones(mkpts.shape[:2], device=mkpts.device)

		return {'keypoints': mkpts,
				'descriptors': feats,
				'scales': sc }


	@torch.inference_mode()
	def match_lighterglue(self, d0, d1):
		"""
			Match XFeat sparse features with LightGlue (smaller version) -- currently does NOT support batched inference because of padding, but its possible to implement easily.
			input:
				d0, d1: Dict('keypoints', 'scores, 'descriptors', 'image_size (Width, Height)')
			output:
				mkpts_0, mkpts_1 -> np.ndarray (N,2) xy coordinate matches from image1 to image2
				
		"""
		if not self.kornia_available:
			raise RuntimeError('We rely on kornia for LightGlue. Install with: pip install kornia')
		elif self.lighterglue is None:
			from modules.lighterglue import LighterGlue
			self.lighterglue = LighterGlue()

		data = {
				'keypoints0': d0['keypoints'][None, ...],
				'keypoints1': d1['keypoints'][None, ...],
				'descriptors0': d0['descriptors'][None, ...],
				'descriptors1': d1['descriptors'][None, ...],
				'image_size0': torch.tensor(d0['image_size']).to(self.dev)[None, ...],
				'image_size1': torch.tensor(d1['image_size']).to(self.dev)[None, ...]
		}

		#Dict -> log_assignment: [B x M+1 x N+1] matches0: [B x M] matching_scores0: [B x M] matches1: [B x N] matching_scores1: [B x N] matches: List[[Si x 2]], scores: List[[Si]]
		out = self.lighterglue(data)

		idxs = out['matches'][0]

		return d0['keypoints'][idxs[:, 0]].cpu().numpy(), d1['keypoints'][idxs[:, 1]].cpu().numpy()


	@torch.inference_mode()
	def match_xfeat(self, img1, img2, top_k = None, min_cossim = -1):
		"""
			Simple extractor and MNN matcher.
			For simplicity it does not support batched mode due to possibly different number of kpts.
			input:
				img1 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
				img2 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
				top_k -> int: keep best k features
			returns:
				mkpts_0, mkpts_1 -> np.ndarray (N,2) xy coordinate matches from image1 to image2
		"""
		if top_k is None: top_k = self.top_k
		img1 = self.parse_input(img1)
		img2 = self.parse_input(img2)

		out1 = self.detectAndCompute(img1, top_k=top_k)[0]
		out2 = self.detectAndCompute(img2, top_k=top_k)[0]

		idxs0, idxs1 = self.match(out1['descriptors'], out2['descriptors'], min_cossim=min_cossim )

		if torch.onnx.is_in_onnx_export():
			return out1['keypoints'][idxs0], out2['keypoints'][idxs1]
		return out1['keypoints'][idxs0].cpu().numpy(), out2['keypoints'][idxs1].cpu().numpy()

	@torch.inference_mode()
	def match_xfeat_star(self, im_set1, im_set2, top_k = None):
		"""
			Extracts coarse feats, then match pairs and finally refine matches, currently supports batched mode.
			input:
				im_set1 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
				im_set2 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
				top_k -> int: keep best k features
			returns:
				matches -> List[torch.Tensor(N, 4)]: List of size B containing tensor of pairwise matches (x1,y1,x2,y2)
		"""
		if top_k is None: top_k = self.top_k
		im_set1 = self.parse_input(im_set1)
		im_set2 = self.parse_input(im_set2)

		#Compute coarse feats
		out1 = self.detectAndComputeDense(im_set1, top_k=top_k)
		out2 = self.detectAndComputeDense(im_set2, top_k=top_k)

		#Match batches of pairs
		idx0_b, idx1_b = self.batch_match(out1['descriptors'], out2['descriptors'] )

		#Refine coarse matches
		match_mkpts, batch_index = self.refine_matches(out1, out2, idx0_b, idx1_b, fine_conf = 0.25)

		if torch.onnx.is_in_onnx_export():
			return match_mkpts, batch_index

		B = im_set1.shape[0]
		matches = []
		for b in range(B):
			matches.append(match_mkpts[batch_index == b, :])

		return matches if B > 1 else (matches[0][:, :2].cpu().numpy(), matches[0][:, 2:].cpu().numpy())

	def preprocess_tensor(self, x):
		""" Guarantee that image is divisible by 32 to avoid aliasing artifacts. """
		if isinstance(x, np.ndarray) and len(x.shape) == 3:
			x = torch.tensor(x).permute(2,0,1)[None]
		x = x.to(self.dev).float()

		H, W = x.shape[-2:]
		_H, _W = (H//32) * 32, (W//32) * 32
		rh, rw = H/_H, W/_W

		x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=False)
		return x, rh, rw

	def get_kpts_heatmap(self, kpts, softmax_temp = 1.0):
		scores = F.softmax(kpts*softmax_temp, 1)[:, :64]
		B, _, H, W = scores.shape
		heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
		heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H*8, W*8)
		return heatmap

	def NMS(self, x, threshold = 0.05, kernel_size = 5):
		B, _, H, W = x.shape
		pad=kernel_size//2
		local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x)
		pos = (x == local_max) & (x > threshold)
		if torch.onnx.is_in_onnx_export():
			if B != 1:
				raise ValueError('Error: NMS does not support batched mode in ONNX export.')
			return pos.nonzero()[None, ..., 2:].flip(-1)
		pos_batched = [k.nonzero()[..., 1:].flip(-1) for k in pos]

		pad_val = max([len(x) for x in pos_batched])
		pos = torch.zeros((B, pad_val, 2), dtype=torch.long, device=x.device)

		#Pad kpts and build (B, N, 2) tensor
		for b in range(len(pos_batched)):
			pos[b, :len(pos_batched[b]), :] = pos_batched[b]

		return pos

	@torch.inference_mode()
	def batch_match(self, feats1, feats2, min_cossim = -1):
		cossim = torch.bmm(feats1, feats2.permute(0,2,1))
		match12 = torch.argmax(cossim, dim=-1)
		match21 = torch.argmax(cossim.permute(0,2,1), dim=-1)

		indices = torch.arange(match12.shape[1], device=feats1.device).unsqueeze(0).repeat(match12.shape[0], 1)

		mutual = (match21.gather(1, match12) == indices)

		if min_cossim > 0:
			cossim_max, _ = cossim.max(dim=2)
			good = cossim_max > min_cossim
			mutual = mutual & good

		idx0_b = torch.cat([mutual.nonzero()[:, 0, None], indices[mutual][:, None]], axis=1)
		idx1_b = torch.cat([mutual.nonzero()[:, 0, None], match12[mutual][:, None]], axis=1)

		return idx0_b, idx1_b

	def subpix_softmax2d(self, heatmaps, temp = 3):
		N, H, W = heatmaps.shape
		heatmaps = torch.softmax(temp * heatmaps.view(-1, H*W), -1).view(-1, H, W)
		x, y = torch.meshgrid(torch.arange(H, device =  heatmaps.device ), torch.arange(W, device =  heatmaps.device ), indexing = 'ij')
		x = x - (W//2)
		y = y - (H//2)

		coords_x = (x[None, ...] * heatmaps)
		coords_y = (y[None, ...] * heatmaps)
		coords = torch.cat([coords_x[..., None], coords_y[..., None]], -1).view(N, H*W, 2)
		coords = coords.sum(1)

		return coords

	def refine_matches(self, d0, d1, idx0_b, idx1_b, fine_conf = 0.25):
		if torch.onnx.is_in_onnx_export():
			# Improve compatibility when opset is less than 14
			feats1 = d0['descriptors'].flatten(0, 1)[idx0_b[:, 0] * d0['descriptors'].shape[1] + idx0_b[:, 1]]
			feats2 = d1['descriptors'].flatten(0, 1)[idx1_b[:, 0] * d1['descriptors'].shape[1] + idx1_b[:, 1]]
			mkpts_0 = d0['keypoints'].flatten(0, 1)[idx0_b[:, 0] * d0['keypoints'].shape[1] + idx0_b[:, 1]]
			mkpts_1 = d1['keypoints'].flatten(0, 1)[idx1_b[:, 0] * d1['keypoints'].shape[1] + idx1_b[:, 1]]
			sc0 = d0['scales'].flatten(0, 1)[idx0_b[:, 0] * d0['scales'].shape[1] + idx0_b[:, 1]]
		else:
			feats1 = d0['descriptors'][idx0_b[:, 0], idx0_b[:, 1]]
			feats2 = d1['descriptors'][idx1_b[:, 0], idx1_b[:, 1]]
			mkpts_0 = d0['keypoints'][idx0_b[:, 0], idx0_b[:, 1]]
			mkpts_1 = d1['keypoints'][idx1_b[:, 0], idx1_b[:, 1]]
			sc0 = d0['scales'][idx0_b[:, 0], idx0_b[:, 1]]

		#Compute fine offsets
		offsets = self.net.fine_matcher(torch.cat([feats1, feats2],dim=-1))
		conf = F.softmax(offsets*3, dim=-1).max(dim=-1)[0]
		offsets = self.subpix_softmax2d(offsets.view(-1,8,8))

		mkpts_0 += offsets* (sc0[:,None]) #*0.9 #* (sc0[:,None])

		mask_good = conf > fine_conf
		mkpts_0 = mkpts_0[mask_good]
		mkpts_1 = mkpts_1[mask_good]

		match_mkpts = torch.cat([mkpts_0, mkpts_1], dim=-1)
		batch_index = idx0_b[mask_good, 0]

		return match_mkpts, batch_index

	@torch.inference_mode()
	def match(self, feats1, feats2, min_cossim = 0.82):

		cossim = feats1 @ feats2.t()
		cossim_t = feats2 @ feats1.t()
		
		_, match12 = cossim.max(dim=1)
		_, match21 = cossim_t.max(dim=1)

		idx0 = torch.arange(len(match12), device=match12.device)
		mutual = match21[match12] == idx0

		if min_cossim > 0:
			cossim, _ = cossim.max(dim=1)
			good = cossim > min_cossim
			idx0 = idx0[mutual & good]
			idx1 = match12[mutual & good]
		else:
			idx0 = idx0[mutual]
			idx1 = match12[mutual]

		return idx0, idx1

	def create_xy(self, h, w, dev):
		y, x = torch.meshgrid(torch.arange(h, device = dev), 
								torch.arange(w, device = dev), indexing='ij')
		xy = torch.cat([x[..., None],y[..., None]], -1).reshape(-1,2)
		return xy

	def extractDense(self, x, top_k = 8_000):
		if top_k < 1:
			top_k = 100_000_000

		x, rh1, rw1 = self.preprocess_tensor(x)

		M1, K1, H1 = self.net(x)
		
		B, C, _H1, _W1 = M1.shape
		
		xy1 = (self.create_xy(_H1, _W1, M1.device) * 8).expand(B,-1,-1)

		M1 = M1.permute(0,2,3,1).flatten(1, 2) # B, H*W, C
		H1 = H1.permute(0,2,3,1).flatten(1) # B, H*W

		_, top_k = torch.topk(H1, k = torch.min(H1.shape[1], torch.tensor(top_k)), dim=-1)

		feats = torch.gather( M1, 1, top_k[...,None].expand(-1, -1, 64))
		mkpts = torch.gather(xy1, 1, top_k[...,None].expand(-1, -1, 2))
		if torch.onnx.is_in_onnx_export():
			# Avoid warning of torch.tensor being treated as a constant when exporting to ONNX
			mkpts[..., 0] = mkpts[..., 0] * rw1
			mkpts[..., 1] = mkpts[..., 1] * rh1
		else:
			mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1,-1)

		return mkpts, feats

	def extract_dualscale(self, x, top_k, s1 = 0.6, s2 = 1.3):
		x1 = F.interpolate(x, scale_factor=s1, align_corners=False, mode='bilinear')
		x2 = F.interpolate(x, scale_factor=s2, align_corners=False, mode='bilinear')

		B, _, _, _ = x.shape

		mkpts_1, feats_1 = self.extractDense(x1, int(top_k*0.20))
		mkpts_2, feats_2 = self.extractDense(x2, int(top_k*0.80))

		mkpts = torch.cat([mkpts_1/s1, mkpts_2/s2], dim=1)
		sc1 = torch.ones(mkpts_1.shape[:2], device=mkpts_1.device) * (1/s1)
		sc2 = torch.ones(mkpts_2.shape[:2], device=mkpts_2.device) * (1/s2)
		sc = torch.cat([sc1, sc2],dim=1)
		feats = torch.cat([feats_1, feats_2], dim=1)

		return mkpts, sc, feats

	def parse_input(self, x):
		if len(x.shape) == 3:
			x = x[None, ...]

		if isinstance(x, np.ndarray):
			x = torch.tensor(x).permute(0,3,1,2)/255

		return x
