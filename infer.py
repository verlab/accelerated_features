"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/
"""

import os
import cv2
import numpy as np
from modules.model import *
from scipy.ndimage import maximum_filter


class InterpolateSparse2d():
    """ Efficiently interpolate numpy array at given sparse 2D positions. """

    def __init__(self, mode='bicubic', align_corners=False):
        self.mode = mode
        self.align_corners = align_corners

    def normgrid(self, x, H, W):
        """ Normalize coords to [-1,1]. """
        return 2.0 * (x / np.array([W - 1, H - 1], dtype=x.dtype)) - 1.0
    def forward(self, x, pos, H, W):
        """
        Input
            x: numpy array of shape [B, C, H, W] feature tensor
            pos: numpy array of shape [B, N, 2] tensor of positions
            H, W: int, original resolution of input 2d positions -- used in normalization [-1,1]

        Returns
            [B, N, C] sampled channels at 2d positions
        """
        B, C, _, _ = x.shape
        grids = self.normgrid(pos, H, W).astype(np.float32)

        output = grid_sample_numpy(x, grids[None,], mode=self.mode, align_corners=False)
        # x1 = F.grid_sample(torch.tensor(x), torch.tensor(grids).unsqueeze(-2), mode=self.mode, align_corners=False)
        # output_torch = x1.permute(0, 2, 3, 1).squeeze(-2).numpy()
        # return output_torch

        return output


def bilinear_grid_sample(im, grid, align_corners=False):
    """Given an input and a flow-field grid, computes the output using input
    values and pixel locations from grid. Supported only bilinear interpolation
    method to sample the input pixels.
    Args:
        im (torch.Tensor): Input feature map, shape (N, C, H, W)
        grid (torch.Tensor): Point coordinates, shape (N, Hg, Wg, 2)
        align_corners (bool): If set to True, the extrema (-1 and 1) are
            considered as referring to the center points of the input’s
            corner pixels. If set to False, they are instead considered as
            referring to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.
    Returns:
        torch.Tensor: A tensor with sampled points, shape (N, C, Hg, Wg)
    """

    n, c, h, w = im.shape

    gn, gh, gw, _ = grid.shape

    assert n == gn

    x = grid[:, :, :, 0]

    y = grid[:, :, :, 1]

    if align_corners:

        x = ((x + 1) / 2) * (w - 1)

        y = ((y + 1) / 2) * (h - 1)

    else:

        x = ((x + 1) * w - 1) / 2

        y = ((y + 1) * h - 1) / 2

    x = x.contiguous().view(n, -1)

    y = y.contiguous().view(n, -1)

    x0 = torch.floor(x).long()

    y0 = torch.floor(y).long()

    x1 = x0 + 1

    y1 = y0 + 1

    wa = ((x1 - x) * (y1 - y)).unsqueeze(1)

    wb = ((x1 - x) * (y - y0)).unsqueeze(1)

    wc = ((x - x0) * (y1 - y)).unsqueeze(1)

    wd = ((x - x0) * (y - y0)).unsqueeze(1)

    # Apply default for grid_sample function zero padding

    im_padded = F.pad(im, pad=[1, 1, 1, 1], mode='constant', value=0)

    padded_h = h + 2

    padded_w = w + 2

    # save points positions after padding

    x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1

    # Clip coordinates to padded image size

    x0 = torch.where(x0 < 0, torch.tensor(0), x0)

    x0 = torch.where(x0 > padded_w - 1, torch.tensor(padded_w - 1), x0)

    x1 = torch.where(x1 < 0, torch.tensor(0), x1)

    x1 = torch.where(x1 > padded_w - 1, torch.tensor(padded_w - 1), x1)

    y0 = torch.where(y0 < 0, torch.tensor(0), y0)

    y0 = torch.where(y0 > padded_h - 1, torch.tensor(padded_h - 1), y0)

    y1 = torch.where(y1 < 0, torch.tensor(0), y1)

    y1 = torch.where(y1 > padded_h - 1, torch.tensor(padded_h - 1), y1)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # x0 = torch.where(x0 < 0, torch.tensor(0).to(device), x0)

    # x0 = torch.where(x0 > padded_w - 1, torch.tensor(padded_w - 1).to(device), x0)

    # x1 = torch.where(x1 < 0, torch.tensor(0).to(device), x1)

    # x1 = torch.where(x1 > padded_w - 1, torch.tensor(padded_w - 1).to(device), x1)

    # y0 = torch.where(y0 < 0, torch.tensor(0).to(device), y0)

    # y0 = torch.where(y0 > padded_h - 1, torch.tensor(padded_h - 1).to(device), y0)

    # y1 = torch.where(y1 < 0, torch.tensor(0).to(device), y1)

    # y1 = torch.where(y1 > padded_h - 1, torch.tensor(padded_h - 1).to(device), y1)

    im_padded = im_padded.view(n, c, -1)

    x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)

    x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)

    x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)

    x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)

    Ia = torch.gather(im_padded, 2, x0_y0)

    Ib = torch.gather(im_padded, 2, x0_y1)

    Ic = torch.gather(im_padded, 2, x1_y0)

    Id = torch.gather(im_padded, 2, x1_y1)

    return (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)


def grid_sample_numpy(im, grid, mode="bilinear", align_corners=False):

    n, c, h, w = im.shape
    gn, gh, gw, _ = grid.shape
    assert n == gn

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]

    if align_corners:
        x = ((x + 1) / 2) * (w - 1)
        y = ((y + 1) / 2) * (h - 1)
    else:
        x = ((x + 1) * w - 1) / 2
        y = ((y + 1) * h - 1) / 2

    x = x.reshape(n, -1)
    y = y.reshape(n, -1)

    if mode == "bilinear":
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        wa = ((x1 - x) * (y1 - y)).reshape(n, 1, -1)
        wb = ((x1 - x) * (y - y0)).reshape(n, 1, -1)
        wc = ((x - x0) * (y1 - y)).reshape(n, 1, -1)
        wd = ((x - x0) * (y - y0)).reshape(n, 1, -1)

        im_padded = np.pad(im, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
        padded_h = h + 2
        padded_w = w + 2

        x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1

        x0 = np.clip(x0, 0, padded_w - 1)
        x1 = np.clip(x1, 0, padded_w - 1)
        y0 = np.clip(y0, 0, padded_h - 1)
        y1 = np.clip(y1, 0, padded_h - 1)

        im_padded = im_padded.reshape(n, c, -1)

        x0_y0 = (x0 + y0 * padded_w).reshape(n, 1, -1)
        x0_y1 = (x0 + y1 * padded_w).reshape(n, 1, -1)
        x1_y0 = (x1 + y0 * padded_w).reshape(n, 1, -1)
        x1_y1 = (x1 + y1 * padded_w).reshape(n, 1, -1)

        Ia = np.take_along_axis(im_padded, x0_y0, axis=2)
        Ib = np.take_along_axis(im_padded, x0_y1, axis=2)
        Ic = np.take_along_axis(im_padded, x1_y0, axis=2)
        Id = np.take_along_axis(im_padded, x1_y1, axis=2)

        output = (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)

    elif mode == "nearest":
        x_nearest = np.round(x).astype(np.int64)
        y_nearest = np.round(y).astype(np.int64)

        x_nearest = np.clip(x_nearest, 0, w - 1)
        y_nearest = np.clip(y_nearest, 0, h - 1)

        # Apply default for grid_sample function zero padding
        im_padded = np.pad(im, pad_width=((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

        padded_h = h + 2
        padded_w = w + 2

        x_nearest = x_nearest + 1
        y_nearest = y_nearest + 1

        im_padded = im_padded.reshape(n, c, -1)

        nearest_indices = (x_nearest + y_nearest * padded_w).reshape(n, 1, -1)

        sampled = np.take_along_axis(im_padded, nearest_indices, axis=-1)

        output = sampled.reshape(n, c, gh, gw)

    elif mode == "bicubic":
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)

        dx = x - x0
        dy = y - y0

        def cubic_contribution(t, a = -0.5):
            a = -0.75
            abs_t = np.abs(t)
            abs_t2 = abs_t ** 2
            abs_t3 = abs_t ** 3
            return np.where(abs_t <= 1, (a + 2) * abs_t3 - (a + 3) * abs_t2 + 1,
                            np.where(abs_t <= 2, a * abs_t3 - (5 * a) * abs_t2 + (8 * a) * abs_t - (4 * a),
                                     0))

        weights_x = [cubic_contribution(dx + 1), cubic_contribution(dx), cubic_contribution(dx - 1), cubic_contribution(dx - 2)]
        weights_y = [cubic_contribution(dy + 1), cubic_contribution(dy), cubic_contribution(dy - 1), cubic_contribution(dy - 2)]

        output = []
        for C in range(c):
            result = np.zeros((n, 1, gh * gw))
            for i in range(4):
                for j in range(4):
                    wx = weights_x[i].reshape(n, 1, gh * gw)
                    wy = weights_y[j].reshape(n, 1, gh * gw)
                    ix = x0 + i - 1
                    iy = y0 + j - 1
                    ix = np.clip(ix, 0, w - 1)
                    iy = np.clip(iy, 0, h - 1)
                    value = im[:, C][None, ][np.arange(n)[:, None], :, iy[:, None], ix]
                    value = value.reshape(n, 1, gh * gw)
                    result += wx * wy * value
            output.append(result[0])
        output = np.array(output)[None, ]

    else:
        exit(0)

    output = output.transpose((0, 3, 2, 1)).squeeze(-2)
    return output


def preprocess(x):
    rh1 = x.shape[0] / 640
    rw1 = x.shape[1] / 640
    img = cv2.resize(x, [640, 640], interpolation=cv2.INTER_LINEAR)
    img = img.transpose((2, 0, 1))[None, ]
    B, _, _H1, _W1, = img.shape
    return img, rh1, rw1, B, _H1, _W1

class infer_torch():
    def __init__(self, weights='xfeat.pt'):
        import torch
        self.dev = torch.device('cpu') #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = XFeatModel().to(self.dev).eval()
        if weights is not None:
            if isinstance(weights, str):
                print('loading weights from: ' + weights)
                self.net.load_state_dict(torch.load(weights, map_location=self.dev))
            else:
                self.net.load_state_dict(weights)

    def forward(self, x):
        x = torch.tensor(x).to(self.dev).float() / 255.
        M1, K1, H1 = self.net(x)
        return M1.cpu().numpy(), K1.cpu().numpy(), H1.cpu().numpy()


class infer_onnx():
    def __init__(self, weights='xfeat.onnx'):
        from onnxruntime import InferenceSession
        self.model = InferenceSession(weights, providers={'CPUExecutionProvider'})
        self.input_name = [i.name for i in self.model.get_inputs()][0]
        self.output_name = [i.name for i in self.model.get_outputs()]

    def forward(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32) / 255.
        M1, K1, H1 = self.model.run(self.output_name, {self.input_name: x})
        return M1, K1, H1


class infer_rknn():
    def __init__(self, weights='xfeat.onnx'):
        from rknn.api import RKNN
        self.rknn = RKNN()
        self.rknn.config(mean_values=[0, 0, 0], std_values=[1, 1, 1], target_platform='rk3588', optimization_level=2)
        # Load model
        print('--> Loading model')
        ret = self.rknn.load_onnx(model=weights)
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        print('done')

        # Build model
        print('--> Building model')
        ret = self.rknn.build(do_quantization=True, dataset='weights/dataset.txt')
        if ret != 0:
            print('Build model failed!')
            exit(ret)
        print('done')

        # Init runtime environment
        print('--> Init runtime environment')
        ret = self.rknn.init_runtime()
        if ret != 0:
            print('Init runtime environment failed!')

    def forward(self, x):
        # Inference
        print('--> Running model')
        M1, K1, H1 = self.rknn.inference(inputs=[x], data_format=['nchw'])
        return M1, K1, H1


class infer_rknnlite():
    def __init__(self, weights='xfeat.rknn'):
        from rknnlite.api import RKNNLite
        self.rknn_lite = RKNNLite()

        # Load RKNN model
        print('--> Load RKNN model')
        ret = self.rknn_lite.load_rknn(weights)
        if ret != 0:
            print('Load RKNN model failed')
            exit(ret)
        print('done')

        # Init runtime environment
        print('--> Init runtime environment')
        # For RK3576 / RK3588, specify which NPU core the model runs on through the core_mask parameter.
        ret = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')

    def forward(self, x):
        # Inference
        print('--> Running model')
        M1, K1, H1 = self.rknn_lite.inference(inputs=[x], data_format=['nchw'])
        return M1, K1, H1


class XFeat(nn.Module):
    """
        Implements the inference module for XFeat.
        It supports inference for both sparse and semi-dense feature extraction & matching.
    """

    def __init__(self, weights=os.path.abspath(os.path.dirname(__file__)) + 'xfeat.pt', top_k=4096, infer_type='rknn_lite'):
        super().__init__()
        if infer_type == 'torch':
            self.net = infer_torch("weights/xfeat.pt")
        elif infer_type == 'onnx':
            self.net = infer_onnx("weights/xfeat.onnx")
        elif infer_type == "rknn":
            self.net = infer_rknn("weights/xfeat.onnx")
        elif infer_type == 'rknn_lite':
            self.net = infer_rknnlite("weights/xfeat.rknn")
        else:
            print("infer type should in ['torch', 'onnx', 'rknn', 'rknn_lite']")

        self.top_k = top_k
        self.interpolator = InterpolateSparse2d('bicubic')
        self._nearest = InterpolateSparse2d('nearest')
        self._bilinear = InterpolateSparse2d('bilinear')

    @torch.inference_mode()
    def detectAndCompute(self, x, top_k=None):
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
        img, rh1, rw1, B, _H1, _W1 = preprocess(x)
        M1, K1, H1 = self.net.forward(img)
        M1 = self.normalize(M1, axis=1)

        # Convert logits to heatmap and extract kpts
        K1h = self.get_kpts_heatmap(K1)
        mkpts = self.NMS(K1h, threshold=0.05, kernel_size=5)

        # Compute reliability scores
        scores = (self._nearest.forward(K1h, mkpts, _H1, _W1) * self._bilinear.forward(H1, mkpts, _H1, _W1)).squeeze(-1)
        # scores = (grid_sample_scipy(K1h, mkpts, H=_H1, W=_W1, mode='nearest') * grid_sample_scipy(H1, mkpts, H=_H1, W=_W1, mode='bilinear')).squeeze(-1)
        # 找出mkpts中所有元素全为0的行的索引,对应值设置为-1
        scores[np.all(mkpts == 0, axis=-1)] = -1

        # 根据scores的降序排列获取索引
        idxs = np.argsort(-scores, axis=-1)
        #
        # # 选择top-k特征点的坐标和分数
        mkpts_x = np.take_along_axis(mkpts[..., 0], idxs, axis=-1)[:, :top_k]
        mkpts_y = np.take_along_axis(mkpts[..., 1], idxs, axis=-1)[:, :top_k]
        mkpts = np.stack([mkpts_x, mkpts_y], axis=-1)
        scores = np.take_along_axis(scores, idxs, axis=-1)[:, :top_k]

        # 在关键点位置插值描述符
        feats = self.interpolator.forward(M1, mkpts, H=_H1, W=_W1)
        # feats = grid_sample_scipy(M1, mkpts, H=_H1, W=_W1, mode='bicubic')
        # L2-Normalize
        # feats = feats.cpu().numpy()#(1,4096,64)
        feats = self.normalize(feats, axis=-1)

        # 校正关键点尺度
        mkpts *= np.array([rw1, rh1]).reshape(1, 1, -1)

        valid = scores > 0

        return [
            {'keypoints': mkpts[b][valid[b]],
             'scores': scores[b][valid[b]],
             'descriptors': feats[b][valid[b]]} for b in range(B)
        ]

    @torch.inference_mode()
    def match_xfeat(self, img1, img2, top_k=None, min_cossim=-1):
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
        # img1 = self.parse_input(img1)
        # img2 = self.parse_input(img2)

        out1 = self.detectAndCompute(img1, top_k=top_k)[0]
        out2 = self.detectAndCompute(img2, top_k=top_k)[0]

        idxs0, idxs1 = self.match(out1['descriptors'], out2['descriptors'], min_cossim=min_cossim)
        return out1['keypoints'][idxs0], out2['keypoints'][idxs1]
        # return out1['keypoints'][idxs0][:sum(out1['scores'][idxs0]>0.2)], out2['keypoints'][idxs1][:sum(out1['scores'][idxs0]>0.2)]

    def get_kpts_heatmap(self, kpts, softmax_temp=1.0):
        scores = self.softmax(kpts, softmax_temp)[:, :64]
        B, _, H, W = scores.shape
        heatmap = scores.transpose(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        heatmap = heatmap.transpose(0, 1, 3, 2, 4).reshape(B, 1, H * 8, W * 8)
        return heatmap

    def NMS(self, x, threshold=0.05, kernel_size=5):
        # 最大值滤波
        pad = kernel_size // 2
        max_filtered = maximum_filter(x, size=(1, 1, kernel_size, kernel_size), mode='constant', cval=-np.inf)

        # 比较和应用阈值
        local_max = (x == max_filtered) & (x > threshold)

        # 查找非零位置
        pos_batched = []
        for i in range(local_max.shape[0]):
            nonzero_coords = np.argwhere(local_max[i])[:, ::-1]  # 注意坐标顺序的反转
            pos_batched.append(nonzero_coords)

        # 处理填充
        pad_val = max(len(x) for x in pos_batched)
        pos = np.zeros((local_max.shape[0], pad_val, 2), dtype=np.int32)

        # 填充kpts并构建(B, N, 2)张量
        for b in range(len(pos_batched)):
            pos[b, :len(pos_batched[b]), :] = pos_batched[b][:, :2]

        return pos.astype(np.float64)

    def match(self, feats1, feats2, min_cossim=0.82):
        # 计算余弦相似度矩阵
        cossim = np.dot(feats1, feats2.T)
        cossim_t = np.dot(feats2, feats1.T)

        # 找到每个特征向量在另一组中的最大余弦相似度对应的索引
        match12 = np.argmax(cossim, axis=1)
        match21 = np.argmax(cossim_t, axis=1)

        # 创建索引数组
        idx0 = np.arange(len(match12))

        # 找到互为最大余弦相似度的索引对
        mutual = match21[match12] == idx0

        if min_cossim > 0:
            # 过滤出余弦相似度大于min_cossim的匹配对
            cossim = np.amax(cossim, axis=1)
            good = cossim > min_cossim
            idx0 = idx0[mutual & good]
            idx1 = match12[mutual & good]
        else:
            # 没有设置最小余弦相似度阈值，直接使用互为最大余弦相似度的匹配对
            idx0 = idx0[mutual]
            idx1 = match12[mutual]

        return idx0, idx1

    def normalize(self, x, axis):
        norms = np.linalg.norm(x, axis=axis, keepdims=True)
        return x / norms

    def softmax(self, x, temp=1.0):
        e_x = np.exp(x / temp - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)


def export_model(infer_type='onnx'):
    if infer_type == 'onnx':
        # from modules.model import *
        torch_model = XFeatModel()  # .to(dev).eval()
        torch_model.load_state_dict(torch.load('weights/xfeat.pt', map_location='cpu'))
        torch_model.eval()  # set the model to inference mode
        batch_size = 1  # 批处理大小
        input_shape = (3, 640, 640)  # 输入数据
        x = torch.randn(batch_size, *input_shape)  # 生成张量
        export_onnx_file = "weights/xfeat.onnx"  # 目的ONNX文件名
        torch.onnx.export(torch_model,
                          x,
                          export_onnx_file,
                          opset_version=12,
                          do_constant_folding=True,  # 是否执行常量折叠优化
                          input_names=["input"],  # 输入名
                          output_names=["output"],  # 输出名
                          dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # 批处理变量
                                        "output": {0: "batch"}})
    else:
        from rknn.api import RKNN
        rknn = RKNN()
        rknn.config(mean_values=[0, 0, 0], std_values=[1, 1, 1], target_platform='rk3588', optimization_level=2)
        rknn.load_onnx(model='weights/xfeat.onnx')
        rknn.build(do_quantization=True, dataset='weights/dataset.txt')
        rknn.export_rknn("weights/xfeat.rknn")


def warp_corners_and_draw_matches(ref_points, dst_points, img1_path, img2_path):
    img1 = cv2.imread(img1_path) if isinstance(img1_path, str) else img1_path
    img2 = cv2.imread(img2_path) if isinstance(img2_path, str) else img2_path
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i - 1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(mask)) if mask[i]]

    # Draw inlier matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 12))
    plt.imshow(img_matches[..., ::-1]), plt.show()

    return img_matches


if __name__ == '__main__':
    x = np.random.random(size=(1, 64, 80, 80))  # np.load('1.npy')
    grids = np.random.random(size=(1, 4096, 2))  # np.load('2.npy')
    mode = "nearest"  #["nearest", "bilinear", "bicubic"]

    output_torch = F.grid_sample(torch.tensor(x), torch.tensor(grids).unsqueeze(-2), mode=mode, align_corners=False)
    output_torch = output_torch.permute(0, 2, 3, 1).squeeze(-2).detach().cpu().numpy()
    output_numpy = grid_sample_numpy(x, grids[None, ], mode=mode, align_corners=False)
    # out_3 = bilinear_grid_sample(torch.tensor(x), torch.tensor(grids).unsqueeze(-2))
    print(f"Difference between NumPy and PyTorch mode <{mode}> results: {np.abs(output_numpy - output_torch).mean()}")

    # export_model()

    # xfeat = XFeat(infer_type="torch")
    # x = cv2.imread('assets/ref.png', cv2.IMREAD_COLOR)
    # x = xfeat.parse_input(x)
    # outputs = xfeat.detectAndCompute(x, top_k=4096)
    # print(len(outputs))

    xfeat = XFeat(infer_type="onnx") # infer_type: ["torch", "onnx", "rknn", "rknn_int8", "rknn_lite"]
    im1 = cv2.imread('assets/ref.png', cv2.IMREAD_COLOR)
    im2 = cv2.imread('assets/tgt.png', cv2.IMREAD_COLOR)
    p1, p2 = xfeat.match_xfeat(im1, im2, top_k=4096)
    print(p1[0], p2[0])
    box_show = warp_corners_and_draw_matches(p1, p2, im1, im2)

    print("finish!")
