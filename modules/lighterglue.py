
from kornia.feature.lightglue import LightGlue
from torch import nn
import torch
import os

class LighterGlue(nn.Module):
    """
        Lighter version of LightGlue :)
    """

    default_conf_xfeat = {
    "name": "xfeat",  # just for interfacing
    "input_dim": 64,  # input descriptor dimension (autoselected from weights)
    "descriptor_dim": 96,
    "add_scale_ori": False,
    "add_laf": False,  # for KeyNetAffNetHardNet
    "scale_coef": 1.0,  # to compensate for the SIFT scale bigger than KeyNet
    "n_layers": 6,
    "num_heads": 1,
    "flash": True,  # enable FlashAttention if available.
    "mp": False,  # enable mixed precision
    "depth_confidence": -1,  # early stopping, disable with -1
    "width_confidence": 0.95,  # point pruning, disable with -1
    "filter_threshold": 0.1,  # match threshold
    "weights": None,
    }

    def __init__(self, weights = os.path.abspath(os.path.dirname(__file__)) + '/../weights/xfeat-lighterglue.pt'):
        super().__init__()
        LightGlue.default_conf = self.default_conf_xfeat
        self.net = LightGlue(None)
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if os.path.exists(weights):
            state_dict = torch.load(weights, map_location=self.dev)
        else:
            state_dict = torch.hub.load_state_dict_from_url("https://github.com/verlab/accelerated_features/raw/main/weights/xfeat-lighterglue.pt")

        # rename old state dict entries
        for i in range(self.net.conf.n_layers):
            pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            state_dict = {k.replace('matcher.', ''): v for k, v in state_dict.items()}

        self.net.load_state_dict(state_dict, strict=False)
        self.net.to(self.dev)

    @torch.inference_mode()
    def forward(self, data):
        result = self.net( {   'image0': {'keypoints': data['keypoints0'], 'descriptors': data['descriptors0'], 'image_size': data['image_size0']},
                               'image1': {'keypoints': data['keypoints1'], 'descriptors': data['descriptors1'], 'image_size': data['image_size1']}  
                           } )
        return result
