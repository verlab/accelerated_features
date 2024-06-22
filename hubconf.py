dependencies = ['torch']
from modules.xfeat import XFeat as _XFeat
import torch

def XFeat(pretrained=True, top_k=4096, detection_threshold=0.05):
    """
    XFeat model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    weights = None
    if pretrained:
        weights = torch.hub.load_state_dict_from_url("https://github.com/verlab/accelerated_features/raw/main/weights/xfeat.pt")
    
    model = _XFeat(weights, top_k=top_k, detection_threshold=detection_threshold)
    return model
