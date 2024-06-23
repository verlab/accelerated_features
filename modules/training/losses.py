import torch
import torch.nn.functional as F

from modules.dataset.megadepth import megadepth_warper

from modules.training import utils

from third_party.alike_wrapper import extract_alike_kpts

def dual_softmax_loss(X, Y, temp = 0.2):
    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

    dist_mat = (X @ Y.t()) * temp
    conf_matrix12 = F.log_softmax(dist_mat, dim=1)
    conf_matrix21 = F.log_softmax(dist_mat.t(), dim=1)

    with torch.no_grad():
        conf12 = torch.exp( conf_matrix12 ).max(dim=-1)[0]
        conf21 = torch.exp( conf_matrix21 ).max(dim=-1)[0]
        conf = conf12 * conf21

    target = torch.arange(len(X), device = X.device)

    loss = F.nll_loss(conf_matrix12, target) + \
           F.nll_loss(conf_matrix21, target)

    return loss, conf

def smooth_l1_loss(input, target, beta=2.0, size_average=True):
    diff = torch.abs(input - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss.mean() if size_average else loss.sum()

def fine_loss(f1, f2, pts1, pts2, fine_module, ws=7):
    '''
        Compute Fine features and spatial loss
    '''
    C, H, W = f1.shape
    N = len(pts1)

    #Sort random offsets
    with torch.no_grad():
        a = -(ws//2)
        b = (ws//2)
        offset_gt = (a - b) * torch.rand(N, 2, device = f1.device) + b
        pts2_random = pts2 + offset_gt

    #pdb.set_trace()
    patches1 = utils.crop_patches(f1.unsqueeze(0), (pts1+0.5).long(), size=ws).view(C, N, ws * ws).permute(1, 2, 0) #[N, ws*ws, C]
    patches2 = utils.crop_patches(f2.unsqueeze(0), (pts2_random+0.5).long(), size=ws).view(C, N, ws * ws).permute(1, 2, 0)  #[N, ws*ws, C]

    #Apply transformer
    patches1, patches2 = fine_module(patches1, patches2)

    features = patches1.view(N, ws, ws, C)[:, ws//2, ws//2, :].view(N, 1, 1, C) # [N, 1, 1, C]
    patches2 = patches2.view(N, ws, ws, C) # [N, w, w, C]

    #Dot Product
    heatmap_match = (features * patches2).sum(-1)
    offset_coords = utils.subpix_softmax2d(heatmap_match)

    #Invert offset because center crop inverts it
    offset_gt = -offset_gt 

    #MSE
    error = ((offset_coords - offset_gt)**2).sum(-1).mean()

    #error = smooth_l1_loss(offset_coords, offset_gt)

    return error


def alike_distill_loss(kpts, img):

    C, H, W = kpts.shape
    kpts = kpts.permute(1,2,0) 
    img = img.permute(1,2,0).expand(-1,-1,3).cpu().numpy() * 255

    with torch.no_grad():
        alike_kpts = torch.tensor( extract_alike_kpts(img), device=kpts.device )
        labels = torch.ones((H, W), dtype = torch.long, device = kpts.device) * 64 # -> Default is non-keypoint (bin 64)
        offsets = (((alike_kpts/8) - (alike_kpts/8).long())*8).long()
        offsets =  offsets[:, 0] + 8*offsets[:, 1]  # Linear IDX
        labels[(alike_kpts[:,1]/8).long(), (alike_kpts[:,0]/8).long()] = offsets

    kpts = kpts.view(-1,C)
    labels = labels.view(-1)

    mask = labels < 64
    idxs_pos = mask.nonzero().flatten()
    idxs_neg = (~mask).nonzero().flatten()
    perm = torch.randperm(idxs_neg.size(0))[:len(idxs_pos)//32]
    idxs_neg = idxs_neg[perm]
    idxs = torch.cat([idxs_pos, idxs_neg])

    kpts = kpts[idxs]
    labels = labels[idxs]

    with torch.no_grad():
        predicted = kpts.max(dim=-1)[1]
        acc =  (labels == predicted)
        acc = acc.sum() / len(acc)

    kpts = F.log_softmax(kpts)
    loss = F.nll_loss(kpts, labels, reduction = 'mean')

    return loss, acc


def keypoint_position_loss(kpts1, kpts2, pts1, pts2, softmax_temp = 1.0):
    '''
        Computes coordinate classification loss, by re-interpreting the 64 bins to 8x8 grid and optimizing
        for correct offsets
    '''
    C, H, W = kpts1.shape
    kpts1 = kpts1.permute(1,2,0) * softmax_temp
    kpts2 = kpts2.permute(1,2,0) * softmax_temp

    with torch.no_grad():
        #Generate meshgrid
        x, y = torch.meshgrid(torch.arange(W, device=kpts1.device), torch.arange(H, device=kpts1.device), indexing ='xy')
        xy = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
        xy*=8

        #Generate collision map
        hashmap = torch.ones((H*8, W*8, 2), dtype = torch.long, device = kpts1.device) * -1
        hashmap[(pts1[:,1]).long(), (pts1[:,0]).long(), :] = (pts2).long()

        #Estimate offset of src kpts 
        _, kpts1_offsets = kpts1.max(dim=-1)
        kpts1_offsets_x = kpts1_offsets  % 8
        kpts1_offsets_y = kpts1_offsets // 8
        kpts1_offsets_xy = torch.cat([kpts1_offsets_x.unsqueeze(-1), 
                                      kpts1_offsets_y.unsqueeze(-1)], dim=-1)
        #pdb.set_trace()
        kpts1_coords = xy + kpts1_offsets_xy

        #find src -> tgt pts
        kpts1_coords = kpts1_coords.view(-1,2)
        gt_12 = hashmap[kpts1_coords[:,1], kpts1_coords[:,0]]
        mask_valid = torch.all(gt_12 >= 0, dim=-1)
        gt_12 = gt_12[mask_valid]

        #find offset labels
        labels2 = (gt_12/8) - (gt_12/8).long()
        labels2 = (labels2 * 8).long()
        labels2 = labels2[:, 0] + 8*labels2[:, 1] #linear index
        
    kpts2_selected = kpts2[(gt_12[:, 1]/8).long(), (gt_12[:, 0]/8).long()]        

    kpts1_selected = F.log_softmax(kpts1.view(-1,C)[mask_valid], dim=-1)
    kpts2_selected = F.log_softmax(kpts2_selected, dim=-1)

    #Here we enforce softmax to keep current max on src kps
    with torch.no_grad():
        _, labels1 =  kpts1_selected.max(dim=-1)

    predicted2 = kpts2_selected.max(dim=-1)[1]
    acc =  (labels2 == predicted2)
    acc = acc.sum() / len(acc)

    loss = F.nll_loss(kpts1_selected, labels1, reduction = 'mean') + \
           F.nll_loss(kpts2_selected, labels2, reduction = 'mean')
    
    #pdb.set_trace()

    return loss, acc

def coordinate_classification_loss(coords1, pts1, pts2, conf):
    '''
        Computes the fine coordinate classification loss, by re-interpreting the 64 bins to 8x8 grid and optimizing
        for correct offsets after warp
    '''
    #Do not backprop coordinate warps
    with torch.no_grad():

        coords1_detached = pts1 * 8 

        #find offset
        offsets1_detached = (coords1_detached/8) - (coords1_detached/8).long()
        offsets1_detached = (offsets1_detached * 8).long()
        labels1 = offsets1_detached[:, 0] + 8*offsets1_detached[:, 1]

    #pdb.set_trace()
    coords1_log = F.log_softmax(coords1, dim=-1)

    predicted = coords1.max(dim=-1)[1]
    acc =  (labels1 == predicted)
    acc = acc[conf > 0.1]
    acc = acc.sum() / len(acc)

    loss = F.nll_loss(coords1_log, labels1, reduction = 'none')
    
    #Weight loss by confidence, giving more emphasis on reliable matches
    conf = conf / conf.sum()
    loss = (loss * conf).sum()

    return loss * 2., acc

def keypoint_loss(heatmap, target):
    # Compute L1 loss
    L1_loss = F.l1_loss(heatmap, target)
    return L1_loss * 3.0

def hard_triplet_loss(X,Y, margin = 0.5):

    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

    dist_mat = torch.cdist(X, Y, p=2.0)
    dist_pos = torch.diag(dist_mat)
    dist_neg = dist_mat + 100.*torch.eye(*dist_mat.size(), dtype = dist_mat.dtype, 
            device = dist_mat.get_device() if dist_mat.is_cuda else torch.device("cpu"))

    #filter repeated patches on negative distances to avoid weird stuff on gradients
    dist_neg = dist_neg + dist_neg.le(0.01).float()*100.

    #Margin Ranking Loss
    hard_neg = torch.min(dist_neg, 1)[0]

    loss = torch.clamp(margin + dist_pos - hard_neg, min=0.)

    return loss.mean()
