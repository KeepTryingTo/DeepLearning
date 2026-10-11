
import torch
import torch.nn.functional as F




def pseudo_nms(fmap, pool_size=3):
    """
    apply max pooling to get the same effect of nms

    Args:
        fmap(Tensor): output tensor of previous step
        pool_size(int): size of max-pooling
    """
    pad = (pool_size-1)//2
    fmap_max = F.max_pool2d(fmap, pool_size, stride=1, padding=pad)
    keep = (fmap_max == fmap).float()
    return fmap * keep