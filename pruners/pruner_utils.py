import numpy as np
import torch

COMBINATIONS = np.array(np.meshgrid(np.arange(start=-1, stop=2), np.arange(start=-1, stop=2))).T.reshape(-1, 2)

def get_distance_mask(source, target, batch_size, percent):

    dist = (torch.sum(source, dim=0) / batch_size) - (torch.sum(target, dim=0) / batch_size)
    nz = dist.detach().numpy()[np.nonzero(dist.detach().numpy())]
    perc_value = np.percentile(abs(nz), (100 - percent))
    dist_mask = np.where((abs(dist) > perc_value), 1, 0)
    source_masked, target_masked = (source.detach() * dist_mask), (target.detach() * dist_mask)

    return source_masked, target_masked

def simplify_locs(locs_src, base_src):
    
    lim = base_src.shape[1]
    
    locs_src[0], locs_src[1], locs_src[2] = locs_src[0][locs_src[1] != 0], locs_src[1][locs_src[1] != 0], \
                                                         locs_src[2][locs_src[1] != 0]
    locs_src[0], locs_src[1], locs_src[2] = locs_src[0][locs_src[1] != lim - 1], locs_src[1][locs_src[1] != lim - 1], \
                                            locs_src[2][locs_src[1] != lim - 1]
    locs_src[0], locs_src[1], locs_src[2] = locs_src[0][locs_src[0] != 0], locs_src[1][locs_src[0] != 0], \
                                            locs_src[2][locs_src[0] != 0]
    locs_src[0], locs_src[1], locs_src[2] = locs_src[0][locs_src[0] != base_src.shape[2] - 1] ,locs_src[1][locs_src[0] != lim - 1],\
                                            locs_src[2][locs_src[0] != lim - 1]
    
    return locs_src
