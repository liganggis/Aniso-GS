import torch
import torch.nn as nn
import torch.nn.functional as F

def get_encoder(encoding, input_dim=3, 
                multires=6, # freq
                degree=4, # SH
                num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=2048, # hash/tiled grid
                align_corners=False, interpolation='linear', # grid
                **kwargs):

    if encoding == 'None':
        return lambda x, **kwargs: x, input_dim

    elif encoding == 'sh':
        from submodules.shencoder import SHEncoder
        encoder = SHEncoder(input_dim=input_dim, degree=degree)

    elif encoding == 'hashgrid':
        from submodules.gridencoder import GridEncoder
        encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='hash', align_corners=align_corners, interpolation=interpolation)
    
    elif encoding == 'tiledgrid':
        from submodules.gridencoder import GridEncoder
        encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='tiled', align_corners=align_corners, interpolation=interpolation)
    
    else:
        raise NotImplementedError('Unknown encoding mode, choose from [None, frequency, sh, hashgrid, tiledgrid]')

    return encoder, encoder.output_dim