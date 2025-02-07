# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
from typing import Union

from .ema import EMA
from .unet import UNetModel
from .p_unet import PUnet
import pdb
import torch
import os


MODEL_CONFIGS = {
    "unet": {
        "in_channels": 3,
        "model_channels": 192,
        "out_channels": 3,
        "num_res_blocks": 3,
        "attention_resolutions": [2, 4, 8],
        "dropout": 0.1,
        "channel_mult": [1, 2, 3, 4],
        "num_classes": 1000,
        "use_checkpoint": False,
        "num_heads": 4,
        "num_head_channels": 64,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "unet128": {
        "in_channels": 3,
        "model_channels": 192,
        "out_channels": 3,
        "num_res_blocks": 3,
        "attention_resolutions": [2, 4, 8],
        "dropout": 0.1,
        "channel_mult": [1, 2, 3, 4],
        "num_classes": 1000,
        "use_checkpoint": False,
        "num_heads": 4,
        "num_head_channels": 64,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "unet256": {
        "in_channels": 3,
        "model_channels": 32,
        "out_channels": 3,
        "num_res_blocks": 6,
        "attention_resolutions": [16,8],
        "dropout": 0.1,
        "channel_mult": [1, 2, 3, 4],
        "num_classes": 1000,
        "use_checkpoint": False,
        "num_heads": 4,
        "num_head_channels": 64,
        "use_scale_shift_norm": True,
        "resblock_updown": True,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "punet256": {
        'input_channels': 3,
        'input_height': 256,
        'ch': 32,
        'ch_mult': (1, 2, 4, 8),
        'num_res_blocks': 6,
        'attn_resolutions': (32,16, 8),
        'resamp_with_conv': True,
    },
    "punet128": {
        'input_channels': 3,
        'input_height': 128,
        'ch': 32,
        'ch_mult': (1, 2, 4, 8),
        'num_res_blocks': 6,
        'attn_resolutions': (16, 8),
        'resamp_with_conv': True,
    },
    "punet64": {
        'input_channels': 3,
        'input_height': 64,
        'ch': 32,
        'ch_mult': (1, 2, 4, 8),
        'num_res_blocks': 6,
        'attn_resolutions': (8, 4, 2),
        'resamp_with_conv': True,
    },
}


def instantiate_model(
    architechture: str, use_ema: bool=False
) -> Union[UNetModel, PUnet]:
    assert (
        architechture in MODEL_CONFIGS
    ), f"Model architecture {architechture} is missing its config."
    if architechture.startswith('punet'):
        model = PUnet(**MODEL_CONFIGS[architechture])
    else:   
        model = UNetModel(**MODEL_CONFIGS[architechture])

    if use_ema:
        return EMA(model=model)
    else:
        return model
if __name__=='__main__':
    from tqdm import tqdm
    model = instantiate_model('punet256',False).cuda()
    for i in tqdm(range(100)):
        time = torch.rand(32).cuda()

        data = torch.randn(32,3,256,256).cuda()
        res = model(time,data)
        res.sum().backward()
     
    print(model)