import logging
import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.initializer import initialize_from_cfg
from models.reconstructions.mobilenetv3 import MobileBottleneck

logger = logging.getLogger("global_logger")

__all__ = [
    "CRAD"
]

def conv3x3(inplanes, outplanes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        inplanes,
        outplanes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(inplanes, outplanes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False)


class CRAD(nn.Module):
    def __init__(
        self,
        layers,
        frozen_layers=[],
        initializer=None,
        **kwargs
    ):
        super(CRAD, self).__init__()
        
        ch = kwargs['inplanes'][0]
        coord_ch = kwargs['inplanes'][0]
        
        self.frozen_layers = frozen_layers
        
        h, w = kwargs['feature_size']
        self.save_recon = kwargs['save_recon']
        self.mse_lamb = kwargs['mse_lamb']
        self.cos_lamb = kwargs['cos_lamb']
        self.mse_coef = kwargs['mse_coef']
        self.noise_std = kwargs['noise_std']
        ch_exp = kwargs['ch_exp']
        
        self.query = nn.ParameterList([
            nn.Parameter(nn.init.xavier_normal_(torch.empty(1, ch, kwargs['local_resol'], kwargs['local_resol']))), 
            nn.Parameter(nn.init.xavier_normal_(torch.empty(1, ch*h*w, kwargs['global_resol'], kwargs['global_resol'])))
        ])
        
        self.coord_ = nn.Sequential(
            nn.Linear(coord_ch, coord_ch), 
            nn.Tanh(), 
            nn.Linear(coord_ch,2), 
            nn.Tanh()
        )
        self.coord = nn.Sequential(
            nn.Conv2d(coord_ch,coord_ch,1,1,0), 
            nn.Tanh(), 
            nn.Conv2d(coord_ch,2,1,1,0), 
            nn.Tanh()
        )

        self.layer = nn.Sequential(
            *[MobileBottleneck(432, 432, 3, 1, ch_exp, True, "HS") for _ in range(layers)]
        )        
        self.recover = conv1x1(2*ch, coord_ch)
        
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=8)

        initialize_from_cfg(self, initializer)
    
    def forward(self, input):
        feature_align = input["feature_align"]
        B, C, H, W = feature_align.shape
        
        coord_ = self.coord_(nn.functional.adaptive_avg_pool2d(feature_align, (1, 1)).view(B,-1))
        coord = self.coord(feature_align)

        add_noise = True if torch.rand(1).item() < 0.5 else False
        if self.training and add_noise: coord = coord + torch.randn(B,2,H,W, device=coord.device)*self.noise_std

        query = torch.cat([
            F.grid_sample(self.query[1], coord_.view(1,B,1,2), align_corners=False).permute(2,3,0,1).view(B,-1,H,W), 
            F.grid_sample(self.query[0].repeat(B,1,1,1), coord.permute(0,2,3,1), align_corners=False)
        ], dim=1)
        
        feature_rec = self.layer(query)
        feature_rec = self.recover(feature_rec)
        
        if not self.training and input.get('save_recon', False):
            clsnames = input["clsname"]
            filenames = input["filename"]
            for clsname, filename, feat_rec in zip(clsnames, filenames, feature_rec):
                filedir, filename = os.path.split(filename)
                _, defename = os.path.split(filedir)
                filename_, _ = os.path.splitext(filename)
                save_dir = os.path.join(self.save_recon.save_dir, clsname, defename)
                os.makedirs(save_dir, exist_ok=True)
                feature_rec_np = feat_rec.detach().cpu().numpy()
                np.save(os.path.join(save_dir, filename_ + ".npy"), feature_rec_np)
        
        mse = torch.mean((feature_rec - feature_align)**2, dim=1)
        mse = 1 - torch.round((mse*self.mse_coef).clamp(0,1))
        cos = F.cosine_similarity(feature_rec, feature_align, dim=1)
        
        sim = (self.mse_lamb*mse + self.cos_lamb*cos).unsqueeze(1)
        feature_rec = sim*feature_align + (1-sim)*feature_rec

        pred = torch.sqrt(
            torch.sum((feature_rec - feature_align)**2, dim=1, keepdim=True)
        )
        pred = self.upsample(pred)
        
        return {
            'feature_rec': feature_rec,
            'feature_align': feature_align,
            'pred': pred
        }
