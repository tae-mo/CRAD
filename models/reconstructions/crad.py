import logging
import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.initializer import initialize_from_cfg

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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class CRAD(nn.Module):
    def __init__(
        self,
        block,
        layers,
        frozen_layers=[],
        groups=1,
        width_per_group=64,
        norm_layer=None,
        initializer=None,
        **kwargs
    ):
        super(CRAD, self).__init__()
        block = globals()[block]
        
        ch = kwargs['inplanes'][0]
        coord_ch = kwargs['inplanes'][0]
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 2*ch
        self.dilation = 1
        self.frozen_layers = frozen_layers
        self.groups = groups
        self.base_width = width_per_group
        
        h, w = kwargs['feature_size']
        self.save_recon = kwargs['save_recon']
        self.mse_lamb = kwargs['mse_lamb']
        self.cos_lamb = kwargs['cos_lamb']
        self.mse_coef = kwargs['mse_coef']
        self.noise_std = kwargs['noise_std']
        
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
            self._make_layer(block, 2*ch, layers[0]),
        )
        self.recover = conv1x1(2*ch, coord_ch)
        
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=8)

        initialize_from_cfg(self, initializer)
    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)
    
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
