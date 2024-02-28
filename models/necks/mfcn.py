import torch
import torch.nn as nn
import torch.nn.functional as F

# MFCN: multi-scale feature concat network
__all__ = ["MFCN"]

# Wide Resnet
class MFCN(nn.Module):
    def __init__(self, inplanes, outplanes, instrides, outstrides):
        super(MFCN, self).__init__()

        assert isinstance(inplanes, list)
        assert isinstance(outplanes, list) and len(outplanes) == 1
        assert isinstance(outstrides, list) and len(outstrides) == 1
        assert outplanes[0] == sum(inplanes)  # concat
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.instrides = instrides
        self.outstrides = outstrides
        self.scale_factors = [instride // outstrides[0] for instride in instrides]
        self.upsample_list = [
            nn.UpsamplingBilinear2d(scale_factor=scale_factor)
            for scale_factor in self.scale_factors
        ]

    def forward(self, input):
        features = input["features"]
        assert len(self.inplanes) == len(features)

        feature_list = []
        # resize & concatenates
        for i in range(len(features)):
            upsample = self.upsample_list[i]
            feature_resize = upsample(features[i])
            
            feature_list.append(feature_resize)
        feature_align = torch.cat(feature_list, dim=1)
        
        return {"feature_align": feature_align, 
                "outplane": self.get_outplanes(),}

    def get_outplanes(self):
        return self.outplanes

    def get_outstrides(self):
        return self.outstrides
