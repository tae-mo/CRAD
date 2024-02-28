import copy
import importlib
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc_helper import to_device


class ModelHelper(nn.Module):
    """Build model from cfg"""

    def __init__(self, cfg):
        super(ModelHelper, self).__init__()

        self.frozen_layers = []
        for cfg_subnet in cfg:
            mname = cfg_subnet["name"]
            kwargs = cfg_subnet["kwargs"]
            mtype = cfg_subnet["type"]
            if cfg_subnet.get("frozen", False):
                self.frozen_layers.append(mname)
            if cfg_subnet.get("prev", None) is not None:
                prev_module = getattr(self, cfg_subnet["prev"])
                kwargs["inplanes"] = prev_module.get_outplanes()
                kwargs["instrides"] = prev_module.get_outstrides()

            module = self.build(mtype, kwargs)
            self.add_module(mname, module)

        self.times = defaultdict(list)
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def build(self, mtype, kwargs):
        module_name, cls_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name) # backbones/__init__.py
        cls = getattr(module, cls_name)
        return cls(**kwargs)

    def cuda(self):
        self.device = torch.device("cuda")
        return super(ModelHelper, self).cuda()

    def cpu(self):
        self.device = torch.device("cpu")
        return super(ModelHelper, self).cpu()

    def forward(self, input):
        input = copy.copy(input)
        if input["image"].device != self.device:
            input = to_device(input, device=self.device)

        if input.get('eval_mode', False):
            for submodule in self.children():
                self.start.record()
                output = submodule(input)
                
                torch.cuda.synchronize()
                self.end.record()
                torch.cuda.synchronize()
                self.times[f'{submodule.__class__.__name__}'].append(self.start.elapsed_time(self.end))
                input.update(output)
            input['time'] = self.times
        else:
            for submodule in self.children():
                # print(f"{submodule.__class__.__name__} params: {sum(param.numel() for param in submodule.parameters())}")
                output = submodule(input)
                input.update(output)
            # exit()
        
        return input

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
    
    def unfreeze_parameter(self, module):
        module.train()
        for param in module.parameters():
            param.requires_grad = True

    def train(self, mode=True):
        """
        Sets the module in training mode.
        This has any effect only on modules such as Dropout or BatchNorm.

        Returns:
            Module: self
        """
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self