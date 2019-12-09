""" VGG16 network Class
Adapted from Gurkirt Singh's code: https://github.com/gurkirt/realtime-action-detection/blob/master/ssd.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
# adv
from models.grad_reversal import ReverseLayerF
import pdb

class MLP_Block(nn.Module):
    def __init__(self, inplanes, planes):
        super(MLP_Block, self).__init__()
        self.fc = nn.Linear(inplanes, planes)
        # self.bn = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.fc(x)
        # out = self.bn(out)
        out = self.relu(out)

        return out

class VGG16(nn.Module):
    def __init__(self, 
                base, 
                num_classes,
                is_adv=False,   
                is_human_mask_adv=False,   
                alpha=0.0,
                alpha_hm=0.0,
                num_places_classes=365,                 
                num_place_hidden_layers=1,
                num_human_mask_adv_hidden_layers=1):
        super(VGG16, self).__init__()
        self.num_classes = num_classes
        self.size = 300
        self.is_adv = is_adv        
        self.is_human_mask_adv = is_human_mask_adv
        self.alpha = alpha
        self.alpha_hm = alpha_hm
        self.num_places_classes = num_places_classes

        self.vgg = nn.ModuleList(base)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.mlp = nn.Sequential(
            nn.Linear(1024 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),            
        )
        self.fc = nn.Linear(4096, self.num_classes)

        # human mask adv
        if self.is_human_mask_adv:
            self.hm_mlp = nn.Sequential()
            self.hm_mlp = self._make_mlp_layer(MLP_Block, 4096, 4096, num_human_mask_adv_hidden_layers)
            self.hm_mlp.add_module('hm_last_fc', nn.Linear(4096, num_classes))

        # adv
        if self.is_adv:
            self.place_mlp = nn.Sequential()            
            self.place_mlp = self._make_mlp_layer(MLP_Block, 4096, 4096, num_place_hidden_layers)
            self.place_mlp.add_module('p_last_fc', nn.Linear(4096, self.num_places_classes))

    def forward(self, x):
        for k in range(len(self.vgg)):
            x = self.vgg[k](x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)

        # adv
        if self.is_human_mask_adv:
            rev_x_hm = ReverseLayerF.apply(x, self.alpha_hm)
        if self.is_adv:            
            rev_x = ReverseLayerF.apply(x, self.alpha)                    
            dom_x = self.place_mlp(rev_x)

        x = self.fc(x)

        if self.is_human_mask_adv and self.is_adv:
            hm_x = self.hm_mlp(rev_x_hm)
            return x, dom_x, hm_x
        elif self.is_adv:        
            return x, dom_x
        elif self.is_human_mask_adv:
            hm_x = self.hm_mlp(rev_x_hm)
            return x, hm_x
        else:            
            return x

    def _make_mlp_layer(self, block, inplanes, planes, blocks):        
        layers = []
        layers.append(block(inplanes, planes))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)
        
    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}

def build_vgg(**kwargs):
# def build_vgg(size=300, num_classes=24):
#     if size != 300:
#         print("Error: Sorry only SSD300 is supported currently!")
#         return
    model = VGG16(vgg(base['300'], 3), **kwargs)
    return model

def get_fine_tuning_parameters(model, ft_begin_index):    
    if ft_begin_index > 0:
        print('Finetuing only partial layers is not supported')
        return         

    ft_module_names, new_module_names = [], []
    ft_module_names.append('vgg')

    new_module_names.append('mlp')
    new_module_names.append('fc')

    pretrained_parameters, new_parameters = [], []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                print('finetune params:{}'.format(k))
                pretrained_parameters.append(v)
                break
        else:
            for new_module in new_module_names:
                if new_module in k:
                    print('new params:{}'.format(k))                    
                    new_parameters.append(v)
                    break                
   
    return [pretrained_parameters, new_parameters]

def get_adv_fine_tuning_parameters(model, ft_begin_index, new_layer_lr, not_replace_last_fc=False, is_human_mask_adv=False, slower_place_mlp=False, slower_hm_mlp=False):
    if ft_begin_index > 0:
        print('Finetuing only partial layers is not supported')
        return 

    ft_module_names, new_module_names = [], []
    ft_module_names.append('vgg')

    new_module_names.append('mlp')
    if not slower_place_mlp:
        new_module_names.append('place_mlp')
    else:
        ft_module_names.append('place_mlp')
    if is_human_mask_adv:
        if not slower_hm_mlp:
            new_module_names.append('hm_mlp')
        else:
            ft_module_names.append('hm_mlp')
    if not not_replace_last_fc:
        new_module_names.append('fc')

    
    pretrained_parameters, new_parameters = [], []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                print('finetune params:{}'.format(k))
                pretrained_parameters.append(v)
                break
        else:                
            for new_module in new_module_names:
                if new_module in k:
                    print('new params:{}'.format(k))                    
                    new_parameters.append(v)
                    break
            
    return [pretrained_parameters, new_parameters]        
    