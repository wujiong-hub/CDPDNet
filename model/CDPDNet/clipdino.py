import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model


class DinoEncoder(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        self.dinov2 = Dinov2Model.from_pretrained("facebook/dinov2-base")
        self.embed_dim = 768
        self.patch_size = 14

    def get_dinofeatures(self, images):
        model = self.dinov2
        model.eval()
        h, w, d = self.img_size[0], (self.img_size[1] // self.patch_size + 1) * self.patch_size, (self.img_size[2] // self.patch_size + 1) * self.patch_size
        images_resized = F.interpolate(images, size=(h, w, d), mode="trilinear", align_corners=False)

        batch_size, _, h, w, d = images_resized.shape  # [B, 1, H, W, D]
        slices = images_resized.permute(0, 2, 1, 3, 4).reshape(-1, 1, w, d)  # [B*D, 1, H, W]
        slices_rgb = slices.expand(-1, 3, -1, -1)  # [B*D, 3, H, W]
        with torch.no_grad():
            outputs = model(slices_rgb, output_hidden_states=True)  # Output hidden states from all layers
            hidden_states = outputs.hidden_states  # This is a tuple of hidden states from all layers

        # Get the last four layers
        last_four_layers = hidden_states[-4:]  # Select last four layers

        num_patches_w = w // self.patch_size
        num_patches_d = d // self.patch_size

        # Reorganize the last four layers into a tuple
        last_four_layers_reshaped = tuple(
            layer[:,1:].view(batch_size, h, num_patches_w, num_patches_d, self.embed_dim).permute(0, 4, 1, 2, 3)
            for layer in last_four_layers
        )

        return last_four_layers_reshaped

    def forward(self, x):
        return self.get_dinofeatures(x)


class Adaptor(nn.Module):
    def __init__(self, out_channels, embed_dim=768, num_slice=14):
        super().__init__()
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.num_slice = num_slice

        self.enc = nn.Sequential(
            nn.Conv3d(in_channels=self.embed_dim, out_channels=self.embed_dim,
                                         kernel_size=(self.num_slice,1,1), stride=(self.num_slice,1,1),
                                         groups=self.embed_dim),
            nn.Conv3d(in_channels=self.embed_dim, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )  

    def forward(self, x):
        return self.enc(x)

class Alignment(nn.Module):
    def __init__(self, feature_dim, text_dim):
        super().__init__()
        self.gamma = nn.Linear(text_dim, feature_dim)
        nn.init.xavier_uniform_(self.gamma.weight)  # Initialize gamma weights with Xavier uniform
        self.beta = nn.Linear(text_dim, feature_dim)
        nn.init.xavier_uniform_(self.beta.weight)  # Initialize beta weights with Xavier uniform

    def forward(self, feature, text):
        bt, ct = text.shape
        text = text.view(1, bt*ct)

        b, c, d, h, w = feature.shape
        feature_flat = feature.view(b, c, -1) #shape(B,C,H*W*D)

        gamma = self.gamma(text).unsqueeze(-1) #shape(B,C,1)
        beta = self.beta(text).unsqueeze(-1) #shape(B,C,1) this is broadcastable.
        
        modulated = (gamma * feature_flat) + beta
        return modulated.view(b, c, d, h, w)

