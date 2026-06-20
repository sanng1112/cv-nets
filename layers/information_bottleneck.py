import argparse
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from layers.base_layer import BaseLayer
from layers.linear_layer import LinearLayer

class InformationBottleneck(BaseLayer):
    """
    Information Bottleneck Layer.
    Inspired by Tishby et al.'s 'Deep Learning and the Information Bottleneck Principle'.
    Compresses the input by projecting it into a stochastic latent space (adding Gaussian noise).
    During training, it outputs the sampled representation and the KL divergence to a standard Normal prior.
    """
    def __init__(self, in_features: int, bottleneck_dim: int, *args, **kwargs) -> None:
        """
        Chi tiết hàm: `__init__`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        super().__init__()
        self.in_features = in_features
        self.bottleneck_dim = bottleneck_dim
        
        self.mu_proj = LinearLayer(in_features, bottleneck_dim, bias=True)
        self.logvar_proj = LinearLayer(in_features, bottleneck_dim, bias=True)
        
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Chi tiết hàm: `add_arguments`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        return parser

    def forward(self, x: Tensor) -> Tensor:
        """
        Chi tiết hàm: `forward`
        - Chức năng: Thực thi logic nội bộ hoặc cung cấp API cho quá trình xử lý.
        - Cảnh báo: Tham số đầu vào cần tuân thủ cấu trúc chuẩn của module.
        """
        mu = self.mu_proj(x)
        logvar = self.logvar_proj(x)
        
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            
            # Compute KL divergence: KL( N(mu, sigma^2) || N(0, 1) )
            # This should be added to the total loss scaled by beta
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
            
            # We attach the kl_div to the output tensor's attributes or return a tuple
            # For standard sequential compatibility, we can just return z
            # The user must extract the kl divergence manually or modify the training loop
            self.kl_loss = kl_div.mean()
            return z
        else:
            return mu
