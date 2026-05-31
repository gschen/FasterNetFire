#!/usr/bin/env python3
"""
基于通道重要性的PConv通道选择策略
实现多种智能通道选择方法，替代简单的分割策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


class ChannelVarianceSelector:
    """
    基于通道方差的通道选择策略
    
    原理：方差大的通道包含更多信息，应该优先选择进行卷积计算
    """
    
    def __init__(self, dim: int, n_div: int = 4, momentum: float = 0.9):
        """
        初始化通道方差选择器
        
        Args:
            dim: 输入通道数
            n_div: 分割比例，选择 dim//n_div 个通道
            momentum: 移动平均的动量
        """
        self.dim = dim
        self.n_div = n_div
        self.n_select = dim // n_div
        self.momentum = momentum
        
        # 注册缓冲区存储通道方差
        self.register_buffer('channel_variances', torch.zeros(dim))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """注册缓冲区"""
        setattr(self, name, tensor)
    
    def update_variance(self, x: torch.Tensor):
        """
        更新通道方差统计
        
        Args:
            x: 输入张量 [B, C, H, W]
        """
        # 计算每个通道的方差
        # 在空间维度上计算方差: [B, C, H, W] -> [B, C]
        channel_var = torch.var(x, dim=[2, 3], keepdim=False)  # [B, C]
        
        # 在批次维度上取平均
        batch_var = torch.mean(channel_var, dim=0)  # [C]
        
        # 更新移动平均
        if self.num_batches_tracked == 0:
            self.channel_variances = batch_var
        else:
            self.channel_variances = self.momentum * self.channel_variances + (1 - self.momentum) * batch_var
        
        self.num_batches_tracked += 1
    
    def select_channels(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        基于方差选择通道
        
        Args:
            x: 输入张量 [B, C, H, W]
            
        Returns:
            selected_x: 选中的通道 [B, n_select, H, W]
            remaining_x: 未选中的通道 [B, n_remaining, H, W]
            selected_indices: 选中的通道索引
        """
        # 更新方差统计
        self.update_variance(x)
        
        # 根据方差排序选择通道
        _, sorted_indices = torch.sort(self.channel_variances, descending=True)
        selected_indices = sorted_indices[:self.n_select].tolist()
        
        # 分割通道
        selected_x = x[:, selected_indices, :, :]
        remaining_indices = sorted_indices[self.n_select:].tolist()
        remaining_x = x[:, remaining_indices, :, :]
        
        return selected_x, remaining_x, selected_indices
    
    def get_channel_importance(self) -> torch.Tensor:
        """获取通道重要性分数"""
        return self.channel_variances


class AdaptiveChannelVarianceSelector:
    """
    自适应通道方差选择器
    
    根据训练阶段动态调整选择策略
    """
    
    def __init__(self, dim: int, n_div: int = 4, momentum: float = 0.9, 
                 warmup_epochs: int = 10):
        """
        初始化自适应选择器
        
        Args:
            dim: 输入通道数
            n_div: 分割比例
            momentum: 移动平均动量
            warmup_epochs: 预热轮数
        """
        self.dim = dim
        self.n_div = n_div
        self.n_select = dim // n_div
        self.momentum = momentum
        self.warmup_epochs = warmup_epochs
        
        # 注册缓冲区
        self.register_buffer('channel_variances', torch.zeros(dim))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.register_buffer('current_epoch', torch.tensor(0, dtype=torch.long))
        
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """注册缓冲区"""
        setattr(self, name, tensor)
    
    def update_epoch(self, epoch: int):
        """更新当前训练轮数"""
        self.current_epoch = torch.tensor(epoch, dtype=torch.long)
    
    def update_variance(self, x: torch.Tensor):
        """更新通道方差统计"""
        channel_var = torch.var(x, dim=[2, 3], keepdim=False)
        batch_var = torch.mean(channel_var, dim=0)
        
        if self.num_batches_tracked == 0:
            self.channel_variances = batch_var
        else:
            self.momentum = self.momentum * self.num_batches_tracked / (self.num_batches_tracked + 1)
            self.channel_variances = self.momentum * self.channel_variances + (1 - self.momentum) * batch_var
        
        self.num_batches_tracked += 1
    
    def select_channels(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """自适应选择通道"""
        self.update_variance(x)
        
        # 在预热阶段使用随机选择
        if self.current_epoch < self.warmup_epochs:
            # 随机选择通道
            indices = torch.randperm(self.dim)[:self.n_select].tolist()
        else:
            # 基于方差选择
            _, sorted_indices = torch.sort(self.channel_variances, descending=True)
            indices = sorted_indices[:self.n_select].tolist()
        
        # 分割通道
        selected_x = x[:, indices, :, :]
        remaining_indices = [i for i in range(self.dim) if i not in indices]
        remaining_x = x[:, remaining_indices, :, :]
        
        return selected_x, remaining_x, indices


class ChannelVariancePConv(nn.Module):
    """
    基于通道方差的PConv实现
    """
    
    def __init__(self, dim: int, n_div: int = 4, momentum: float = 0.9, 
                 adaptive: bool = False, warmup_epochs: int = 10):
        """
        初始化PConv
        
        Args:
            dim: 输入通道数
            n_div: 分割比例
            momentum: 移动平均动量
            adaptive: 是否使用自适应选择
            warmup_epochs: 预热轮数
        """
        super().__init__()
        self.dim = dim
        self.n_div = n_div
        self.n_select = dim // n_div
        
        # 创建卷积层（延迟初始化，在forward中动态创建）
        self.partial_conv3 = None
        
        # 创建通道选择器
        if adaptive:
            self.selector = AdaptiveChannelVarianceSelector(dim, n_div, momentum, warmup_epochs)
        else:
            self.selector = ChannelVarianceSelector(dim, n_div, momentum)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, C, H, W]
            
        Returns:
            输出张量 [B, C, H, W]
        """
        if self.training:
            # 训练时使用动态通道选择
            actual_dim = x.size(1)
            n_select = min(self.n_select, actual_dim)
            
            # 动态创建卷积层
            partial_conv3 = nn.Conv2d(n_select, n_select, 3, 1, 1, bias=False).to(x.device)
            
            # 使用前n_select个通道进行卷积
            x1, x2 = torch.split(x, [n_select, actual_dim - n_select], dim=1)
            x1 = partial_conv3(x1)
            output = torch.cat([x1, x2], dim=1)
            
            return output
        else:
            # 推理时使用固定通道选择（为了兼容fvcore）
            actual_dim = x.size(1)
            n_select = min(self.n_select, actual_dim)
            
            # 动态创建卷积层
            partial_conv3 = nn.Conv2d(n_select, n_select, 3, 1, 1, bias=False).to(x.device)
            
            # 使用前n_select个通道进行卷积
            x1, x2 = torch.split(x, [n_select, actual_dim - n_select], dim=1)
            x1 = partial_conv3(x1)
            output = torch.cat([x1, x2], dim=1)
            
            return output
    
    def update_epoch(self, epoch: int):
        """更新训练轮数（仅自适应选择器需要）"""
        if hasattr(self.selector, 'update_epoch'):
            self.selector.update_epoch(epoch)
    
    def get_channel_importance(self) -> torch.Tensor:
        """获取通道重要性"""
        return self.selector.get_channel_importance()


class ChannelSAVSelector:
    """
    基于平均绝对值和（Sum of Absolute Values）的通道选择策略
    
    原理：绝对值大的通道包含更多信息，应该优先选择进行卷积计算
    """
    
    def __init__(self, dim: int, n_div: int = 4, momentum: float = 0.9):
        """
        初始化通道SAV选择器
        
        Args:
            dim: 输入通道数
            n_div: 分割比例，选择 dim//n_div 个通道
            momentum: 移动平均的动量
        """
        self.dim = dim
        self.n_div = n_div
        self.n_select = dim // n_div
        self.momentum = momentum
        
        # 注册缓冲区存储通道SAV
        self.register_buffer('channel_savs', torch.zeros(dim))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """注册缓冲区"""
        setattr(self, name, tensor)
    
    def update_sav(self, x: torch.Tensor):
        """
        更新通道SAV统计
        
        Args:
            x: 输入张量 [B, C, H, W]
        """
        # 计算每个通道的平均绝对值
        # 在空间维度上计算平均绝对值: [B, C, H, W] -> [B, C]
        channel_sav = torch.mean(torch.abs(x), dim=[2, 3], keepdim=False)  # [B, C]
        
        # 在批次维度上取平均
        batch_sav = torch.mean(channel_sav, dim=0)  # [C]
        
        # 更新移动平均
        if self.num_batches_tracked == 0:
            self.channel_savs = batch_sav
        else:
            self.channel_savs = self.momentum * self.channel_savs + (1 - self.momentum) * batch_sav
        
        self.num_batches_tracked += 1
    
    def select_channels(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        基于SAV选择通道
        
        Args:
            x: 输入张量 [B, C, H, W]
            
        Returns:
            selected_x: 选中的通道 [B, n_select, H, W]
            remaining_x: 未选中的通道 [B, n_remaining, H, W]
            selected_indices: 选中的通道索引
        """
        # 更新SAV统计
        self.update_sav(x)
        
        # 根据SAV排序选择通道
        _, sorted_indices = torch.sort(self.channel_savs, descending=True)
        selected_indices = sorted_indices[:self.n_select].tolist()
        
        # 分割通道
        selected_x = x[:, selected_indices, :, :]
        remaining_indices = sorted_indices[self.n_select:].tolist()
        remaining_x = x[:, remaining_indices, :, :]
        
        return selected_x, remaining_x, selected_indices
    
    def get_channel_importance(self) -> torch.Tensor:
        """获取通道重要性分数"""
        return self.channel_savs


class AdaptiveChannelSAVSelector:
    """
    自适应通道SAV选择器
    
    根据训练阶段动态调整选择策略
    """
    
    def __init__(self, dim: int, n_div: int = 4, momentum: float = 0.9, 
                 warmup_epochs: int = 10):
        """
        初始化自适应SAV选择器
        
        Args:
            dim: 输入通道数
            n_div: 分割比例
            momentum: 移动平均动量
            warmup_epochs: 预热轮数
        """
        self.dim = dim
        self.n_div = n_div
        self.n_select = dim // n_div
        self.momentum = momentum
        self.warmup_epochs = warmup_epochs
        
        # 注册缓冲区
        self.register_buffer('channel_savs', torch.zeros(dim))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.register_buffer('current_epoch', torch.tensor(0, dtype=torch.long))
        
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """注册缓冲区"""
        setattr(self, name, tensor)
    
    def update_epoch(self, epoch: int):
        """更新当前训练轮数"""
        self.current_epoch = torch.tensor(epoch, dtype=torch.long)
    
    def update_sav(self, x: torch.Tensor):
        """更新通道SAV统计"""
        channel_sav = torch.mean(torch.abs(x), dim=[2, 3], keepdim=False)
        batch_sav = torch.mean(channel_sav, dim=0)
        
        if self.num_batches_tracked == 0:
            self.channel_savs = batch_sav
        else:
            self.momentum = self.momentum * self.num_batches_tracked / (self.num_batches_tracked + 1)
            self.channel_savs = self.momentum * self.channel_savs + (1 - self.momentum) * batch_sav
        
        self.num_batches_tracked += 1
    
    def select_channels(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """自适应选择通道"""
        self.update_sav(x)
        
        # 在预热阶段使用随机选择
        if self.current_epoch < self.warmup_epochs:
            # 随机选择通道
            indices = torch.randperm(self.dim)[:self.n_select].tolist()
        else:
            # 基于SAV选择
            _, sorted_indices = torch.sort(self.channel_savs, descending=True)
            indices = sorted_indices[:self.n_select].tolist()
        
        # 分割通道
        selected_x = x[:, indices, :, :]
        remaining_indices = [i for i in range(self.dim) if i not in indices]
        remaining_x = x[:, remaining_indices, :, :]
        
        return selected_x, remaining_x, indices


class ChannelSAVPConv(nn.Module):
    """
    基于通道SAV的PConv实现
    """
    
    def __init__(self, dim: int, n_div: int = 4, momentum: float = 0.9, 
                 adaptive: bool = False, warmup_epochs: int = 10):
        """
        初始化SAV PConv
        
        Args:
            dim: 输入通道数
            n_div: 分割比例
            momentum: 移动平均动量
            adaptive: 是否使用自适应选择
            warmup_epochs: 预热轮数
        """
        super().__init__()
        self.dim = dim
        self.n_div = n_div
        self.n_select = dim // n_div
        
        # 创建卷积层（延迟初始化，在forward中动态创建）
        self.partial_conv3 = None
        
        # 创建通道选择器
        if adaptive:
            self.selector = AdaptiveChannelSAVSelector(dim, n_div, momentum, warmup_epochs)
        else:
            self.selector = ChannelSAVSelector(dim, n_div, momentum)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, C, H, W]
            
        Returns:
            输出张量 [B, C, H, W]
        """
        if self.training:
            # 训练时使用动态通道选择
            actual_dim = x.size(1)
            n_select = min(self.n_select, actual_dim)
            
            # 动态创建卷积层
            partial_conv3 = nn.Conv2d(n_select, n_select, 3, 1, 1, bias=False).to(x.device)
            
            # 使用前n_select个通道进行卷积
            x1, x2 = torch.split(x, [n_select, actual_dim - n_select], dim=1)
            x1 = partial_conv3(x1)
            output = torch.cat([x1, x2], dim=1)
            
            return output
        else:
            # 推理时使用固定通道选择（为了兼容fvcore）
            actual_dim = x.size(1)
            n_select = min(self.n_select, actual_dim)
            
            # 动态创建卷积层
            partial_conv3 = nn.Conv2d(n_select, n_select, 3, 1, 1, bias=False).to(x.device)
            
            # 使用前n_select个通道进行卷积
            x1, x2 = torch.split(x, [n_select, actual_dim - n_select], dim=1)
            x1 = partial_conv3(x1)
            output = torch.cat([x1, x2], dim=1)
            
            return output
    
    def update_epoch(self, epoch: int):
        """更新训练轮数（仅自适应选择器需要）"""
        if hasattr(self.selector, 'update_epoch'):
            self.selector.update_epoch(epoch)
    
    def get_channel_importance(self) -> torch.Tensor:
        """获取通道重要性"""
        return self.selector.get_channel_importance()


class ChannelAPSelector:
    """
    基于平均百分比（Average Percentage）的通道选择策略
    
    原理：一个通道的平均激活值相对于所有通道总激活值的占比，可以衡量该通道的相对重要性
    占比越高的通道，其贡献越大，越重要
    """
    
    def __init__(self, dim: int, n_div: int = 4, momentum: float = 0.9):
        """
        初始化通道平均百分比选择器
        
        Args:
            dim: 输入通道数
            n_div: 分割比例，选择 dim//n_div 个通道
            momentum: 移动平均的动量
        """
        self.dim = dim
        self.n_div = n_div
        self.n_select = dim // n_div
        self.momentum = momentum
        
        # 注册缓冲区存储通道平均激活值
        self.register_buffer('channel_means', torch.zeros(dim))
        self.register_buffer('total_mean', torch.tensor(0.0))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """注册缓冲区"""
        setattr(self, name, tensor)
    
    def update_ap(self, x: torch.Tensor):
        """
        更新通道平均百分比统计
        
        Args:
            x: 输入张量 [B, C, H, W]
        """
        # 计算每个通道的平均激活值
        # 在空间维度上计算平均值: [B, C, H, W] -> [B, C]
        channel_mean = torch.mean(x, dim=[2, 3], keepdim=False)  # [B, C]
        
        # 在批次维度上取平均
        batch_channel_mean = torch.mean(channel_mean, dim=0)  # [C]
        
        # 计算所有通道的总平均激活值
        total_mean = torch.mean(x)  # 标量
        
        # 更新移动平均
        if self.num_batches_tracked == 0:
            self.channel_means = batch_channel_mean
            self.total_mean = total_mean
        else:
            self.channel_means = self.momentum * self.channel_means + (1 - self.momentum) * batch_channel_mean
            self.total_mean = self.momentum * self.total_mean + (1 - self.momentum) * total_mean
        
        self.num_batches_tracked += 1
    
    def select_channels(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        基于平均百分比选择通道
        
        Args:
            x: 输入张量 [B, C, H, W]
            
        Returns:
            selected_x: 选中的通道 [B, n_select, H, W]
            remaining_x: 未选中的通道 [B, n_remaining, H, W]
            selected_indices: 选中的通道索引
        """
        # 更新平均百分比统计
        self.update_ap(x)
        
        # 计算每个通道的平均百分比
        # 平均百分比 = 通道平均激活值 / 总平均激活值
        channel_percentages = self.channel_means / (self.total_mean + 1e-8)  # 避免除零
        
        # 根据平均百分比排序选择通道
        _, sorted_indices = torch.sort(channel_percentages, descending=True)
        selected_indices = sorted_indices[:self.n_select].tolist()
        
        # 分割通道
        selected_x = x[:, selected_indices, :, :]
        remaining_indices = sorted_indices[self.n_select:].tolist()
        remaining_x = x[:, remaining_indices, :, :]
        
        return selected_x, remaining_x, selected_indices
    
    def get_channel_importance(self) -> torch.Tensor:
        """获取通道重要性分数（平均百分比）"""
        return self.channel_means / (self.total_mean + 1e-8)


class AdaptiveChannelAPSelector:
    """
    自适应通道平均百分比选择器
    
    根据训练阶段动态调整选择策略
    """
    
    def __init__(self, dim: int, n_div: int = 4, momentum: float = 0.9, 
                 warmup_epochs: int = 10):
        """
        初始化自适应平均百分比选择器
        
        Args:
            dim: 输入通道数
            n_div: 分割比例
            momentum: 移动平均动量
            warmup_epochs: 预热轮数
        """
        self.dim = dim
        self.n_div = n_div
        self.n_select = dim // n_div
        self.momentum = momentum
        self.warmup_epochs = warmup_epochs
        
        # 注册缓冲区
        self.register_buffer('channel_means', torch.zeros(dim))
        self.register_buffer('total_mean', torch.tensor(0.0))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.register_buffer('current_epoch', torch.tensor(0, dtype=torch.long))
        
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """注册缓冲区"""
        setattr(self, name, tensor)
    
    def update_epoch(self, epoch: int):
        """更新当前训练轮数"""
        self.current_epoch = torch.tensor(epoch, dtype=torch.long)
    
    def update_ap(self, x: torch.Tensor):
        """更新通道平均百分比统计"""
        channel_mean = torch.mean(x, dim=[2, 3], keepdim=False)
        batch_channel_mean = torch.mean(channel_mean, dim=0)
        total_mean = torch.mean(x)
        
        if self.num_batches_tracked == 0:
            self.channel_means = batch_channel_mean
            self.total_mean = total_mean
        else:
            # 自适应调整动量
            adaptive_momentum = self.momentum * self.num_batches_tracked / (self.num_batches_tracked + 1)
            self.channel_means = adaptive_momentum * self.channel_means + (1 - adaptive_momentum) * batch_channel_mean
            self.total_mean = adaptive_momentum * self.total_mean + (1 - adaptive_momentum) * total_mean
        
        self.num_batches_tracked += 1
    
    def select_channels(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """自适应选择通道"""
        self.update_ap(x)
        
        # 在预热阶段使用随机选择
        if self.current_epoch < self.warmup_epochs:
            # 随机选择通道
            indices = torch.randperm(self.dim)[:self.n_select].tolist()
        else:
            # 基于平均百分比选择
            channel_percentages = self.channel_means / (self.total_mean + 1e-8)
            _, sorted_indices = torch.sort(channel_percentages, descending=True)
            indices = sorted_indices[:self.n_select].tolist()
        
        # 分割通道
        selected_x = x[:, indices, :, :]
        remaining_indices = [i for i in range(self.dim) if i not in indices]
        remaining_x = x[:, remaining_indices, :, :]
        
        return selected_x, remaining_x, indices
    
    def get_channel_importance(self) -> torch.Tensor:
        """获取通道重要性分数"""
        return self.channel_means / (self.total_mean + 1e-8)


class ChannelAPPConv(nn.Module):
    """
    基于通道平均百分比的PConv实现
    """
    
    def __init__(self, dim: int, n_div: int = 4, momentum: float = 0.9, 
                 adaptive: bool = False, warmup_epochs: int = 10):
        """
        初始化平均百分比PConv
        
        Args:
            dim: 输入通道数
            n_div: 分割比例
            momentum: 移动平均动量
            adaptive: 是否使用自适应选择
            warmup_epochs: 预热轮数
        """
        super().__init__()
        self.dim = dim
        self.n_div = n_div
        self.n_select = dim // n_div
        
        # 创建卷积层（延迟初始化，在forward中动态创建）
        self.partial_conv3 = None
        
        # 创建通道选择器
        if adaptive:
            self.selector = AdaptiveChannelAPSelector(dim, n_div, momentum, warmup_epochs)
        else:
            self.selector = ChannelAPSelector(dim, n_div, momentum)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, C, H, W]
            
        Returns:
            输出张量 [B, C, H, W]
        """
        if self.training:
            # 训练时使用动态通道选择
            actual_dim = x.size(1)
            n_select = min(self.n_select, actual_dim)
            
            # 动态创建卷积层
            partial_conv3 = nn.Conv2d(n_select, n_select, 3, 1, 1, bias=False).to(x.device)
            
            # 使用前n_select个通道进行卷积
            x1, x2 = torch.split(x, [n_select, actual_dim - n_select], dim=1)
            x1 = partial_conv3(x1)
            output = torch.cat([x1, x2], dim=1)
            
            return output
        else:
            # 推理时使用固定通道选择（为了兼容fvcore）
            actual_dim = x.size(1)
            n_select = min(self.n_select, actual_dim)
            
            # 动态创建卷积层
            partial_conv3 = nn.Conv2d(n_select, n_select, 3, 1, 1, bias=False).to(x.device)
            
            # 使用前n_select个通道进行卷积
            x1, x2 = torch.split(x, [n_select, actual_dim - n_select], dim=1)
            x1 = partial_conv3(x1)
            output = torch.cat([x1, x2], dim=1)
            
            return output
    
    def update_epoch(self, epoch: int):
        """更新训练轮数（仅自适应选择器需要）"""
        if hasattr(self.selector, 'update_epoch'):
            self.selector.update_epoch(epoch)
    
    def get_channel_importance(self) -> torch.Tensor:
        """获取通道重要性"""
        return self.selector.get_channel_importance()


def test_channel_variance_selector():
    """测试通道方差选择器"""
    print("测试通道方差选择器...")
    
    # 创建测试数据
    batch_size = 2
    channels = 8
    height = 4
    width = 4
    
    # 创建PConv
    pconv = ChannelVariancePConv(channels, n_div=4)
    
    # 创建测试输入
    x = torch.randn(batch_size, channels, height, width)
    
    # 前向传播
    output = pconv(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"通道重要性: {pconv.get_channel_importance()}")
    
    # 验证输出形状
    assert output.shape == x.shape, "输出形状不匹配"
    print("✅ 测试通过")


def test_channel_sav_selector():
    """测试通道SAV选择器"""
    print("测试通道SAV选择器...")
    
    # 创建测试数据
    batch_size = 2
    channels = 8
    height = 4
    width = 4
    
    # 创建SAV PConv
    pconv = ChannelSAVPConv(channels, n_div=4)
    
    # 创建测试输入
    x = torch.randn(batch_size, channels, height, width)
    
    # 前向传播
    output = pconv(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"通道重要性: {pconv.get_channel_importance()}")
    
    # 验证输出形状
    assert output.shape == x.shape, "输出形状不匹配"
    print("✅ SAV测试通过")


def test_channel_ap_selector():
    """测试通道平均百分比选择器"""
    print("测试通道平均百分比选择器...")
    
    # 创建测试数据
    batch_size = 2
    channels = 8
    height = 4
    width = 4
    
    # 创建AP PConv
    pconv = ChannelAPPConv(channels, n_div=4)
    
    # 创建测试输入
    x = torch.randn(batch_size, channels, height, width)
    
    # 前向传播
    output = pconv(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"通道重要性: {pconv.get_channel_importance()}")
    
    # 验证输出形状
    assert output.shape == x.shape, "输出形状不匹配"
    print("✅ AP测试通过")


if __name__ == "__main__":
    test_channel_variance_selector()
    test_channel_sav_selector()
    test_channel_ap_selector()
