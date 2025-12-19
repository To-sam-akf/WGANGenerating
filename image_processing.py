#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图像处理模块
提供丰富的图像处理功能，包括数据增强、图像质量评估、滤波等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import cv2
import random
import math
from typing import List, Tuple, Optional, Union

class AdvancedImageTransforms:
    """高级图像变换类"""
    
    def __init__(self, image_size=64):
        self.image_size = image_size
    
    def get_training_transforms(self, augment_prob=0.5):
        """获取训练时的数据增强变换"""
        transforms_list = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        
        # 添加随机数据增强
        if random.random() < augment_prob:
            # 随机旋转
            transforms_list.append(transforms.RandomRotation(degrees=10))
            
            # 随机颜色抖动
            transforms_list.append(
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                )
            )
            
            # 随机高斯模糊
            if random.random() < 0.3:
                transforms_list.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)))
        
        # 基础变换
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        return transforms.Compose(transforms_list)
    
    def get_validation_transforms(self):
        """获取验证时的变换（无增强）"""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def get_test_transforms(self):
        """获取测试时的变换"""
        return self.get_validation_transforms()

class ImageQualityAssessment:
    """图像质量评估类"""
    
    @staticmethod
    def calculate_psnr(img1, img2, max_val=1.0):
        """计算PSNR (Peak Signal-to-Noise Ratio)"""
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(max_val / torch.sqrt(mse))
    
    @staticmethod
    def calculate_ssim(img1, img2, window_size=11, max_val=1.0):
        """计算SSIM (Structural Similarity Index)"""
        def gaussian_window(size, sigma=1.5):
            coords = torch.arange(size, dtype=torch.float32)
            coords -= size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g /= g.sum()
            return g.unsqueeze(0) * g.unsqueeze(1)
        
        # 创建高斯窗口
        window = gaussian_window(window_size).unsqueeze(0).unsqueeze(0)
        
        # 常数
        C1 = (0.01 * max_val) ** 2
        C2 = (0.03 * max_val) ** 2
        
        # 计算均值
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=1)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=1)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # 计算方差和协方差
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=1) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=1) - mu1_mu2
        
        # 计算SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()
    
    @staticmethod
    def calculate_image_sharpness(image):
        """计算图像清晰度（基于拉普拉斯算子）"""
        if isinstance(image, torch.Tensor):
            # 转换为numpy数组
            if image.dim() == 4:  # batch
                image = image[0]
            if image.dim() == 3:  # CHW
                image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
        
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # 计算拉普拉斯算子
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()
    
    @staticmethod
    def detect_blur(image, threshold=100):
        """检测图像是否模糊"""
        sharpness = ImageQualityAssessment.calculate_image_sharpness(image)
        return sharpness < threshold

class ImageFilters:
    """图像滤波器类"""
    
    @staticmethod
    def apply_gaussian_blur(image, kernel_size=5, sigma=1.0):
        """应用高斯模糊"""
        if isinstance(image, torch.Tensor):
            # 使用PyTorch的高斯模糊
            blur = transforms.GaussianBlur(kernel_size, sigma)
            return blur(image)
        else:
            # 使用PIL的高斯模糊
            return image.filter(ImageFilter.GaussianBlur(radius=sigma))
    
    @staticmethod
    def apply_unsharp_mask(image, radius=2, percent=150, threshold=3):
        """应用锐化滤波器"""
        if isinstance(image, Image.Image):
            return image.filter(ImageFilter.UnsharpMask(radius, percent, threshold))
        else:
            # 对于tensor，需要转换为PIL再处理
            pil_image = TF.to_pil_image(image)
            filtered = pil_image.filter(ImageFilter.UnsharpMask(radius, percent, threshold))
            return TF.to_tensor(filtered)
    
    @staticmethod
    def apply_edge_enhance(image):
        """应用边缘增强"""
        if isinstance(image, Image.Image):
            return image.filter(ImageFilter.EDGE_ENHANCE)
        else:
            pil_image = TF.to_pil_image(image)
            filtered = pil_image.filter(ImageFilter.EDGE_ENHANCE)
            return TF.to_tensor(filtered)
    
    @staticmethod
    def apply_median_filter(image, size=3):
        """应用中值滤波"""
        if isinstance(image, torch.Tensor):
            # 转换为numpy进行中值滤波
            if image.dim() == 4:
                image = image[0]
            if image.dim() == 3:
                image = image.permute(1, 2, 0)
            
            np_image = (image.cpu().numpy() * 255).astype(np.uint8)
            filtered = cv2.medianBlur(np_image, size)
            
            # 转换回tensor
            filtered_tensor = torch.from_numpy(filtered.astype(np.float32) / 255.0)
            if len(filtered_tensor.shape) == 3:
                filtered_tensor = filtered_tensor.permute(2, 0, 1)
            
            return filtered_tensor
        else:
            return image.filter(ImageFilter.MedianFilter(size))

class ColorSpaceConverter:
    """颜色空间转换器"""
    
    @staticmethod
    def rgb_to_hsv(rgb_tensor):
        """RGB转HSV"""
        rgb = rgb_tensor.clone()
        
        # 确保输入在[0,1]范围内
        rgb = torch.clamp(rgb, 0, 1)
        
        r, g, b = rgb[0], rgb[1], rgb[2]
        
        max_val, max_idx = torch.max(rgb, dim=0)
        min_val, _ = torch.min(rgb, dim=0)
        
        diff = max_val - min_val
        
        # 计算H
        h = torch.zeros_like(max_val)
        
        # R是最大值
        mask = (max_idx == 0) & (diff != 0)
        h[mask] = (60 * ((g[mask] - b[mask]) / diff[mask]) + 360) % 360
        
        # G是最大值
        mask = (max_idx == 1) & (diff != 0)
        h[mask] = (60 * ((b[mask] - r[mask]) / diff[mask]) + 120) % 360
        
        # B是最大值
        mask = (max_idx == 2) & (diff != 0)
        h[mask] = (60 * ((r[mask] - g[mask]) / diff[mask]) + 240) % 360
        
        # 计算S
        s = torch.zeros_like(max_val)
        s[max_val != 0] = diff[max_val != 0] / max_val[max_val != 0]
        
        # V就是max_val
        v = max_val
        
        return torch.stack([h / 360.0, s, v], dim=0)
    
    @staticmethod
    def hsv_to_rgb(hsv_tensor):
        """HSV转RGB"""
        h, s, v = hsv_tensor[0] * 360, hsv_tensor[1], hsv_tensor[2]
        
        c = v * s
        x = c * (1 - torch.abs((h / 60) % 2 - 1))
        m = v - c
        
        rgb = torch.zeros_like(hsv_tensor)
        
        # 根据H的值确定RGB
        mask = (h >= 0) & (h < 60)
        rgb[0][mask] = c[mask]
        rgb[1][mask] = x[mask]
        rgb[2][mask] = 0
        
        mask = (h >= 60) & (h < 120)
        rgb[0][mask] = x[mask]
        rgb[1][mask] = c[mask]
        rgb[2][mask] = 0
        
        mask = (h >= 120) & (h < 180)
        rgb[0][mask] = 0
        rgb[1][mask] = c[mask]
        rgb[2][mask] = x[mask]
        
        mask = (h >= 180) & (h < 240)
        rgb[0][mask] = 0
        rgb[1][mask] = x[mask]
        rgb[2][mask] = c[mask]
        
        mask = (h >= 240) & (h < 300)
        rgb[0][mask] = x[mask]
        rgb[1][mask] = 0
        rgb[2][mask] = c[mask]
        
        mask = (h >= 300) & (h < 360)
        rgb[0][mask] = c[mask]
        rgb[1][mask] = 0
        rgb[2][mask] = x[mask]
        
        rgb += m
        
        return torch.clamp(rgb, 0, 1)

class ImageAugmentation:
    """图像增强类"""
    
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def random_cutout(self, image, n_holes=1, length=16):
        """随机遮挡"""
        h, w = image.shape[-2:]
        mask = torch.ones((h, w), dtype=torch.float32)
        
        for _ in range(n_holes):
            y = torch.randint(0, h, (1,)).item()
            x = torch.randint(0, w, (1,)).item()
            
            y1 = max(0, y - length // 2)
            y2 = min(h, y + length // 2)
            x1 = max(0, x - length // 2)
            x2 = min(w, x + length // 2)
            
            mask[y1:y2, x1:x2] = 0
        
        return image * mask.unsqueeze(0)
    
    def random_noise(self, image, noise_type='gaussian', intensity=0.1):
        """添加随机噪声"""
        if random.random() > self.prob:
            return image
        
        if noise_type == 'gaussian':
            noise = torch.randn_like(image) * intensity
        elif noise_type == 'uniform':
            noise = (torch.rand_like(image) - 0.5) * 2 * intensity
        elif noise_type == 'salt_pepper':
            noise = torch.zeros_like(image)
            # 盐噪声
            salt_mask = torch.rand_like(image) < intensity / 2
            noise[salt_mask] = 1
            # 椒噪声
            pepper_mask = torch.rand_like(image) < intensity / 2
            noise[pepper_mask] = -1
        else:
            return image
        
        return torch.clamp(image + noise, -1, 1)
    
    def random_brightness_contrast(self, image, brightness_range=0.2, contrast_range=0.2):
        """随机亮度和对比度调整"""
        if random.random() > self.prob:
            return image
        
        # 亮度调整
        brightness_factor = 1 + random.uniform(-brightness_range, brightness_range)
        image = image * brightness_factor
        
        # 对比度调整
        contrast_factor = 1 + random.uniform(-contrast_range, contrast_range)
        mean = image.mean(dim=(1, 2), keepdim=True)
        image = (image - mean) * contrast_factor + mean
        
        return torch.clamp(image, -1, 1)
    
    def random_hue_saturation(self, image, hue_range=0.1, saturation_range=0.2):
        """随机色调和饱和度调整"""
        if random.random() > self.prob:
            return image
        
        # 转换到HSV空间
        # 先反归一化到[0,1]
        image_01 = (image + 1) / 2
        hsv = ColorSpaceConverter.rgb_to_hsv(image_01)
        
        # 调整色调
        hue_shift = random.uniform(-hue_range, hue_range)
        hsv[0] = (hsv[0] + hue_shift) % 1.0
        
        # 调整饱和度
        saturation_factor = 1 + random.uniform(-saturation_range, saturation_range)
        hsv[1] = torch.clamp(hsv[1] * saturation_factor, 0, 1)
        
        # 转换回RGB
        rgb_01 = ColorSpaceConverter.hsv_to_rgb(hsv)
        
        # 重新归一化到[-1,1]
        return rgb_01 * 2 - 1

class ImageProcessor:
    """图像处理器主类"""
    
    def __init__(self, image_size=64):
        self.image_size = image_size
        self.transforms = AdvancedImageTransforms(image_size)
        self.quality_assessor = ImageQualityAssessment()
        self.filters = ImageFilters()
        self.color_converter = ColorSpaceConverter()
        self.augmentation = ImageAugmentation()
    
    def preprocess_image(self, image_path, mode='train'):
        """预处理单张图像"""
        image = Image.open(image_path).convert('RGB')
        
        if mode == 'train':
            transform = self.transforms.get_training_transforms()
        else:
            transform = self.transforms.get_validation_transforms()
        
        return transform(image)
    
    def postprocess_image(self, tensor, denormalize=True):
        """后处理图像tensor"""
        if denormalize:
            # 反归一化从[-1,1]到[0,1]
            tensor = (tensor + 1) / 2
        
        # 确保值在有效范围内
        tensor = torch.clamp(tensor, 0, 1)
        
        return tensor
    
    def enhance_image_quality(self, image, enhance_type='sharpen'):
        """增强图像质量"""
        if enhance_type == 'sharpen':
            return self.filters.apply_unsharp_mask(image)
        elif enhance_type == 'denoise':
            return self.filters.apply_median_filter(image)
        elif enhance_type == 'edge_enhance':
            return self.filters.apply_edge_enhance(image)
        else:
            return image
    
    def batch_process_images(self, image_batch, operations=None):
        """批量处理图像"""
        if operations is None:
            operations = ['denoise', 'sharpen']
        
        processed_batch = image_batch.clone()
        
        for operation in operations:
            if operation == 'denoise':
                for i in range(processed_batch.size(0)):
                    processed_batch[i] = self.filters.apply_median_filter(processed_batch[i])
            elif operation == 'sharpen':
                for i in range(processed_batch.size(0)):
                    processed_batch[i] = self.filters.apply_unsharp_mask(processed_batch[i])
            elif operation == 'augment':
                for i in range(processed_batch.size(0)):
                    processed_batch[i] = self.augmentation.random_noise(processed_batch[i])
        
        return processed_batch
    
    def evaluate_image_quality(self, image1, image2=None):
        """评估图像质量"""
        results = {}
        
        # 计算清晰度
        results['sharpness'] = self.quality_assessor.calculate_image_sharpness(image1)
        results['is_blurry'] = self.quality_assessor.detect_blur(image1)
        
        # 如果提供了参考图像，计算PSNR和SSIM
        if image2 is not None:
            results['psnr'] = self.quality_assessor.calculate_psnr(image1, image2)
            results['ssim'] = self.quality_assessor.calculate_ssim(image1, image2)
        
        return results

# 便捷函数
def create_image_processor(image_size=64):
    """创建图像处理器实例"""
    return ImageProcessor(image_size)

def apply_data_augmentation(image, augment_prob=0.5):
    """应用数据增强"""
    augmenter = ImageAugmentation(prob=augment_prob)
    
    # 应用多种增强
    image = augmenter.random_noise(image)
    image = augmenter.random_brightness_contrast(image)
    image = augmenter.random_hue_saturation(image)
    
    return image

def enhance_generated_images(image_batch, enhancement_type='sharpen'):
    """增强生成的图像"""
    processor = ImageProcessor()
    enhanced_batch = []
    
    for i in range(image_batch.size(0)):
        enhanced = processor.enhance_image_quality(image_batch[i], enhancement_type)
        enhanced_batch.append(enhanced)
    
    return torch.stack(enhanced_batch)