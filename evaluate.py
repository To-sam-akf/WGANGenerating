import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
import os
from tqdm import tqdm

class InceptionV3(nn.Module):
    """用于计算FID和IS的Inception V3模型"""
    
    def __init__(self, resize_input=True, normalize_input=True, requires_grad=False):
        super(InceptionV3, self).__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        
        # 加载预训练的Inception V3模型
        inception = models.inception_v3(pretrained=True)
        
        # 移除最后的分类层
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c
        
        # 添加自适应平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        if self.normalize_input:
            x = 2 * x - 1  # 归一化到[-1, 1]
        
        # 前向传播
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        
        # 全局平均池化
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        return x

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """计算两个多元高斯分布之间的Frechet距离"""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, "均值向量维度不匹配"
    assert sigma1.shape == sigma2.shape, "协方差矩阵维度不匹配"
    
    diff = mu1 - mu2
    
    # 计算协方差矩阵的平方根
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # 数值稳定性检查
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def get_activations(images, model, batch_size=50, device='cuda'):
    """获取图像的Inception特征"""
    model.eval()
    
    if len(images.shape) != 4:
        raise ValueError('图像张量必须是4维的')
    
    n_batches = len(images) // batch_size + (1 if len(images) % batch_size != 0 else 0)
    pred_arr = np.empty((len(images), 2048))
    
    for i in tqdm(range(n_batches), desc='计算特征'):
        start = i * batch_size
        end = min(start + batch_size, len(images))
        
        batch = images[start:end].to(device)
        
        with torch.no_grad():
            pred = model(batch)
            pred_arr[start:end] = pred.cpu().numpy()
    
    return pred_arr

def calculate_fid(real_images, fake_images, device='cuda'):
    """计算FID (Frechet Inception Distance)"""
    print("正在计算FID...")
    
    # 初始化Inception模型
    inception_model = InceptionV3().to(device)
    inception_model.eval()
    
    # 获取真实图像和生成图像的特征
    real_features = get_activations(real_images, inception_model, device=device)
    fake_features = get_activations(fake_images, inception_model, device=device)
    
    # 计算均值和协方差
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # 计算FID
    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    
    return fid_score

def calculate_inception_score(images, device='cuda', splits=10):
    """计算IS (Inception Score)"""
    print("正在计算IS...")
    
    # 加载预训练的Inception模型
    inception_model = models.inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    
    def get_pred(x):
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()
    
    # 获取预测概率
    preds = np.zeros((len(images), 1000))
    
    batch_size = 32
    for i in tqdm(range(0, len(images), batch_size), desc='计算IS'):
        batch = images[i:i+batch_size].to(device)
        with torch.no_grad():
            batch_preds = get_pred(batch)
            preds[i:i+batch_size] = batch_preds
    
    # 计算IS
    scores = []
    for i in range(splits):
        part = preds[i * (len(preds) // splits): (i + 1) * (len(preds) // splits), :]
        py = np.mean(part, axis=0)
        scores.append(np.exp(np.mean([np.sum(p * np.log(p / py)) for p in part])))
    
    return np.mean(scores), np.std(scores)

def calculate_lpips(real_images, fake_images, device='cuda'):
    """计算LPIPS (Learned Perceptual Image Patch Similarity)"""
    try:
        import lpips
        loss_fn = lpips.LPIPS(net='alex').to(device)
        
        # 随机选择一些图像对进行比较
        n_samples = min(1000, len(real_images), len(fake_images))
        indices = np.random.choice(len(real_images), n_samples, replace=False)
        
        real_sample = real_images[indices]
        fake_sample = fake_images[indices]
        
        lpips_scores = []
        batch_size = 32
        
        for i in tqdm(range(0, n_samples, batch_size), desc='计算LPIPS'):
            end_idx = min(i + batch_size, n_samples)
            real_batch = real_sample[i:end_idx].to(device)
            fake_batch = fake_sample[i:end_idx].to(device)
            
            with torch.no_grad():
                lpips_score = loss_fn(real_batch, fake_batch)
                lpips_scores.extend(lpips_score.cpu().numpy())
        
        return np.mean(lpips_scores)
    except ImportError:
        print("LPIPS库未安装，跳过LPIPS计算")
        return None

def evaluate_generator(generator, real_dataloader, device='cuda', num_samples=5000, nz=100):
    """全面评估生成器性能"""
    generator.eval()
    
    print(f"正在生成 {num_samples} 张图像进行评估...")
    
    # 生成假图像
    fake_images = []
    batch_size = 64
    
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc='生成图像'):
            current_batch_size = min(batch_size, num_samples - i)
            noise = torch.randn(current_batch_size, nz, 1, 1, device=device)
            fake_batch = generator(noise)
            fake_images.append(fake_batch.cpu())
    
    fake_images = torch.cat(fake_images, dim=0)
    
    # 收集真实图像
    print("正在收集真实图像...")
    real_images = []
    count = 0
    
    for batch in tqdm(real_dataloader, desc='收集真实图像'):
        real_images.append(batch)
        count += batch.size(0)
        if count >= num_samples:
            break
    
    real_images = torch.cat(real_images, dim=0)[:num_samples]
    
    # 确保图像在[0,1]范围内
    real_images = (real_images + 1) / 2  # 从[-1,1]转换到[0,1]
    fake_images = (fake_images + 1) / 2  # 从[-1,1]转换到[0,1]
    
    # 计算评估指标
    results = {}
    
    # FID
    try:
        fid_score = calculate_fid(real_images, fake_images, device)
        results['FID'] = fid_score
        print(f"FID: {fid_score:.4f}")
    except Exception as e:
        print(f"FID计算失败: {e}")
        results['FID'] = None
    
    # IS
    try:
        is_mean, is_std = calculate_inception_score(fake_images, device)
        results['IS_mean'] = is_mean
        results['IS_std'] = is_std
        print(f"IS: {is_mean:.4f} ± {is_std:.4f}")
    except Exception as e:
        print(f"IS计算失败: {e}")
        results['IS_mean'] = None
        results['IS_std'] = None
    
    # LPIPS
    try:
        lpips_score = calculate_lpips(real_images, fake_images, device)
        if lpips_score is not None:
            results['LPIPS'] = lpips_score
            print(f"LPIPS: {lpips_score:.4f}")
    except Exception as e:
        print(f"LPIPS计算失败: {e}")
        results['LPIPS'] = None
    
    return results

def save_evaluation_results(results, filepath):
    """保存评估结果"""
    import json
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"评估结果已保存到: {filepath}")