## 📋 项目概述

这是一个基于 **Wasserstein GAN (WGAN)** 的二次元头像生成项目，通过深度学习和生成对抗网络技术，能够自动生成高质量的二次元人物头像。

## ✨ 主要特性

- **WGAN-GP 架构**：使用 Wasserstein 距离和梯度惩罚，训练更稳定
- **谱归一化**：判别器采用谱归一化技术提高训练稳定性
- **多种评估指标**：支持 FID、IS、LPIPS 等多种生成质量评估
- **丰富的图像处理**：提供数据增强、质量评估、滤波等功能
- **灵活的训练配置**：预设多种训练配置，可快速启动训练
- **完整的可视化**：自动生成训练过程和评估结果的可视化图表

## 📁 项目结构

```
43_生成对抗网络生成二次元头像_代码/
├── main.py                 # 主训练脚本
├── train.py                # 训练启动器（预设配置）
├── net.py                  # 生成器和判别器网络定义
├── dataset.py              # 数据集加载和处理
├── evaluate.py             # 模型评估指标计算
├── image_processing.py     # 图像处理和增强工具
├── plot.py                 # 训练结果可视化脚本
├── faces.zip               # 二次元头像数据集（需解压）
└── readme.md               # 项目文档
```

## 🔧 核心模块说明

### net.py
- **Generator**：生成器网络，从噪声向量生成 64×64 的彩色头像
- **Discriminator**：判别器网络，评估生成图像的真实性
- `gradient_penalty()`：计算 WGAN-GP 的梯度惩罚
- `create_noise()`：生成随机噪声向量

### dataset.py
- **FaceDataset**：自定义数据集类，支持多种格式图像加载
- `get_dataloader()`：创建数据加载器
- `denormalize()`：反归一化函数
- `save_image_grid()`：保存图像网格

### evaluate.py
- **InceptionV3**：用于计算 FID 的预训练模型
- `calculate_fid()`：计算 Frechet Inception Distance
- `calculate_inception_score()`：计算 Inception Score
- `calculate_lpips()`：计算 LPIPS 感知损失
- `evaluate_generator()`：全面评估生成器性能

### image_processing.py
提供丰富的图像处理功能：
- **AdvancedImageTransforms**：高级数据增强
- **ImageQualityAssessment**：图像质量评估（PSNR、SSIM）
- **ImageFilters**：图像滤波（高斯、中值、锐化等）
- **ColorSpaceConverter**：颜色空间转换（RGB ↔ HSV）
- **ImageAugmentation**：数据增强（噪声、亮度、色调等）

### main.py
主训练脚本，包含 **WGANTrainer** 类：
- 完整的训练循环
- 模型检查点保存/加载
- 自动评估和可视化
- 详细的训练日志

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch torchvision torchaudio
pip install pandas matplotlib numpy scipy pillow opencv-python tqdm
pip install lpips  # 可选，用于 LPIPS 计算
```

### 2. 准备数据集

```bash
# 解压二次元头像数据集
unzip faces.zip -d ./faces/
```

确保数据集结构如下：
```
faces/
└── faces/
    ├── image1.jpg
    ├── image2.png
    └── ...
```

### 3. 开始训练


**直接运行 main.py 与自定义参数：**

```bash
python main.py \
    --data_dir ./faces/faces \
    --output_dir ./output \
    --batch_size 64 \
    --num_epochs 200 \
    --lr_g 0.00002 \
    --lr_d 0.0002
```

### 4. 查看训练结果

```bash
# 生成训练可视化图表
python plot.py
```

## 📊 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 64 | 批次大小 |
| `--num_epochs` | 400 | 训练轮数 |
| `--lr_g` | 0.00002 | 生成器学习率 |
| `--lr_d` | 0.0002 | 判别器学习率 |
| `--lambda_gp` | 5.0 | 梯度惩罚权重 |
| `--n_critic` | 5 | 每次更新生成器前更新判别器次数 |
| `--nz` | 100 | 噪声向量维度 |
| `--ngf` | 128 | 生成器特征图数量 |
| `--ndf` | 128 | 判别器特征图数量 |
| `--image_size` | 64 | 生成图像大小 |

## 📈 训练输出

训练完成后，输出目录结构如下：

```
output/
├── images/                  # 生成的图像样本
│   ├── epoch_0005.png
│   ├── epoch_0010.png
│   └── ...
├── checkpoints/             # 模型检查点
│   ├── latest.pth           # 最新模型
│   └── best.pth             # 最佳模型（FID最低）
├── trainrecord.txt          # 训练记录（损失、评分等）
├── eva.txt                  # 评估结果（FID、IS、LPIPS）
├── config.json              # 训练配置参数
├── training_curves.png      # 损失曲线图
└── final_evaluation.json    # 最终评估指标
```

## 📊 评估指标

### FID (Frechet Inception Distance)
- 衡量生成图像分布与真实图像分布的距离
- **越低越好**，理想值接近 0
- 计算方式：基于 Inception 网络的特征统计

### IS (Inception Score)
- 衡量生成图像的质量和多样性
- **越高越好**，通常范围 1-10
- 计算方式：使用 Inception 网络的分类概率

### LPIPS (Learned Perceptual Image Patch Similarity)
- 基于深度学习的感知相似度
- **越低越好**，范围 0-1
- 更符合人类视觉感知

## 💡 训练技巧

1. **数据量少时**：使用较小的 `batch_size`（32）和较短的训练周期
2. **显存不足时**：减小 `batch_size` 或使用梯度累积
3. **训练不稳定**：增加 `lambda_gp` 值（梯度惩罚权重）
4. **生成效果差**：
   - 增加训练轮数
   - 调整学习率
   - 检查数据集质量
5. **恢复训练**：使用 `--resume` 参数从最新检查点继续训练

```bash
python main.py --resume --output_dir ./output
```

## 🎨 生成新图像

训练完成后，可以使用以下代码生成新的头像：

```python
import torch
from net import Generator

# 加载训练好的模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
netG = Generator(nz=100, ngf=128, nc=3).to(device)

checkpoint = torch.load('output/checkpoints/best.pth', map_location=device)
netG.load_state_dict(checkpoint['netG_state_dict'])
netG.eval()

# 生成图像
with torch.no_grad():
    noise = torch.randn(16, 100, 1, 1, device=device)
    fake_images = netG(noise)
    
# 保存生成的图像
from dataset import save_image_grid
save_image_grid(fake_images, 'generated_faces.png', nrow=4)
```

## 🔍 故障排除

| 问题 | 解决方案 |
|------|--------|
| 找不到数据集 | 检查 `faces.zip` 是否正确解压到 `./faces/` 目录 |
| CUDA 内存不足 | 减小 `batch_size` 参数 |
| 损失没有下降 | 检查数据集质量，尝试调整学习率或梯度惩罚权重 |
| 生成图像质量差 | 增加训练轮数，或从最佳模型继续训练 |

## 📚 参考文献

- Wasserstein GAN: https://arxiv.org/abs/1701.07875
- WGAN-GP: https://arxiv.org/abs/1704.00028
- Spectral Normalization: https://arxiv.org/abs/1802.05957
