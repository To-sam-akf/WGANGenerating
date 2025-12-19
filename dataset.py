import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class FaceDataset(Dataset):
    """二次元头像数据集类"""
    
    def __init__(self, data_dir, transform=None, image_size=64):
        """
        Args:
            data_dir (str): 数据集目录路径
            transform: 图像变换
            image_size (int): 图像大小
        """
        self.data_dir = data_dir
        self.image_size = image_size
        
        # 获取所有图片文件路径
        self.image_paths = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))
        
        print(f"找到 {len(self.image_paths)} 张图片")
        
        # 默认变换（添加数据增强）
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # 颜色抖动
                transforms.RandomRotation(degrees=5),  # 随机旋转
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1, 1]
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            # 加载图片
            image = Image.open(img_path).convert('RGB')
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            
            return image
        except Exception as e:
            print(f"加载图片失败: {img_path}, 错误: {e}")
            # 返回一个随机图片作为替代
            return torch.randn(3, self.image_size, self.image_size)

def get_dataloader(data_dir, batch_size=64, image_size=64, num_workers=4, shuffle=True):
    """获取数据加载器"""
    dataset = FaceDataset(data_dir, image_size=image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return dataloader

def denormalize(tensor):
    """反归一化，将[-1,1]的tensor转换为[0,1]"""
    return (tensor + 1) / 2

def save_image_grid(tensor, filename, nrow=8):
    """保存图片网格"""
    from torchvision.utils import save_image
    tensor = denormalize(tensor)
    save_image(tensor, filename, nrow=nrow, padding=2)
