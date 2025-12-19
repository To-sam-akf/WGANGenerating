import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import argparse
import json

from net import Generator, Discriminator, gradient_penalty, create_noise
from dataset import get_dataloader, save_image_grid
from evaluate import evaluate_generator, save_evaluation_results

class WGANTrainer:
    """WGAN训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
        
        # 初始化网络
        self.netG = Generator(args.nz, args.ngf, args.nc).to(self.device)
        self.netD = Discriminator(args.nc, args.ndf).to(self.device)
        
        print(f"生成器参数数量: {sum(p.numel() for p in self.netG.parameters()):,}")
        print(f"判别器参数数量: {sum(p.numel() for p in self.netD.parameters()):,}")
        
        # 初始化优化器
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=args.lr_g, betas=(args.beta1, 0.999))
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=args.lr_d, betas=(args.beta1, 0.999))
        
        # 学习率调度器
        self.schedulerG = optim.lr_scheduler.ExponentialLR(self.optimizerG, gamma=0.99)
        self.schedulerD = optim.lr_scheduler.ExponentialLR(self.optimizerD, gamma=0.99)
        
        # 数据加载器
        self.dataloader = get_dataloader(
            args.data_dir, 
            batch_size=args.batch_size, 
            image_size=args.image_size,
            num_workers=args.num_workers
        )
        
        # 固定噪声用于可视化
        self.fixed_noise = create_noise(64, args.nz, self.device)
        
        # 训练记录
        self.losses_g = []
        self.losses_d = []
        self.d_real_scores = []
        self.d_fake_scores = []
        
        # 最佳模型记录
        self.best_fid = float('inf')
        
    def train_discriminator(self, real_data):
        """训练判别器"""
        self.netD.zero_grad()
        batch_size = real_data.size(0)
        
        # 训练真实数据
        real_data = real_data.to(self.device)
        d_real = self.netD(real_data)
        
        # 训练生成数据
        noise = create_noise(batch_size, self.args.nz, self.device)
        fake_data = self.netG(noise).detach()
        d_fake = self.netD(fake_data)
        
        # WGAN损失
        d_loss_real = -torch.mean(d_real)
        d_loss_fake = torch.mean(d_fake)
        
        # 梯度惩罚
        gp = gradient_penalty(self.netD, real_data, fake_data, self.device)
        
        # 总损失
        d_loss = d_loss_real + d_loss_fake + self.args.lambda_gp * gp
        
        d_loss.backward()
        
        # 添加梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.netD.parameters(), max_norm=1.0)
        
        self.optimizerD.step()
        
        return d_loss.item(), torch.mean(d_real).item(), torch.mean(d_fake).item()
    
    def train_generator(self):
        """训练生成器"""
        self.netG.zero_grad()
        
        # 生成假数据
        noise = create_noise(self.args.batch_size, self.args.nz, self.device)
        fake_data = self.netG(noise)
        
        # 计算生成器损失
        d_fake = self.netD(fake_data)
        g_loss = -torch.mean(d_fake)
        
        g_loss.backward()
        
        # 添加梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=1.0)
        
        self.optimizerG.step()
        
        return g_loss.item()
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'netG_state_dict': self.netG.state_dict(),
            'netD_state_dict': self.netD.state_dict(),
            'optimizerG_state_dict': self.optimizerG.state_dict(),
            'optimizerD_state_dict': self.optimizerD.state_dict(),
            'losses_g': self.losses_g,
            'losses_d': self.losses_d,
            'best_fid': self.best_fid,
            'args': self.args
        }
        
        # 保存最新检查点
        torch.save(checkpoint, os.path.join(self.args.output_dir, 'checkpoints', 'latest.pth'))
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, os.path.join(self.args.output_dir, 'checkpoints', 'best.pth'))
            print(f"保存最佳模型 (FID: {self.best_fid:.4f})")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.netG.load_state_dict(checkpoint['netG_state_dict'])
            self.netD.load_state_dict(checkpoint['netD_state_dict'])
            self.optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
            self.optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
            
            self.losses_g = checkpoint.get('losses_g', [])
            self.losses_d = checkpoint.get('losses_d', [])
            self.best_fid = checkpoint.get('best_fid', float('inf'))
            
            print(f"从 {checkpoint_path} 加载检查点")
            return checkpoint['epoch']
        return 0
    
    def generate_samples(self, epoch):
        """生成样本图像"""
        self.netG.eval()
        with torch.no_grad():
            fake_images = self.netG(self.fixed_noise)
            save_path = os.path.join(self.args.output_dir, 'images', f'epoch_{epoch:04d}.png')
            save_image_grid(fake_images, save_path, nrow=8)
        self.netG.train()
    
    def plot_losses(self):
        """绘制损失曲线"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.losses_g, label='Generator Loss')
        plt.plot(self.losses_d, label='Discriminator Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.d_real_scores, label='D(real)')
        plt.plot(self.d_fake_scores, label='D(fake)')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.title('Discriminator Scores')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.output_dir, 'training_curves.png'))
        plt.close()
    
    def train(self):
        """主训练循环"""
        print("开始训练...")
        start_epoch = 0
        
        # 初始化训练记录文件
        trainrecord_path = os.path.join(self.args.output_dir, 'trainrecord.txt')
        eva_path = os.path.join(self.args.output_dir, 'eva.txt')
        
        # 如果不是恢复训练，则创建新的记录文件
        if not self.args.resume:
            with open(trainrecord_path, 'w') as f:
                f.write("Epoch,D_Loss,G_Loss,D_Real,D_Fake,Time(s)\n")
            with open(eva_path, 'w') as f:
                f.write("Epoch,FID,IS_Mean,IS_Std,LPIPS\n")
        
        # 加载检查点（如果存在）
        if self.args.resume:
            checkpoint_path = os.path.join(self.args.output_dir, 'checkpoints', 'latest.pth')
            start_epoch = self.load_checkpoint(checkpoint_path)
        
        # 训练循环
        iteration = 0
        
        for epoch in range(start_epoch, self.args.num_epochs):
            epoch_start_time = time.time()
            
            # 训练一个epoch
            for i, real_data in enumerate(tqdm(self.dataloader, desc=f'Epoch {epoch+1}/{self.args.num_epochs}')):
                
                # 训练判别器
                for _ in range(self.args.n_critic):
                    d_loss, d_real, d_fake = self.train_discriminator(real_data)
                
                # 训练生成器
                g_loss = self.train_generator()
                
                # 记录损失
                self.losses_g.append(g_loss)
                self.losses_d.append(d_loss)
                self.d_real_scores.append(d_real)
                self.d_fake_scores.append(d_fake)
                
                # 打印进度
                if iteration % self.args.print_freq == 0:
                    print(f'[{epoch+1}/{self.args.num_epochs}][{i+1}/{len(self.dataloader)}] '
                          f'Loss_D: {d_loss:.4f} Loss_G: {g_loss:.4f} '
                          f'D(x): {d_real:.4f} D(G(z)): {d_fake:.4f}')
                
                iteration += 1
            
            # 更新学习率
            self.schedulerG.step()
            self.schedulerD.step()
            
            # 生成样本图像
            if (epoch + 1) % self.args.sample_freq == 0:
                self.generate_samples(epoch + 1)
            
            # 评估模型
            if (epoch + 1) % self.args.eval_freq == 0:
                print("正在评估模型...")
                results = evaluate_generator(
                    self.netG, self.dataloader, self.device, 
                    num_samples=min(5000, len(self.dataloader.dataset)), 
                    nz=self.args.nz
                )
                
                # 保存评估结果
                eval_path = os.path.join(self.args.output_dir, f'evaluation_epoch_{epoch+1}.json')
                save_evaluation_results(results, eval_path)
                
                # 保存评估结果到eva.txt文件
                eva_path = os.path.join(self.args.output_dir, 'eva.txt')
                with open(eva_path, 'a') as f:
                    fid = results.get('FID', 'N/A')
                    is_mean = results.get('IS_mean', 'N/A')
                    is_std = results.get('IS_std', 'N/A')
                    lpips = results.get('LPIPS', 'N/A')
                    f.write(f"Epoch {epoch+1},{fid},{is_mean},{is_std},{lpips}\n")
                
                # 检查是否是最佳模型
                if results.get('FID') is not None and results['FID'] < self.best_fid:
                    self.best_fid = results['FID']
                    self.save_checkpoint(epoch + 1, is_best=True)
            
            # 保存检查点
            if (epoch + 1) % self.args.save_freq == 0:
                self.save_checkpoint(epoch + 1)
            
            # 绘制损失曲线
            if (epoch + 1) % self.args.plot_freq == 0:
                self.plot_losses()
            
            epoch_time = time.time() - epoch_start_time
            print(f'Epoch {epoch+1} 完成，用时: {epoch_time:.2f}s')
            
            # 保存每个epoch的loss到trainrecord.txt文件
            trainrecord_path = os.path.join(self.args.output_dir, 'trainrecord.txt')
            # 计算当前epoch的平均loss
            epoch_start_idx = max(0, len(self.losses_g) - len(self.dataloader))
            epoch_g_loss = np.mean(self.losses_g[epoch_start_idx:]) if len(self.losses_g) > 0 else 0
            epoch_d_loss = np.mean(self.losses_d[epoch_start_idx:]) if len(self.losses_d) > 0 else 0
            epoch_d_real = np.mean(self.d_real_scores[epoch_start_idx:]) if len(self.d_real_scores) > 0 else 0
            epoch_d_fake = np.mean(self.d_fake_scores[epoch_start_idx:]) if len(self.d_fake_scores) > 0 else 0
            
            with open(trainrecord_path, 'a') as f:
                f.write(f"Epoch {epoch+1},{epoch_d_loss:.6f},{epoch_g_loss:.6f},{epoch_d_real:.6f},{epoch_d_fake:.6f},{epoch_time:.2f}\n")
        
        print("训练完成！")
        
        # 最终评估
        print("进行最终评估...")
        final_results = evaluate_generator(
            self.netG, self.dataloader, self.device, 
            num_samples=min(10000, len(self.dataloader.dataset)), 
            nz=self.args.nz
        )
        
        final_eval_path = os.path.join(self.args.output_dir, 'final_evaluation.json')
        save_evaluation_results(final_results, final_eval_path)
        
        # 保存最终评估结果到eva.txt文件
        eva_path = os.path.join(self.args.output_dir, 'eva.txt')
        with open(eva_path, 'a') as f:
            fid = final_results.get('FID', 'N/A')
            is_mean = final_results.get('IS_mean', 'N/A')
            is_std = final_results.get('IS_std', 'N/A')
            lpips = final_results.get('LPIPS', 'N/A')
            f.write(f"Final,{fid},{is_mean},{is_std},{lpips}\n")
        
        # 保存最终模型
        self.save_checkpoint(self.args.num_epochs)
        
        # 绘制最终损失曲线
        self.plot_losses()

def main():
    parser = argparse.ArgumentParser(description='WGAN训练')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='./faces/faces', help='数据集路径')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--image_size', type=int, default=64, help='图像大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作进程数')
    
    # 网络参数
    parser.add_argument('--nz', type=int, default=100, help='噪声向量维度')
    parser.add_argument('--ngf', type=int, default=128, help='生成器特征图数量')
    parser.add_argument('--ndf', type=int, default=128, help='判别器特征图数量')
    parser.add_argument('--nc', type=int, default=3, help='图像通道数')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=400, help='训练轮数')
    parser.add_argument('--lr_g', type=float, default=0.00002, help='生成器学习率')
    parser.add_argument('--lr_d', type=float, default=0.0002, help='判别器学习率')
    parser.add_argument('--beta1', type=float, default=0.0, help='Adam优化器beta1参数')
    parser.add_argument('--lambda_gp', type=float, default=5.0, help='梯度惩罚权重')
    parser.add_argument('--n_critic', type=int, default=5, help='每次更新生成器前更新判别器的次数')
    
    # 输出参数
    parser.add_argument('--print_freq', type=int, default=100, help='打印频率')
    parser.add_argument('--sample_freq', type=int, default=5, help='生成样本频率')
    parser.add_argument('--save_freq', type=int, default=10, help='保存模型频率')
    parser.add_argument('--eval_freq', type=int, default=10, help='评估频率')
    parser.add_argument('--plot_freq', type=int, default=10, help='绘图频率')
    
    # 其他参数
    parser.add_argument('--resume', action='store_true', help='从检查点恢复训练')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 保存配置
    config_path = os.path.join(args.output_dir, 'config.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # 开始训练
    trainer = WGANTrainer(args)
    trainer.train()

if __name__ == '__main__':
    main()
