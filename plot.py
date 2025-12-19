#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WGAN训练过程可视化脚本
读取训练记录文件并生成多个独立的曲线图
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_training_data(file_path):
    """
    加载训练记录数据
    """
    try:
        # 读取CSV文件
        data = pd.read_csv(file_path)
        print(f"成功加载训练数据，共 {len(data)} 个epoch")
        return data
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def plot_discriminator_loss(data, output_dir):
    """
    绘制判别器损失曲线
    """
    plt.figure(figsize=(12, 8))
    epochs = data['Epoch'].str.extract(r'(\d+)')[0].astype(int)
    
    plt.plot(epochs, data['D_Loss'], 'b-', linewidth=2, label='判别器损失')
    plt.title('WGAN 判别器损失变化曲线', fontsize=16, fontweight='bold')
    plt.xlabel('训练轮数 (Epoch)', fontsize=14)
    plt.ylabel('判别器损失 (D_Loss)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # 添加趋势线
    z = np.polyfit(epochs, data['D_Loss'], 1)
    p = np.poly1d(z)
    plt.plot(epochs, p(epochs), 'r--', alpha=0.8, label=f'趋势线 (斜率: {z[0]:.4f})')
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'discriminator_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存: discriminator_loss.png")

def plot_generator_loss(data, output_dir):
    """
    绘制生成器损失曲线
    """
    plt.figure(figsize=(12, 8))
    epochs = data['Epoch'].str.extract(r'(\d+)')[0].astype(int)
    
    plt.plot(epochs, data['G_Loss'], 'g-', linewidth=2, label='生成器损失')
    plt.title('WGAN 生成器损失变化曲线', fontsize=16, fontweight='bold')
    plt.xlabel('训练轮数 (Epoch)', fontsize=14)
    plt.ylabel('生成器损失 (G_Loss)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # 添加趋势线
    z = np.polyfit(epochs, data['G_Loss'], 1)
    p = np.poly1d(z)
    plt.plot(epochs, p(epochs), 'r--', alpha=0.8, label=f'趋势线 (斜率: {z[0]:.4f})')
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'generator_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存: generator_loss.png")

def plot_discriminator_scores(data, output_dir):
    """
    绘制判别器对真实和虚假样本的评分曲线
    """
    plt.figure(figsize=(12, 8))
    epochs = data['Epoch'].str.extract(r'(\d+)')[0].astype(int)
    
    plt.plot(epochs, data['D_Real'], 'b-', linewidth=2, label='真实样本评分 (D_Real)', alpha=0.8)
    plt.plot(epochs, data['D_Fake'], 'r-', linewidth=2, label='虚假样本评分 (D_Fake)', alpha=0.8)
    
    plt.title('WGAN 判别器评分变化曲线', fontsize=16, fontweight='bold')
    plt.xlabel('训练轮数 (Epoch)', fontsize=14)
    plt.ylabel('判别器评分', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # 添加理想分离线
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='理想分离线')
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'discriminator_scores.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存: discriminator_scores.png")

def plot_training_time(data, output_dir):
    """
    绘制每个epoch的训练时间曲线
    """
    plt.figure(figsize=(12, 8))
    epochs = data['Epoch'].str.extract(r'(\d+)')[0].astype(int)
    
    plt.plot(epochs, data['Time(s)'], 'm-', linewidth=2, label='每轮训练时间')
    plt.title('WGAN 训练时间变化曲线', fontsize=16, fontweight='bold')
    plt.xlabel('训练轮数 (Epoch)', fontsize=14)
    plt.ylabel('训练时间 (秒)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # 添加平均时间线
    avg_time = data['Time(s)'].mean()
    plt.axhline(y=avg_time, color='orange', linestyle='--', alpha=0.8, 
                label=f'平均时间: {avg_time:.1f}秒')
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_time.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存: training_time.png")

def plot_loss_comparison(data, output_dir):
    """
    绘制判别器和生成器损失对比图
    """
    plt.figure(figsize=(12, 8))
    epochs = data['Epoch'].str.extract(r'(\d+)')[0].astype(int)
    
    plt.plot(epochs, data['D_Loss'], 'b-', linewidth=2, label='判别器损失 (D_Loss)', alpha=0.8)
    plt.plot(epochs, data['G_Loss'], 'g-', linewidth=2, label='生成器损失 (G_Loss)', alpha=0.8)
    
    plt.title('WGAN 损失对比曲线', fontsize=16, fontweight='bold')
    plt.xlabel('训练轮数 (Epoch)', fontsize=14)
    plt.ylabel('损失值', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # 添加零线
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存: loss_comparison.png")

def generate_summary_stats(data, output_dir):
    """
    生成训练统计摘要
    """
    epochs = data['Epoch'].str.extract(r'(\d+)')[0].astype(int)
    
    stats = {
        '总训练轮数': len(data),
        '最终判别器损失': data['D_Loss'].iloc[-1],
        '最终生成器损失': data['G_Loss'].iloc[-1],
        '平均训练时间': data['Time(s)'].mean(),
        '总训练时间': data['Time(s)'].sum() / 3600,  # 转换为小时
        '判别器损失变化': data['D_Loss'].iloc[-1] - data['D_Loss'].iloc[0],
        '生成器损失变化': data['G_Loss'].iloc[-1] - data['G_Loss'].iloc[0]
    }
    
    print("\n=== 训练统计摘要 ===")
    for key, value in stats.items():
        if '时间' in key and '总' in key:
            print(f"{key}: {value:.2f} 小时")
        elif '时间' in key:
            print(f"{key}: {value:.1f} 秒")
        else:
            print(f"{key}: {value:.4f}")
    
    # 保存统计信息到文件
    with open(os.path.join(output_dir, 'training_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("WGAN 训练统计摘要\n")
        f.write("=" * 30 + "\n")
        for key, value in stats.items():
            if '时间' in key and '总' in key:
                f.write(f"{key}: {value:.2f} 小时\n")
            elif '时间' in key:
                f.write(f"{key}: {value:.1f} 秒\n")
            else:
                f.write(f"{key}: {value:.4f}\n")
    
    print("已保存: training_summary.txt")

def load_evaluation_data(file_path):
    """
    加载评估数据
    """
    try:
        # 读取CSV文件
        data = pd.read_csv(file_path)
        print(f"成功加载评估数据，共 {len(data)} 个epoch")
        return data
    except Exception as e:
        print(f"加载评估数据失败: {e}")
        return None

def plot_fid_score(data, output_dir):
    """
    绘制FID分数曲线
    """
    plt.figure(figsize=(12, 8))
    epochs = data['Epoch'].str.extract(r'(\d+)')[0].astype(int)
    
    plt.plot(epochs, data['FID'], 'r-', linewidth=2, label='FID分数', marker='o', markersize=4)
    plt.title('WGAN FID分数变化曲线', fontsize=16, fontweight='bold')
    plt.xlabel('训练轮数 (Epoch)', fontsize=14)
    plt.ylabel('FID分数 (越低越好)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # 添加趋势线
    z = np.polyfit(epochs, data['FID'], 1)
    p = np.poly1d(z)
    plt.plot(epochs, p(epochs), 'b--', alpha=0.8, label=f'趋势线 (斜率: {z[0]:.4f})')
    plt.legend(fontsize=12)
    
    # 标注最佳值
    min_fid = data['FID'].min()
    min_epoch = epochs[data['FID'].idxmin()]
    plt.annotate(f'最佳FID: {min_fid:.2f}\n(Epoch {min_epoch})', 
                xy=(min_epoch, min_fid), xytext=(min_epoch+20, min_fid+10),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fid_score.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存: fid_score.png")



def main():
    """
    主函数
    """
    # 文件路径
    train_record_path = r'c:\Users\sanmu\Desktop\WGAN\output\trainrecord.txt'
    eva_record_path = r'c:\Users\sanmu\Desktop\WGAN\output\eva.txt'
    output_dir = r'c:\Users\sanmu\Desktop\WGAN\output\plots'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载训练数据
    print("正在加载训练数据...")
    train_data = load_training_data(train_record_path)
    
    # 加载评估数据
    print("正在加载评估数据...")
    eva_data = load_evaluation_data(eva_record_path)
    
    print(f"\n开始生成可视化图表...")
    print(f"输出目录: {output_dir}")
    
    # 生成训练相关图表
    if train_data is not None:
        plot_discriminator_loss(train_data, output_dir)
        plot_generator_loss(train_data, output_dir)
        plot_discriminator_scores(train_data, output_dir)
        plot_training_time(train_data, output_dir)
        plot_loss_comparison(train_data, output_dir)
        generate_summary_stats(train_data, output_dir)
    
    # 生成评估相关图表
    if eva_data is not None:
        plot_fid_score(eva_data, output_dir)    
    print(f"\n所有图表已生成完成！")
    print(f"请查看输出目录: {output_dir}")
    print("\n生成的图表文件:")
    if train_data is not None:
        print("训练相关图表:")
        print("- discriminator_loss.png (判别器损失曲线)")
        print("- generator_loss.png (生成器损失曲线)")
        print("- discriminator_scores.png (判别器评分曲线)")
        print("- training_time.png (训练时间曲线)")
        print("- loss_comparison.png (损失对比曲线)")
        print("- training_summary.txt (训练统计摘要)")
    if eva_data is not None:
        print("评估相关图表:")
        print("- fid_score.png (FID分数曲线)")
        print("- is_score.png (IS分数曲线)")
        print("- evaluation_comparison.png (评估指标对比)")
        print("- evaluation_summary.txt (评估统计摘要)")

if __name__ == '__main__':
    main()