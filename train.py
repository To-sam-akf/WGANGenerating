#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化的训练启动脚本
提供预设的训练配置，方便快速开始训练
"""

import os
import sys
import subprocess
import argparse

def check_data_directory(data_dir):
    """检查数据目录是否存在"""
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录 {data_dir} 不存在")
        print("请确保已将face.zip解压到正确位置")
        return False
    
    # 检查是否有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_count = 0
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_count += 1
                if image_count >= 10:  # 找到足够的图像文件就停止
                    break
        if image_count >= 10:
            break
    
    if image_count < 10:
        print(f"警告: 在 {data_dir} 中只找到 {image_count} 个图像文件")
        print("请确保数据集已正确解压")
        return False
    
    print(f"数据目录检查通过，找到图像文件")
    return True

def run_training(config_name, custom_args=None):
    """运行训练"""
    
    # 预设配置
    configs = {
        'quick': {
            'description': '快速测试配置 (小批次，少轮数)',
            'args': [
                '--batch_size', '32',
                '--num_epochs', '50',
                '--eval_freq', '10',
                '--sample_freq', '5'
            ]
        },
        'standard': {
            'description': '标准训练配置',
            'args': [
                '--batch_size', '64',
                '--num_epochs', '200',
                '--eval_freq', '20',
                '--sample_freq', '5'
            ]
        },
        'high_quality': {
            'description': '高质量训练配置 (大批次，多轮数)',
            'args': [
                '--batch_size', '128',
                '--num_epochs', '300',
                '--lr_g', '0.0001',
                '--lr_d', '0.0003',
                '--eval_freq', '25',
                '--sample_freq', '10'
            ]
        },
        'gpu_optimized': {
            'description': 'GPU优化配置 (适合高端GPU)',
            'args': [
                '--batch_size', '256',
                '--num_epochs', '400',
                '--num_workers', '8',
                '--eval_freq', '30',
                '--sample_freq', '10'
            ]
        }
    }
    
    if config_name not in configs:
        print(f"错误: 未知配置 '{config_name}'")
        print("可用配置:")
        for name, config in configs.items():
            print(f"  {name}: {config['description']}")
        return False
    
    # 检查数据目录
    data_dir = './faces/faces'
    if not check_data_directory(data_dir):
        return False
    
    # 构建命令
    cmd = ['python', 'main.py'] + configs[config_name]['args']
    
    # 添加自定义参数
    if custom_args:
        cmd.extend(custom_args)
    
    print(f"使用配置: {config_name} - {configs[config_name]['description']}")
    print(f"执行命令: {' '.join(cmd)}")
    print("\n开始训练...")
    
    try:
        # 运行训练
        result = subprocess.run(cmd, check=True)
        print("\n训练完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n训练失败，错误码: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        return False

def main():
    parser = argparse.ArgumentParser(description='WGAN训练启动脚本')
    parser.add_argument('config', nargs='?', default='standard', 
                       help='训练配置 (quick/standard/high_quality/gpu_optimized)')
    parser.add_argument('--list-configs', action='store_true', 
                       help='列出所有可用配置')
    parser.add_argument('--resume', action='store_true', 
                       help='从检查点恢复训练')
    parser.add_argument('--output-dir', type=str, 
                       help='自定义输出目录')
    
    args, unknown_args = parser.parse_known_args()
    
    # 列出配置
    if args.list_configs:
        configs = {
            'quick': '快速测试配置 (小批次，少轮数)',
            'standard': '标准训练配置',
            'high_quality': '高质量训练配置 (大批次，多轮数)',
            'gpu_optimized': 'GPU优化配置 (适合高端GPU)'
        }
        print("可用训练配置:")
        for name, desc in configs.items():
            print(f"  {name}: {desc}")
        return
    
    # 构建自定义参数
    custom_args = unknown_args.copy()
    
    if args.resume:
        custom_args.append('--resume')
    
    if args.output_dir:
        custom_args.extend(['--output_dir', args.output_dir])
    
    # 运行训练
    success = run_training(args.config, custom_args)
    
    if success:
        print("\n训练成功完成！")
        print("检查输出目录中的结果:")
        output_dir = args.output_dir if args.output_dir else './output'
        print(f"  - 生成的图像: {output_dir}/images/")
        print(f"  - 模型检查点: {output_dir}/checkpoints/")
        print(f"  - 评估结果: {output_dir}/final_evaluation.json")
    else:
        print("\n训练失败，请检查错误信息")
        sys.exit(1)

if __name__ == '__main__':
    main()