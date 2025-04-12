"""
run_all.py
----------
整合教师模型训练和联邦 DS-HAR 训练的入口文件。流程如下：
  1. 解析命令行参数（见 options.py）。
  2. 若 --teacher_train 设置为 True，则训练教师模型，否则直接加载教师模型。
  3. 利用教师模型启动联邦 DS-HAR 训练。
"""

import os
import torch
from utils.options import args_parser
from MR_model import MR_model
from train_fed_dshar import train_fed_dshar

def main():
    args = args_parser()
    
    # 配置 GPU（若使用 GPU）
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = "cuda" if torch.cuda.is_available() and args.device=="cuda" else "cpu"
    
    # 设置输出目录，若不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 设置随机种子与确定性（此处可自行扩展）
    torch.manual_seed(args.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # --------------------- 教师模型（Teacher Model） ---------------------

    MR_model(args)

    print("开始联邦 DS-HAR 训练...")
    
    train_fed_dshar(args)

if __name__ == '__main__':
    main()
