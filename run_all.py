"""
run_all.py
----------
Entry file for integrating teacher model training and federated DS-HAR training. The process is as follows:

Parse command-line arguments (see options.py).

If --teacher_train is set to True, train the teacher model; otherwise, load the teacher model directly.

Use the teacher model to initiate the federated DS-HAR training.
"""

import os
import torch
from utils.options import args_parser
from MR_model import MR_model
from train_fed_dshar import train_fed_dshar

def main():
    args = args_parser()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = "cuda" if torch.cuda.is_available() and args.device=="cuda" else "cpu"
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    torch.manual_seed(args.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # --------------------- （MR Model） ---------------------

    MR_model(args)

    print("FedHARDS-HAR train...")
    
    train_fed_dshar(args)

if __name__ == '__main__':
    main()
