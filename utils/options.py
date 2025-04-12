import argparse

def args_parser():
    parser = argparse.ArgumentParser(
        description="Federated DS-HAR Training with Teacher Model Integration"
    )

    # --------------------------------------------------------------------------
    # System Settings
    # --------------------------------------------------------------------------
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic training")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES setting")
    parser.add_argument("--device", type=str, default="cuda", help="Device type: 'cuda' or 'cpu'")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for logs and models")

    # --------------------------------------------------------------------------
    # Basic Training Settings
    # --------------------------------------------------------------------------
    parser.add_argument("--exp", type=str, default="Fed", help="Experiment name")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size per GPU")
    parser.add_argument("--base_lr", type=float, default=0.0001, help="Base learning rate")
    parser.add_argument("--pretrained", type=int, default=0, help="Flag for using pretrained models (0: no, 1: yes)")
    parser.add_argument("--dataset", type=str, default="harbox", help="Name of the dataset to use")
    # --------------------------------------------------------------------------
    # Federated Learning Settings
    # --------------------------------------------------------------------------
    parser.add_argument("--n_clients", type=int, default=30, help="Number of federated training clients")
    parser.add_argument("--iid", action="store_true", help="Use IID data distribution; if not set, non-IID is used")

    parser.add_argument("--non_iid_prob", type=float, default=0.9, help="Probability parameter for non-IID sampling (default: 0.9)")
    parser.add_argument("--alpha_dirichlet", type=float, default=0.8, help="Dirichlet parameter for non-IID sampling (default: 0.8)")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes in the dataset (default: 5)")
    parser.add_argument("--local_ep", type=int, default=5, help="Number of local epochs")

    # --------------------------------------------------------------------------
    # Stage Settings
    # --------------------------------------------------------------------------
    parser.add_argument("--begin", type=int, default=10, help="Ramp-up begin epoch")
    parser.add_argument("--end", type=int, default=49, help="Ramp-up end epoch")
    parser.add_argument("--a", type=float, default=0.8, help="Consistency weight factor")

    # --------------------------------------------------------------------------
    # Noise Settings
    # --------------------------------------------------------------------------
    parser.add_argument("--level_n_system", type=float, default=1, help="Fraction of noisy clients")
    parser.add_argument("--level_n_lowerb", type=float, default=0.6, help="Lower bound of noise level")
    parser.add_argument("--level_n_upperb", type=float, default=0.9, help="Upper bound of noise level")
    parser.add_argument("--n_type", type=str, default="instance", help="Type of noise")
    parser.add_argument("--p", type=float, default=1.0, 
                        help="High-noise class fraction parameter (e.g., 0.5 means 50% of classes are high-noise); default is 0.5")
    parser.add_argument("--high_noise_lower", type=float, default=0.5, 
                        help="Lower bound of noise rate for high-noise classes; default is 0.5")
    parser.add_argument("--high_noise_upper", type=float, default=1.0, 
                        help="Upper bound of noise rate for high-noise classes; default is 1.0")
    parser.add_argument("--low_noise_lower", type=float, default=0.0, 
                        help="Lower bound of noise rate for low-noise classes; default is 0.0")
    parser.add_argument("--low_noise_upper", type=float, default=0.5, 
                        help="Upper bound of noise rate for low-noise classes; default is 0.5")

    # --------------------------------------------------------------------------
    # Teacher Model Training Parameters
    # --------------------------------------------------------------------------
    parser.add_argument("--teacher_train", action="store_true", help="Whether to retrain the teacher model")
    parser.add_argument("--teacher_epochs", type=int, default=100, help="Number of epochs for teacher model training")
    parser.add_argument("--teacher_lr", type=float, default=0.0001, help="Learning rate for the teacher model")
    parser.add_argument("--teacher_save_path", type=str, default="MR_model.pth",
                        help="File path to save the teacher model")

    # --------------------------------------------------------------------------
    # Federated DS-HAR Training Parameters
    # --------------------------------------------------------------------------
    parser.add_argument("--fed_epochs", type=int, default=100, help="Number of federated training epochs")
    parser.add_argument("--lr_fed", type=float, default=0.0001, help="Learning rate for federated training")
    parser.add_argument("--clip_grad", type=float, default=100.0, help="Gradient clipping threshold")
    # Uncomment the following if using domain adaptation loss
    # parser.add_argument("--dp_loss_type", type=str, default="dis", help="Domain adaptation loss type")
    # parser.add_argument("--dp_input_dim", type=int, default=725, help="Input dimension for domain adaptation loss")
    # parser.add_argument("--dp_weight", type=float, default=1.0, help="Weight for the domain adaptation loss")

    args = parser.parse_args()
    return args
