# FedDSHAR
FedDSHAR: A Dual-Strategy Federated Learning Approach for Human Activity Recognition amid Extremely Noise Label User

## Overview
FedDSHAR is a novel federated learning framework designed for Human Activity Recognition (HAR) that addresses the challenge of noisy labels in user data. The framework leverages silver-standard HAR datasets from user devices to train a robust global model with strong generalization capabilities in a federated learning setting.

## Project Structure
```
FedDSHAR/
├── dataset/          # Dataset processing and management
├── model/            # Model architectures and implementations
├── utils/            # Utility functions and helper modules
├── figures/          # Experimental results and visualizations
├── output/           # Output files and model checkpoints
├── train_fed_dshar.py    # Main training script
├── run_all.py        # Script to run all experiments
├── MR_model.py       # Model implementation
└── requirements.txt   # Python dependencies
```

## Installation
1. Create a Python virtual environment (recommended):
```bash
conda create -n feddshar python=3.7.9
conda activate feddshar
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Prepare your dataset in the `dataset/` directory
2. Run the training script:
```bash
python train_fed_dshar.py
```

3. For running all experiments:
```bash
python run_all.py
```

## Dependencies
- Python 3.7.9
- PyTorch 1.11.0
- torchvision 0.12.0
- numpy 1.21.5
- pandas 1.4.2
- scikit-learn 1.0.2
- efficientnet-pytorch 0.7.1
- pretrainedmodels 0.7.4
- tensorboardx 2.2
- pillow 9.0.1

## Main Baselines
- FedAvg [[paper](http://proceedings.mlr.press/v54/mcmahan17a?ref=https://githubhelp.com)]
- FedProx [[paper](https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html)]
- RoFL [[paper](https://ieeexplore.ieee.org/abstract/document/9713942)] [[code](https://github.com/jangsoohyuk/Robust-Federated-Learning-with-Noisy-Labels)]
- FedLSR [[paper](https://dl.acm.org/doi/abs/10.1145/3511808.3557475)] [[code](https://github.com/Sprinter1999/FedLSR)]
- FedCorr [[paper](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_FedCorr_Multi-Stage_Federated_Learning_for_Label_Noise_Correction_CVPR_2022_paper.html)] [[code](https://github.com/Xu-Jingyi/FedCorr)]
- Fedrono [[paper](https://arxiv.org/abs/2305.05230)][[code](https://github.com/wnn2000/FedNoRo/stargazers)]

## Publication Pending
The content of this repository will be made publicly available once the associated research paper is accepted for publication. Thank you for your understanding and patience.

## License
[To be determined upon publication]
