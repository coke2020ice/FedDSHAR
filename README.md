# FedDSHAR

**FedDSHAR**: A Dual-Strategy Federated Learning Approach for Human Activity Recognition under Extremely Noisy Label Conditions

---

## Abstract

Federated learning (FL) has recently achieved remarkable success in privacy-sensitive healthcare applications, including medical analysis. However, most previous studies assume that collected user data are well-annotated—a strong assumption in practice. For instance, in the human activity recognition (HAR) task, the goal is to train a model that predicts a person’s activity based on sensor data collected over a period of time. Due to diverse and incomplete annotation methods, user-side data invariably contain significant label noise, which can greatly hinder model convergence and degrade performance. To address this issue, we propose **FedDSHAR**, a novel federated learning framework that partitions user-side data into clean and noisy subsets. Two distinct strategies are applied on the respective subsets: strategic time-series augmentation on the clean data and a semi-supervised learning scheme for the noisy data. Extensive experiments on three public, real-world HAR datasets demonstrate that FedDSHAR outperforms six state-of-the-art methods, particularly in scenarios with extreme label noise encountered in distributed noisy HAR applications.

---

## Project Structure


---

## Installation

1. **Create a Python virtual environment (recommended):**

    ```bash
    conda create -n feddshar python=3.7.9
    conda activate feddshar
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

1. Prepare your dataset by placing it in the `dataset/` directory.
2. Run the training script:

    ```bash
    python train_fed_dshar.py
    ```

3. To run all experiments, use:

    ```bash
    python run_all.py
    ```

---

## Benchmark

For benchmark datasets, please refer to the [HARBOX dataset](https://github.com/xmouyang/FL-Datasets-for-HAR).

---

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

---

## Main Baselines

- **FedAvg**  
  [Paper](http://proceedings.mlr.press/v54/mcmahan17a?ref=https://githubhelp.com)
- **FedProx**  
  [Paper](https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html)
- **RoFL**  
  [Paper](https://ieeexplore.ieee.org/abstract/document/9713942) | [Code](https://github.com/jangsoohyuk/Robust-Federated-Learning-with-Noisy-Labels)
- **FedLSR**  
  [Paper](https://dl.acm.org/doi/abs/10.1145/3511808.3557475) | [Code](https://github.com/Sprinter1999/FedLSR)
- **FedCorr**  
  [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Xu_FedCorr_Multi-Stage_Federated_Learning_for_Label_Noise_Correction_CVPR_2022_paper.html) | [Code](https://github.com/Xu-Jingyi/FedCorr)
- **FedNoRo**  
  [Paper](https://arxiv.org/abs/2305.05230) | [Code](https://github.com/wnn2000/FedNoRo/stargazers)

---

## Publication Pending

The contents of this repository will be made publicly available once the associated research paper is accepted for publication. Thank you for your understanding and patience.

---

## License

[To be determined upon publication]
