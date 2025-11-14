# DEB: Detecting and Eliminating Backdoors in Split Neural Network-Based Vertical Federated Learning

This repository contains the official implementation of the paper:

**Detecting and Eliminating Backdoors in Split Neural Network-Based Vertical Federated Learning (DEB)**

DEB is a defense framework designed to **detect** and **eliminate** backdoors in Split Neural Network-based Vertical Federated Learning (VFL). This codebase includes:
- Backdoor attack implementation (BASL)
- The DEB defense method


## 🚀 Usage
You need to run **three main steps**:

### 1. Download the required datasets into the directory: datasets

### 2. Run the Backdoor Attack (BASL)
This simulates the malicious participant embedding a backdoor:
```bash
python basl.py --dataset cifar10 --epochs 100 --attack_epoch 90 --target_label 0
```

### 3. Run DEB Defense on the Last Epoch
After the backdoor model is trained, run DEB to detect and remove backdoors:
```bash
python test_deb.py --dataset cifar10 --epochs 100 --attack_epoch 80 --target_label 0 --tau 0.2 --lambda_w 0.5
```
tau and lambda_w are two hyperparameters corresponding to τ and λ in the paper.

## 🙏 Acknowledgements
We would like to thank the following works for their inspiration and contribution to this project:

1. **Y. He, Z. Shen, J. Hua, Q. Dong, J. Niu, W. Tong, X. Huang, C. Li, and S. Zhong**,  
   *Backdoor attack against split neural network-based vertical federated learning*,  
   IEEE Transactions on Information Forensics and Security.

2. **P. Chen, J. Yang, J. Lin, Z. Lu, Q. Duan, and H. Chai**,  
   *A practical clean-label backdoor attack with limited information in vertical federated learning*,  
   in Proceedings of the IEEE International Conference on Data Mining (ICDM 2023).


