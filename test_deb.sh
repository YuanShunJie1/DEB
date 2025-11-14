# test_deft
# --tau      0.02    0.05     0.1      0.2     0.5  
# --lambda_w 0.1     0.2      0.5      1.0     2.0   

# Main experiments
python test_deft.py --dataset aids --epochs 100 --attack_epoch 80 --target_label 0 --tau 0.2 --lambda_w 0.5
python test_deft.py --dataset letter --epochs 100 --attack_epoch 80 --target_label 0 --tau 0.2 --lambda_w 0.5
python test_deft.py --dataset mnist --epochs 100 --attack_epoch 80 --target_label 2 --tau 0.2 --lambda_w 0.5
python test_deft.py --dataset fmnist --epochs 100 --attack_epoch 80 --target_label 0 --tau 0.2 --lambda_w 0.5
python test_deft.py --dataset cifar10 --epochs 100 --attack_epoch 80 --target_label 0 --tau 0.2 --lambda_w 0.5
python test_deft.py --dataset cinic10 --epochs 100 --attack_epoch 80 --target_label 0 --tau 0.2 --lambda_w 0.5
python test_deft.py --dataset imagenet12 --epochs 100 --attack_epoch 80 --target_label 0 --tau 0.2 --lambda_w 0.5 --batch_size 32

