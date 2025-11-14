# test_deft
# --tau      0.01  0.02    0.05    0.1     0.2  
# --lambda_w 0.1   0.2     0.5     1.0     2.0    5.0   



python basl.py --dataset aids --epochs 100 --attack_epoch 80 --target_label 0 
python basl.py --dataset letter --epochs 100 --attack_epoch 80 --target_label 0 
python basl.py --dataset mnist --epochs 100 --attack_epoch 80 --target_label 2 
python basl.py --dataset fashionmnist --epochs 100 --attack_epoch 90 --target_label 0
python basl.py --dataset cifar10 --epochs 100 --attack_epoch 90 --target_label 0
python basl.py --dataset cinic10 --epochs 100 --attack_epoch 90 --target_label 0
python basl.py --dataset imagenet12 --epochs 100 --attack_epoch 50 --target_label 0 --batch_size 32




