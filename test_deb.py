import argparse
import os
# import utils.models_ours as models, utils.datasets as datasets
import utils.models as models, utils.datasets as datasets

from torch.utils.data import DataLoader
import torchvision

import torch
import random
from torch.utils.data import Dataset

# from basl.ubd import UBDDefense
from attackers.basl import BASL
from deb import DEB

torch.autograd.set_detect_anomaly(True)

class TempDataset(torch.utils.data.Dataset):
    def __init__(self, full_dataset=None, transform=None):
        self.dataset = full_dataset
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]
        if self.transform:
            image = self.transform(image)
        return image, label, index

    def __len__(self):
        return self.dataLen

class BackdoorTestDataset(torch.utils.data.Dataset):
    def __init__(self, full_dataset=None, transform=None, target_label=0):
        
        self.dataset = []
        self.transform = transform
        
        for i in range(len(full_dataset)):
            if full_dataset[i][1] != target_label:
                self.dataset.append([full_dataset[i][0], full_dataset[i][1]])
        
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]
        if self.transform:
            image = self.transform(image)
        return image, label, index

    def __len__(self):
        return self.dataLen

    # "mnist",
    # "fashionmnist",
    # "cifar10",
    # "cifar100",
    # "criteo",
    # "cinic10"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        help='the datasets for evaluation;',
                        type=str,
                        choices=datasets.datasets_choices,
                        default='aids')
    parser.add_argument('--epochs',
                        help='the number of epochs;',
                        type=int,
                        default=100)
    parser.add_argument('--attack_epoch',
                        help='set epoch for attacking, greater than or equal to 2;',
                        type=int,
                        default=80)
    parser.add_argument('--batch_size',
                        help='batch size;',
                        type=int,
                        default=128)
    parser.add_argument('--lr_passive',
                        help='learning rate for the passive parties;',
                        type=float,
                        default=0.1)
    parser.add_argument('--lr_active',
                        help='learning rate for the active party;',
                        type=float,
                        default=0.1)
    parser.add_argument('--lr_attack',
                        help='learning rate for the attacker;',
                        type=float,
                        default=0.1)
    parser.add_argument('--attack_id',
                        help="the ID list of the attacker, like ``--attack_id 0 1'' for [0,1];",
                        nargs='*',
                        type=int,
                        default=0)
    parser.add_argument('--num_passive',
                        help='number of passive parties;',
                        type=int,
                        default=2)
    parser.add_argument('--division',
                        help='choose the data division mode;',
                        type=str,
                        choices=['vertical', 'random', 'imbalanced'],
                        default='vertical')
    parser.add_argument('--round',
                        help='round for log;',
                        type=int,
                        default=0)
    parser.add_argument('--target_label',
                        help='target label, which aim to change to;',
                        type=int,
                        default=0)
    parser.add_argument('--source_label',
                        help='source label, which aim to change;',
                        type=int,
                        default=1)
    parser.add_argument('--trigger',
                        help='set trigger type;',
                        type=str,
                        # choices=['badvfl', 'villain', 'badvfl', 'tifs', 'icdm'],
                        default='tifs')
    parser.add_argument('--schedule', 
                        type=int, 
                        nargs='+', 
                        default=[50, 70])

    parser.add_argument('--gpuid', type=int,  default=0)
    parser.add_argument('--num_classes', type=int,  default=10)
    # self.lambda_w
    parser.add_argument('--lambda_w', type=float,  default=1.0)
    parser.add_argument('--tau', type=float,  default=0.1)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpuid)
    random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    device = torch.device(f'cuda:{args.gpuid}')
    
    # load dataset
    dir = "/".join(os.path.abspath(__file__).split("/")[:-1])
    dataset_path = os.path.join(dir, 'dataset')

    if args.dataset == "cinic10":
        args.num_classes = 10
    elif args.dataset == "cdc":
        args.num_classes = 2
    elif args.dataset == "aids":
        args.num_classes = 2
    elif args.dataset == "cifar100":
        args.num_classes = 100
    elif args.dataset == "cifar10":
        args.num_classes = 10
    elif args.dataset == "imagenet12":
        args.num_classes = 12
    elif args.dataset == "mnist" or args.dataset == "fmnist":
        args.num_classes = 10
    elif args.dataset == "letter":
        args.num_classes = 26

    if args.dataset == "cinic10":
        data_train = torchvision.datasets.ImageFolder(dataset_path + '/CINIC10/train')
        temp_dataset = TempDataset(full_dataset=data_train, transform=datasets.transforms_train_augment[args.dataset])
        dataloader_train = DataLoader(temp_dataset, batch_size=args.batch_size, shuffle=False)
    elif args.dataset == "cdc" or args.dataset == "aids" or args.dataset == "letter":
        data_train = datasets.datasets_dict[args.dataset](train=True)
        temp_dataset = TempDataset(full_dataset=data_train)
        dataloader_train = DataLoader(temp_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        data_train = datasets.datasets_dict[args.dataset](dataset_path, train=True, download=True)
        temp_dataset = TempDataset(full_dataset=data_train, transform=datasets.transforms_train_augment[args.dataset])
        dataloader_train = DataLoader(temp_dataset, batch_size=args.batch_size, shuffle=False)

    if args.dataset == "cinic10":
        data_test = torchvision.datasets.ImageFolder(dataset_path + '/CINIC10/test')
        temp_dataset = TempDataset(full_dataset=data_test, transform=datasets.transforms_test_augment[args.dataset])
        dataloader_test = DataLoader(temp_dataset, batch_size=64, shuffle=True)
        test_dataset = torchvision.datasets.ImageFolder(dataset_path + '/CINIC10/test')
    elif args.dataset == "cdc" or args.dataset == "aids" or args.dataset == "letter":
        data_test = datasets.datasets_dict[args.dataset](train=False)
        temp_dataset = TempDataset(full_dataset=data_test)
        dataloader_test = DataLoader(temp_dataset, batch_size=64, shuffle=True)        
    else:
        data_test = datasets.datasets_dict[args.dataset](dataset_path, train=False, download=True)
        temp_dataset = TempDataset(full_dataset=data_test, transform=datasets.transforms_test_augment[args.dataset])
        dataloader_test = DataLoader(temp_dataset, batch_size=64, shuffle=True)

    model_name = f'attacked_models/entire_model_dataset={args.dataset}_epochs={args.epochs}_attack_epoch={args.attack_epoch}_batch_size={args.batch_size}_trigger={args.trigger}_attack_id={args.attack_id}_num_passive={args.num_passive}_target_label={args.target_label}.pth'
    
    trigger_name = f'attacked_models/entire_model_dataset={args.dataset}_epochs={args.epochs}_attack_epoch={args.attack_epoch}_batch_size={args.batch_size}_trigger={args.trigger}_attack_id={args.attack_id}_num_passive={args.num_passive}_target_label={args.target_label}_trigger.pth'
    
    top_tunned_name = f'results_deb/tunned_top_dataset={args.dataset}_epochs={args.epochs}_attack_epoch={args.attack_epoch}_batch_size={args.batch_size}_trigger={args.trigger}_attack_id={args.attack_id}_num_passive={args.num_passive}_target_label={args.target_label}_tau={args.tau}_lambda_w={args.lambda_w}.pth'
    
    tunned_results_log=open(f'results_deb/results_tunned_top_dataset={args.dataset}_epochs={args.epochs}_attack_epoch={args.attack_epoch}_batch_size={args.batch_size}_trigger={args.trigger}_attack_id={args.attack_id}_num_passive={args.num_passive}_target_label={args.target_label}_tau={args.tau}_lambda_w={args.lambda_w}.txt','w')
    
    entire_model = torch.load(model_name, weights_only=False)
    entire_model = entire_model.to(device)

    trigger = torch.load(trigger_name, weights_only=False)
    trigger = trigger.to(device)    

    basl = BASL(args,entire_model,dataloader_train,dataloader_test,device,trigger=trigger)

    acc_b = basl.test()
    asr_b = basl.backdoor()

    print(f"[Before Tuning] ACC:{acc_b:.4f}, ASR:{asr_b:.4f}")

    deft = DEB(args, entire_model, dataloader_train, dataloader_test, device, trigger=trigger,auxiliary_index=basl.auxiliary_index, threshold=args.tau)

    top_tuned, metrics = deft.evaluate_defense(lr=0.001, fine_tune_epochs=100, top_tunned_name=top_tunned_name)

    acc_a = deft.test()
    asr_a = deft.backdoor(top_tuned=top_tuned)
    
    print(f"[After Tuning] ACC:{acc_a:.4f}, ASR:{asr_a:.4f}")

    if metrics != None:
        tunned_results_log.write(
            f"[Before Tuning] ACC:{acc_b:.4f}, ASR:{asr_b:.4f}\n"
            f"[After Tuning] ACC:{acc_a:.4f}, ASR:{asr_a:.4f}\n"
            f"[Entropy-Based Backdoor Detection Metrics] Precision={metrics[0][0]:.4f}, Recall={metrics[0][1]:.4f}, F1={metrics[0][2]:.4f}\n"
            f"[Distance-Based Backdoor Detection Metrics] Precision={metrics[1][0]:.4f}, Recall={metrics[1][1]:.4f}, F1={metrics[1][2]:.4f}\n"
        )
        tunned_results_log.flush()

if __name__ == '__main__':
    main()
