from torchvision import datasets, transforms
from torch.utils.data import Dataset
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torchvision
from PIL import Image
from torchvision.datasets import VisionDataset
from sklearn.model_selection import train_test_split
import numpy as np
import torch

from ucimlrepo import fetch_ucirepo 
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler



feature_sizes = []


class ImageNet12(Dataset):
    def __init__(self, root="./dataset", train=True, transform=None, **kwargs):
        super().__init__()
        self.train = train
        self.transform = transform
        
        root = os.path.join(root, "imagenet12")

        if train:
            fold = '/home/shunjie/codes/DEFT/basl/data/imagenet12/train'
        else:
            fold = '/home/shunjie/codes/DEFT/basl/data/imagenet12/val'
        self.dataset = torchvision.datasets.ImageFolder(root=fold, transform=self.transform)

    def __getitem__(self, idx):
        img, label = self.dataset[idx][0], self.dataset[idx][1]
        return img, label, idx

    def __len__(self):
        return len(self.dataset)

class CINIC10(VisionDataset):
    def __init__(self, root="./dataset", train=True, transform=None, **kwargs):
        super(CINIC10, self).__init__(root, transform=transform)
        self.train = train
        root = os.path.join(root, "CINIC10")

        if not os.path.exists(root):
            raise ValueError("You should download and unzip the CINIC10 dataset to {} first! Download: https://github.com/BayesWatch/cinic-10".format(root))

        if train:
            fold = '/train'
        else:
            fold = '/test'
        image = torchvision.datasets.ImageFolder(root=root + fold, transform=transform)
        self.data = image.imgs
        self.transform = transform

    def __getitem__(self, idx):
        path, label = self.data[idx]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, idx

    def __len__(self):
        return len(self.data)


class Criteo(Dataset):
    '''
    To load Criteo dataset.
    '''
    def __init__(self, root="./dataset", train=True, **kwargs):
        self.train = train
        root = os.path.join(root, "criteo")

        if not os.path.exists(root):
            raise ValueError("You should download and unzip the Criteo dataset to {} first! Download: https://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz".format(root))
        
        # sample data
        file_out = "train_sampled.txt"
        outpath = os.path.join(root, file_out)
        if not os.path.exists(outpath):
            file_in = "train.txt"
            self.csv_data = pd.read_csv(os.path.join(root, file_in), sep='\t', nrows=70000, index_col=None)

            cols = self.csv_data.columns.values
            for idx, col in enumerate(cols):
                if idx > 0 and idx <= 13:
                    self.csv_data[col] = self.csv_data[col].fillna(0,)
                elif idx >= 14:
                    self.csv_data[col] = self.csv_data[col].fillna('-1',)

            self.csv_data.to_csv(outpath, sep='\t', index=False)
            print("Dataset sampling completed.")
        
        # process data
        file_out = "train_processed.txt"
        outpath = os.path.join(root, file_out)
        if not os.path.exists(outpath):
            file_in = "train_sampled.txt"
            self.csv_data = pd.read_csv(os.path.join(root, file_in), sep='\t', index_col=None)

            cols = self.csv_data.columns.values
            for idx, col in enumerate(cols):
                le = LabelEncoder()
                le.fit(self.csv_data[col])
                self.csv_data[col] = le.transform(self.csv_data[col])

            self.csv_data.to_csv(outpath, sep='\t', index=False)
            print("Dataset processing completed.")

        self.csv_data = pd.read_csv(outpath, sep='\t', index_col=None)
        if train:
            global feature_sizes
            feature_sizes.clear()
            cols = self.csv_data.columns.values
            for col in cols:
                feature_sizes.append(len(self.csv_data[col].value_counts()))
            feature_sizes.pop(0)  # do not contain label

        self.train_data, self.test_data = train_test_split(self.csv_data, test_size=1/7, random_state=42)
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
    
    def __getitem__(self, idx):
        if self.train:
            x = self.train_data.iloc[idx].values
        else:
            x = self.test_data.iloc[idx].values
        x = np.array(x, dtype=np.float32)
        return x[1:], int(x[0]), idx


class AIDSDataset(Dataset):
    def __init__(self, train=True, root="./dataset", test_size=0.25, random_seed=42):
        super(AIDSDataset, self).__init__()
        self.root = root
        self.train = train

        # 下载 Abalone 数据集
        abalone = fetch_ucirepo(id=890)

        # 提取特征和目标变量
        X = abalone.data.features.values  # 转为 NumPy 数组
        y = abalone.data.targets.values.flatten()  # 确保 y 是 1D

        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=y
        )

        # 根据 train 参数选择数据
        if self.train:
            self.X, self.y = self.X_train, self.y_train
        else:
            self.X, self.y = self.X_test, self.y_test

    def __getitem__(self, idx):
        features = self.X[idx]
        label = self.y[idx]

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.X)



class CreditDataset(Dataset):
    def __init__(self, train=True,
                 root="/home/shunjie/codes/DEFT/basl/dataset/credit/",
                 train_file="cs-training.csv",
                 test_file="cs-test.csv"):
        super(CreditDataset, self).__init__()
        self.train = train
        self.root = root
        self.train_path = os.path.join(root, train_file)
        self.test_path = os.path.join(root, test_file)

        # 选择数据文件
        file_path = self.train_path if train else self.test_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # 读取数据
        df = pd.read_csv(file_path)

        # 删除无用索引列
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        # 训练集标签列为 "SeriousDlqin2yrs"
        if train:
            y = df["SeriousDlqin2yrs"].values.astype(np.int64)
            X = df.drop(columns=["SeriousDlqin2yrs"])
        else:
            # 测试集不含标签，这里假设你只想加载特征
            y = np.zeros(len(df), dtype=np.int64)  # 占位符
            X = df

        # 缺失值填充（用中位数）
        X = X.fillna(X.median())

        # 数值标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 转换为 Tensor
        self.X = torch.tensor(X_scaled, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]




import pickle

class CDCDataset(Dataset):
    def __init__(self, train=True, root="./dataset", test_size=0.25, random_seed=42):
        super(CDCDataset, self).__init__()
        self.root = root
        self.train = train
        os.makedirs(self.root, exist_ok=True)
        
        cache_file = os.path.join("/home/shunjie/codes/DEFT/datasets/letter/", "letter.pkl")
        
        # 如果本地缓存存在，直接加载
        if os.path.exists(cache_file):
            print("加载本地缓存数据集...")
            with open(cache_file, "rb") as f:
                cdc_data = pickle.load(f)
        else:
            print("下载 Letter 数据集...")
            cdc_data = fetch_ucirepo(id=59) 
            with open(cache_file, "wb") as f:
                pickle.dump(cdc_data, f)
        
        # 提取特征和目标变量
        X = cdc_data.data.features.values  # 转为 NumPy 数组
        y = cdc_data.data.targets.values.flatten()  # 确保 y 是 1D
        
        # ===== 新增：字母 -> 数字映射 =====
        label2id = {chr(i+65): i for i in range(26)}  # 'A'->0, 'B'->1, ..., 'Z'->25
        y = [label2id[label] for label in y]          # 转换所有标签
        
        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=y
        )

        # 根据 train 变量选择数据
        if self.train:
            self.X, self.y = self.X_train, self.y_train
        else:
            self.X, self.y = self.X_test, self.y_test

    def __getitem__(self, idx):
        features = self.X[idx]
        label = self.y[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.X)




import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np


class AdultDataset(Dataset):
    def __init__(self, train=True, root="./dataset/adult/", test_size=0.25, random_seed=42):
        super(AdultDataset, self).__init__()
        self.root = root
        self.train = train
        self.test_size = test_size
        self.random_seed = random_seed

        # 文件路径
        data_path = os.path.join(self.root, "adult.data")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Adult dataset not found at {data_path}")

        # adult.data 列名定义
        columns = [
            "age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
        ]

        # 加载数据
        df = pd.read_csv(data_path, header=None, names=columns, na_values=" ?", skipinitialspace=True)

        # 删除缺失值
        df.dropna(inplace=True)

        # 标签编码
        df["income"] = df["income"].apply(lambda x: 1 if ">50K" in x else 0)

        # 特征与标签分离
        X = df.drop("income", axis=1)
        y = df["income"].values

        # 区分类别与数值特征
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

        # 预处理管道
        self.preprocessor = ColumnTransformer([
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ])

        # 训练 / 测试划分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_seed, stratify=y
        )

        if train:
            self.X = self.preprocessor.fit_transform(X_train)
            self.y = y_train
        else:
            # 使用相同的编码器变换测试集
            self.X = self.preprocessor.fit(X_train).transform(X_test)
            self.y = y_test

        # 转为 torch 张量
        self.X = torch.tensor(self.X.toarray() if hasattr(self.X, "toarray") else self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]






from sklearn.preprocessing import StandardScaler

class PHISHINGDataset(Dataset):
    def __init__(self, train=True, root="./dataset", test_size=0.25, random_seed=42):
        super(PHISHINGDataset, self).__init__()
        root = '/home/shunjie/codes/DEFT/datasets/phishing/'
        data = pd.read_csv(root + 'Phishing_CM1.csv')
        # 分离特征和标签，并处理标签
        X = data.drop(['Result'], axis=1).astype(float).to_numpy()
        y = data['Result'].to_numpy()
        y = np.where(y == -1, 0, y)  # 把 -1 替换为 0
        y = y.astype(int)            # 转成整数
        y = torch.tensor(y, dtype=torch.long)  # 转成 LongTensor

        # 特征归一化
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = torch.tensor(X, dtype=torch.float32)

        # 官方划分: 8844 train + 2211 test
        n_train = 8844
        indexes_list = np.arange(len(X))
        train_indexes = indexes_list[:n_train]
        test_indexes = indexes_list[n_train:]

        train_data, test_data = X[train_indexes], X[test_indexes]
        train_target, test_target = y[train_indexes], y[test_indexes]

        if train:
            self.data = train_data
            self.targets = train_target
        else:
            self.data = test_data
            self.targets = test_target

    def __getitem__(self, idx):
        features = self.data[idx]
        label = self.targets[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)
    
class UCIHARDataset(Dataset):
    def __init__(self, train=True, root="./dataset", test_size=0.25, random_seed=42):
        super(UCIHARDataset, self).__init__()
        root = '/home/shunjie/codes/DEFT/datasets/har/UCIHARDataset/'
        if train:
            self.data = np.loadtxt(root + '/train/X_train.txt')
            self.targets = np.loadtxt(root + '/train/y_train.txt') - 1
        else:
            self.data = np.loadtxt(root + '/test/X_test.txt')
            self.targets = np.loadtxt(root + '/test/y_test.txt') - 1

        # 根据 train 参数选择数据
        self.X, self.y = self.data, self.targets

    def __getitem__(self, idx):
        features = self.X[idx]
        label = self.y[idx]

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.X)


def min_max_scaling(df):
    df_norm = df.copy()
    for column in df_norm.columns:
        df_norm[column] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())
    return df_norm

class LetterDataset(Dataset):
    def __init__(self, csv_path=None, train=True):
        """
        Args:
            csv_path (string): Path to the csv file.
        """
        csv_path = "/home/shunjie/codes/defend_label_inference/cs/Datasets/letter/letter-recognition.data"
        
        self.train = train
        self.df = pd.read_csv(csv_path)

        # self.df.columns = ['Label', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'Label']
        # self.df = self.df.drop(columns=['Name'])
        # # 假设label是最后一列
        self.df.columns = ['Label'] + [f'f{i}' for i in range(1, self.df.shape[1])]
        
        le = LabelEncoder()
        self.df['Label'] = le.fit_transform(self.df['Label'])

        y = self.df["Label"].values  # 提取标签
        x = self.df.drop(columns=["Label"])  # 去除标签列
        x = min_max_scaling(x)  # 对特征归一化

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=16)

        sc = StandardScaler()

        x_train = sc.fit_transform(x_train)
        x_test = sc.fit_transform(x_test)

        self.train_data = x_train  # numpy array
        self.test_data = x_test

        self.train_labels = y_train.tolist()
        self.test_labels = y_test.tolist()

        print(csv_path, "train", len(self.train_data), "test", len(self.test_data))

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            data, label = self.train_data[index], self.train_labels[index]
        else:
            data, label = self.test_data[index], self.test_labels[index]

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class PendigitsDataset(Dataset):
    def __init__(self, csv_path=None, train=True):
        """
        Args:
            csv_path (string): Path to the csv file (训练和测试数据合并的 CSV)
        """
        # csv_path = "/home/shunjie/codes/DEFT/basl/dataset/pen/"
        if train:
            csv_path = "/home/shunjie/codes/DEFT/basl/dataset/pen/pendigits.tra"
        else:
            csv_path = "/home/shunjie/codes/DEFT/basl/dataset/pen/pendigits.tes"
     
        self.train = train
        self.df = pd.read_csv(csv_path, header=None)

        # Pendigits 数据集前16列是特征，最后1列是标签
        self.df.columns = [f'f{i}' for i in range(16)] + ['Label']

        # 提取标签和特征
        y = self.df['Label'].values
        x = self.df.drop(columns=['Label'])
        
        # 特征归一化到 [0,1]
        x = min_max_scaling(x)

        # 划分训练/测试集
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, random_state=42, stratify=y
        )

        # 标准化
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        self.train_data = x_train
        self.test_data = x_test
        self.train_labels = y_train.tolist()
        self.test_labels = y_test.tolist()

        print(csv_path, "train", len(self.train_data), "test", len(self.test_data))

    def __len__(self):
        return len(self.train_data) if self.train else len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            data, label = self.train_data[index], self.train_labels[index]
        else:
            data, label = self.test_data[index], self.test_labels[index]

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# class PenDigitsDataset(Dataset):
#     def __init__(self, root="/home/shunjie/codes/DEFT/basl/dataset/semeion/",
#                  file_name="semeion.data",
#                  train=True,
#                  test_size=0.25,
#                  random_seed=42):
#         super(PenDigitsDataset, self).__init__()
#         self.root = root
#         self.file_path = os.path.join(root, file_name)
#         self.train = train
#         self.test_size = test_size
#         self.random_seed = random_seed

#         # 检查文件
#         if not os.path.exists(self.file_path):
#             raise FileNotFoundError(f"Semeion dataset not found at {self.file_path}")

#         # 数据读取
#         # Semeion 数据集每行 256（手写像素特征）+ 10（one-hot 标签） = 266 列
#         data = np.loadtxt(self.file_path)
#         X = data[:, :256].astype(np.float32)
#         y = data[:, 256:].astype(np.int64).argmax(axis=1)  # one-hot -> label 0-9

#         # 训练 / 测试划分
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=self.test_size, random_state=self.random_seed, stratify=y
#         )

#         if self.train:
#             self.X = X_train
#             self.y = y_train
#         else:
#             self.X = X_test
#             self.y = y_test

#         # 标准化
#         self.scaler = StandardScaler()
#         self.X = self.scaler.fit_transform(self.X) if self.train else self.scaler.transform(self.X)

#         # 转为 torch.tensor
#         self.X = torch.tensor(self.X, dtype=torch.float32)
#         self.y = torch.tensor(self.y, dtype=torch.long)

#     def __len__(self):
#         return len(self.y)

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]


datasets_choices = [
    "mnist",
    "fashionmnist",
    "fmnist",
    "cifar10",
    "cifar100",
    "criteo",
    "cinic10",
    "aids",
    "cdc",
    "imagenet12",
    "ucihar",
    "phishing",
    "adult",
    "credit",
    "letter",
    "pen"
]
datasets_name = {
    "mnist": "MNIST",
    "fashionmnist": "FashionMNIST",
    "fmnist":"FashionMNIST",
    "cifar10": "CIFAR10",
    "cifar100": "CIFAR100",
    "criteo": "Criteo",
    "cinic10": "CINIC10",
    "aids":"AIDS",
    "cdc":"CDC",
    "imagenet12":"ImageNet12",
    "ucihar":"UCIHAR",
    "phishing":"PHISHING",
    "adult":"Adult",
    "credit":"Credit",
    "letter":"Letter",
    "pen":"Pen"
}

datasets_dict = {
    "mnist": datasets.MNIST,
    "fashionmnist": datasets.FashionMNIST,
    "fmnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "criteo": Criteo,
    "cinic10": CINIC10,
    "aids": AIDSDataset,
    "cdc": CDCDataset,
    "imagenet12": ImageNet12,
    "ucihar": UCIHARDataset,
    "phishing": PHISHINGDataset,
    "adult": AdultDataset,
    "credit": CreditDataset,
    "letter": LetterDataset,
    "pen":PendigitsDataset
}

datasets_classes = {
    "mnist": 10,
    "fashionmnist": 10,
    "fmnist": 10,
    "cifar10": 10,
    "cifar100": 100,
    "criteo": 2,
    "cinic10": 10,
    "aids":2,
    "cdc":26,
    "imagenet12":12,
    "ucihar":6,
    "phishing":2,
    "adult":2,
    "credit": 2,
    "letter": 26,
    "pen":10
}

transforms_default = {
    "mnist": transforms.Compose([transforms.ToTensor()]),
    "fashionmnist": transforms.Compose([transforms.ToTensor()]),
    "fmnist": transforms.Compose([transforms.ToTensor()]),
    "cifar10": transforms.Compose([transforms.ToTensor()]),
    "cifar100": transforms.Compose([transforms.ToTensor()]),
    "criteo": None,
    "cinic10": transforms.Compose([transforms.ToTensor()]),
    "aids":None,
    "cdc":None,
    "ucihar":None,
    "phishing":None,
    "imagenet12": transforms.Compose([transforms.ToTensor()]),
    "adult":None,
    "letter":None,
    "pen":None,
}

MEAN_IMAGENET = (0.485, 0.456, 0.406)
STD_IMAGENET  = (0.229, 0.224, 0.225)  

transforms_train_augment = {
    "mnist": transforms.Compose([
        # transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ]),
    "fashionmnist": transforms.Compose([
        # transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2860], std=[0.3530])
    ]),
    "fmnist": transforms.Compose([
        # transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2860], std=[0.3530])
    ]),
    "cifar10": transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ]),
    "cifar100": transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ]),
    "criteo": None,
    "aids": None,
    "ucihar": None,
    "cdc": None,
    "phishing": None,
    "adult":None,
    "letter":None,
    "pen":None,
    "cinic10": transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404], std=[0.24205776, 0.23828046, 0.25874835])
    ]),
    "imagenet12": transforms.Compose([
                    transforms.CenterCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN_IMAGENET, STD_IMAGENET)])
}


transforms_test_augment = {
    "mnist": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ]),
    "fashionmnist": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]),
    "fmnist": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]),
    "cifar10": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ]),
    "cifar100": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ]),
    "criteo": None,
    "aids": None,
    "cdc": None,
    "ucihar": None,
    "phishing": None,
    "aids": None,
    "adult": None,
    "letter":None,
    "pen":None,
    "cinic10": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404], std=[0.24205776, 0.23828046, 0.25874835])
    ]),
    "imagenet12": transforms.Compose([
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN_IMAGENET, STD_IMAGENET)])
}