import torch
import numpy as np
import os
from .vflbase import BaseVFL
# from ..ubd import UBDDefense
import torch.nn.functional as F

import random
import helper as cc

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels):
        """
        embeddings: tensor [N, D]
        labels: tensor [N]
        """
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def select(N=128, pr=0.5):
    sequence = np.arange(N)
    num_samples = int(pr * N)

    backdoor = np.random.choice(sequence, num_samples, replace=False)
    backdoor = backdoor.tolist()
    
    clean = []
    for i in range(N):
        if i not in backdoor:
            clean.append(i)
    
    indicator = np.zeros(N)
    indicator[backdoor] = 1
    return backdoor, clean, indicator




class BASL(BaseVFL):
    def __init__(self, args, model, train_loader, test_loader, device, trigger):
        super(BASL, self).__init__(args, model, train_loader, test_loader, device)
        self.args = args
        self.rate = 0.8
        if args.dataset == "aids":
            self.auxiliary_number = 50
        else:
            self.auxiliary_number = 500
        self.auxiliary_index  = self.obtain_auxiliary_index()
        self.trigger = trigger

    def obtain_auxiliary_index(self):
        _all = []
        for batch_idx, (inputs, labels, indices) in enumerate(self.train_loader):
            for i in range(len(labels)):
                if labels[i].item() == self.args.target_label:
                    _all.append(indices[i].item())
        
        return random.sample(_all, self.auxiliary_number)
    
    @torch.no_grad()
    def design_vec(self,):
        outputs = []
        self.model.train()
    
        for batch_idx, (inputs, labels, indices) in enumerate(self.train_loader):
            # data = [temp for temp in torch.chunk(inputs, self.args.num_passive, dim=2)]

            if self.args.dataset == 'cdc':
                # CDC 数据集有 20 个特征，前 10 个和后 10 个拆分
                data = [inputs[:, :8].to(self.device), inputs[:, 8:].to(self.device)]
            elif self.args.dataset == 'aids':
                # AIDS 数据集有 23个特征，前 12 个和后 11 个拆分
                data = [inputs[:, :12].to(self.device), inputs[:, 12:].to(self.device)]
            elif self.args.dataset == 'ucihar':
                # ucihar 数据集有 561 个特征，前 281 个和后 280 个拆分
                data = [inputs[:, :281].to(self.device), inputs[:, 281:].to(self.device)]
            elif self.args.dataset == 'phishing':
                # phishing 数据集有 30 个特征，前 15 个和后 15 个拆分
                data = [inputs[:, :15].to(self.device), inputs[:, 15:].to(self.device)]
            elif self.args.dataset == 'adult':
                # phishing 数据集有 30 个特征，前 15 个和后 15 个拆分
                data = [inputs[:, :7].to(self.device), inputs[:, 7:].to(self.device)]                
            elif self.args.dataset == 'letter':
                data = [inputs[:, :8].to(self.device), inputs[:, 8:].to(self.device)]
            elif self.args.dataset == 'pen':
                data = [inputs[:, :8].to(self.device), inputs[:, 8:].to(self.device)]
            else:
                data = [temp.to(self.device) for temp in torch.chunk(inputs, self.args.num_passive, dim=2)] 
                
            for i in range(len(data)):
                data[i] = data[i].to(self.device)
        
            for i in [self.args.attack_id]:
                tmp_emb = self.model.passive[i](data[i])
                for j in range(len(labels)):
                    if indices[j] in self.auxiliary_index:
                        outputs.append(tmp_emb[j].detach().clone())

        outputs = torch.stack(outputs)
        target_clean_vecs = outputs.detach().cpu().numpy()
        dim = cc.filter_dim(target_clean_vecs)
        center = cc.cal_target_center(target_clean_vecs[dim].copy(), kernel_bandwidth=1000) 
        target_vec = cc.search_vec(center,target_clean_vecs)
        target_vec = target_vec.flatten()
        target_vec = torch.tensor(target_vec, requires_grad=True)
        target_vec = target_vec.to(self.device)
        return target_vec
    
    def train(self,):
        for epoch in range(self.args.epochs):
            if epoch >= self.args.attack_epoch:
                if self.trigger is None or epoch % 10 == 0:
                    self.trigger = self.design_vec()

            self.train_one(epoch)
            self.test()
            if epoch >= self.args.attack_epoch: self.backdoor()
            self.scheduler_entire.step()

            if epoch % 10 == 0:
                save_path = os.path.join('/home/shunjie/codes/DEFT/basl/imagenet12', f"model_dataset={self.args.dataset}_epoch={epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    #'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler_entire.state_dict(),
                    'trigger': self.trigger,
                    'args': self.args,
                }, save_path)
                print(f"Model saved at epoch {epoch} to {save_path}")

    def test(self):
        print("\n============== Test ==============")
        self.model.eval()
        self.model.active.eval()
        for i in range(self.args.num_passive):
            self.model.passive[i].eval()

        test_loss = 0
        correct = 0
        num_iter = (len(self.test_loader.dataset)//(self.args.batch_size))+1
        with torch.no_grad():
            for i, (inputs, labels, indices) in enumerate(self.test_loader):
                # data, labels, index = batch_data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                if self.args.dataset == 'cdc':
                    # CDC 数据集有 20 个特征，前 10 个和后 10 个拆分
                    data = [inputs[:, :8].to(self.device), inputs[:, 8:].to(self.device)]
                elif self.args.dataset == 'aids':
                     # AIDS 数据集有 23个特征，前 12 个和后 11 个拆分
                    data = [inputs[:, :12].to(self.device), inputs[:, 12:].to(self.device)]
                elif self.args.dataset == 'ucihar':
                    # ucihar 数据集有 561 个特征，前 281 个和后 280 个拆分
                    data = [inputs[:, :281].to(self.device), inputs[:, 281:].to(self.device)]
                elif self.args.dataset == 'phishing':
                    # phishing 数据集有 30 个特征，前 15 个和后 15 个拆分
                    data = [inputs[:, :15].to(self.device), inputs[:, 15:].to(self.device)]

                elif self.args.dataset == 'adult':
                    # phishing 数据集有 30 个特征，前 15 个和后 15 个拆分
                    data = [inputs[:, :7].to(self.device), inputs[:, 7:].to(self.device)]
                elif self.args.dataset == 'letter':
                    data = [inputs[:, :8].to(self.device), inputs[:, 8:].to(self.device)]
                elif self.args.dataset == 'pen':
                    data = [inputs[:, :8].to(self.device), inputs[:, 8:].to(self.device)]
                else:
                    # 其他情况，按照 dim=2 拆分
                    data = [temp.to(self.device) for temp in torch.chunk(inputs, self.args.num_passive, dim=2)] 
                    
                for i in range(len(data)):
                    data[i] = data[i].to(self.device)
                
                logits  = self.model(data)

                losses = F.cross_entropy(logits, labels, reduction='none')
                test_loss += torch.sum(losses).item()
                
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
        
        test_loss = test_loss / len(self.test_loader.dataset)
        
        test_acc = 100. * correct / len(self.test_loader.dataset)
        
        # print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        #     test_loss, correct, len(self.test_loader.dataset), test_acc))
        
        print('ACC: {:.2f}%, {}/{}\n'.format(test_acc, correct, len(self.test_loader.dataset)))
        
        #print('ASR: {}/{} ({:.2f}%)\n'.format(correct, len(self.test_loader.dataset), test_acc))        
        return test_acc

    def train_one(self, epoch):
        self.iteration = len(self.train_loader.dataset)
        self.model.train()

        for batch_idx, (inputs, labels, indices) in enumerate(self.train_loader):
            if self.args.dataset == 'cdc':
                # CDC 数据集有 20 个特征，前 10 个和后 10 个拆分
                data = [inputs[:, :8].to(self.device), inputs[:, 8:].to(self.device)]
            elif self.args.dataset == 'aids':
                    # AIDS 数据集有 23个特征，前 12 个和后 11 个拆分
                data = [inputs[:, :12].to(self.device), inputs[:, 12:].to(self.device)]
            elif self.args.dataset == 'ucihar':
                data = [inputs[:, :281].to(self.device), inputs[:, 281:].to(self.device)]
            elif self.args.dataset == 'phishing':
                data = [inputs[:, :15].to(self.device), inputs[:, 15:].to(self.device)]
            elif self.args.dataset == 'adult':
                data = [inputs[:, :7].to(self.device), inputs[:, 7:].to(self.device)]
            elif self.args.dataset == 'letter':
                data = [inputs[:, :8].to(self.device), inputs[:, 8:].to(self.device)]
            elif self.args.dataset == 'pen':
                data = [inputs[:, :8].to(self.device), inputs[:, 8:].to(self.device)]
            else:
                data = [temp.to(self.device) for temp in torch.chunk(inputs, self.args.num_passive, dim=2)]
            labels = labels.to(self.device)
            
            emb = []
            for i in range(self.args.num_passive):
                # print("Here", data[i].shape)
                tmp_emb = self.model.passive[i](data[i])
                emb.append(tmp_emb)

            if epoch >= self.args.attack_epoch:
                condition = []
                for attacker in [self.args.attack_id]:
                    for i in range(len(labels)):
                        if indices[i].item() in self.auxiliary_index:
                            condition.append(i)
                    emb[attacker][condition].data = self.trigger

            # forward propagation
            agg_emb = self.model._aggregate(emb)
            logit = self.model.active(None, None, agged_inputs=agg_emb, agged=True)
            loss = self.loss(logit, labels)                  
            
            self.optimizer_entire.zero_grad()
            loss.backward()
            self.optimizer_entire.step()

            # if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == self.iteration:
            print('Epoch:{}/{}, Step:{} \tLoss: {:.6f}'.format(epoch+1, self.args.epochs, batch_idx+1, loss.item()))
        
        return 

    def backdoor(self):
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (inputs, labels, indices) in enumerate(self.test_loader):
                
                if self.args.dataset == 'cdc':
                    # CDC 数据集有 20 个特征，前 10 个和后 10 个拆分
                    data = [inputs[:, :8].to(self.device), inputs[:, 8:].to(self.device)]
                elif self.args.dataset == 'aids':
                     # AIDS 数据集有 23个特征，前 12 个和后 11 个拆分
                    data = [inputs[:, :12].to(self.device), inputs[:, 12:].to(self.device)]
                elif self.args.dataset == 'ucihar':
                    # ucihar 数据集有 561 个特征，前 281 个和后 280 个拆分
                    data = [inputs[:, :281].to(self.device), inputs[:, 281:].to(self.device)]
                elif self.args.dataset == 'phishing':
                    # phishing 数据集有 30 个特征，前 15 个和后 15 个拆分
                    data = [inputs[:, :15].to(self.device), inputs[:, 15:].to(self.device)]

                elif self.args.dataset == 'adult':
                    # phishing 数据集有 30 个特征，前 15 个和后 15 个拆分
                    data = [inputs[:, :7].to(self.device), inputs[:, 7:].to(self.device)]
                elif self.args.dataset == 'letter':
                    data = [inputs[:, :8].to(self.device), inputs[:, 8:].to(self.device)]
                elif self.args.dataset == 'pen':
                    data = [inputs[:, :8].to(self.device), inputs[:, 8:].to(self.device)]
                else:
                    # 其他情况，按照 dim=2 拆分
                    data = [temp.to(self.device) for temp in torch.chunk(inputs, self.args.num_passive, dim=2)]            
                    
                targets = torch.tensor([self.args.target_label for i in range(len(indices))])
                targets = targets.to(self.device)

                emb = []
                for i in range(self.args.num_passive):
                    if i == self.args.attack_id:
                        triggers = self.trigger.repeat(len(inputs), 1)
                        triggers = triggers.to(self.device)
                        emb.append(triggers)
                    else:
                        tmp_emb = self.model.passive[i](data[i])
                        emb.append(tmp_emb)
                
                agg_emb = torch.cat(emb, dim=1)

                logit = self.model.active(None, None, agged_inputs=agg_emb, agged=True)
                pred = logit.argmax(dim=1, keepdim=True)
                correct += pred.eq(targets.view_as(pred)).sum().item()
        
        test_acc = 100. * correct / len(self.test_loader.dataset)
        print('ASR: {:.2f}%, {}/{}\n'.format(test_acc, correct, len(self.test_loader.dataset)))        
        
        return test_acc



    # def observe(self,):
    #     self.iteration = len(self.train_loader.dataset)
    #     self.model.train()

    #     for epoch in range(10):
    #         for batch_idx, (inputs, labels, indices) in enumerate(self.train_loader):
    #             if self.args.dataset == 'cdc':
    #                 # CDC 数据集有 20 个特征，前 10 个和后 10 个拆分
    #                 data = [inputs[:, :8].to(self.device), inputs[:, 8:].to(self.device)]
    #             elif self.args.dataset == 'aids':
    #                     # AIDS 数据集有 23个特征，前 12 个和后 11 个拆分
    #                 data = [inputs[:, :12].to(self.device), inputs[:, 12:].to(self.device)]
    #             elif self.args.dataset == 'ucihar':
    #                 # ucihar 数据集有 561 个特征，前 281 个和后 280 个拆分
    #                 data = [inputs[:, :281].to(self.device), inputs[:, 281:].to(self.device)]
    #             elif self.args.dataset == 'phishing':
    #                 # phishing 数据集有 30 个特征，前 15 个和后 15 个拆分
    #                 data = [inputs[:, :15].to(self.device), inputs[:, 15:].to(self.device)]
    #             elif self.args.dataset == 'adult':
    #                 # phishing 数据集有 30 个特征，前 15 个和后 15 个拆分
    #                 data = [inputs[:, :7].to(self.device), inputs[:, 7:].to(self.device)]
    #             elif self.args.dataset == 'letter':
    #                 data = [inputs[:, :8].to(self.device), inputs[:, 8:].to(self.device)]
    #             elif self.args.dataset == 'pen':
    #                 data = [inputs[:, :8].to(self.device), inputs[:, 8:].to(self.device)]
    #             else:
    #                 data = [temp.to(self.device) for temp in torch.chunk(inputs, self.args.num_passive, dim=2)]
    #             labels = labels.to(self.device)
                
    #             emb = []
    #             for i in range(self.args.num_passive):
    #                 tmp_emb = self.model.passive[i](data[i])
    #                 emb.append(tmp_emb)

    #             # forward propagation
    #             agg_emb = self.model._aggregate(emb)
    #             logit = self.model.active(None, None, agged_inputs=agg_emb, agged=True)
    #             loss = self.loss(logit, labels)                  
                
    #             self.optimizer_entire.zero_grad()
    #             loss.backward()
    #             self.optimizer_entire.step()

    #         acc_o = self.test()
    #         asr_o = self.backdoor()
    #         print(f"[Epoch={epoch}] ACC:{acc_o:.4f}, ASR:{asr_o:.4f}")
    #     return 



    # def obtain_active_embeddings(self):
    #     self.model.eval()
    #     active_idx = 1
    #     class_embeddings = {cls: [] for cls in range(self.args.num_classes)}

    #     for inputs, labels, _ in self.train_loader:
    #         labels = labels.to(self.device)

    #         # 数据划分
    #         if self.args.dataset == "cdc":
    #             data = [inputs[:, :8].to(self.device), inputs[:, 8:].to(self.device)]
    #         elif self.args.dataset == "aids":
    #             data = [inputs[:, :12].to(self.device), inputs[:, 12:].to(self.device)]
    #         elif self.args.dataset == 'ucihar':
    #             # ucihar 数据集有 561 个特征，前 281 个和后 280 个拆分
    #             data = [inputs[:, :281].to(self.device), inputs[:, 281:].to(self.device)]
    #         elif self.args.dataset == 'phishing':
    #             # phishing 数据集有 30 个特征，前 15 个和后 15 个拆分
    #             data = [inputs[:, :15].to(self.device), inputs[:, 15:].to(self.device)]
    #         else:
    #             data = [x.to(self.device) for x in torch.chunk(inputs, self.args.num_passive, dim=2)]

    #         # 批量计算主动嵌入
    #         active_emb = self.model.passive[active_idx](data[active_idx])  # [B, D]

    #         # 批量收集每个类别的嵌入到 list
    #         for cls in torch.unique(labels):
    #             cls_mask = labels == cls
    #             if cls_mask.any():
    #                 emb_list = [emb for emb in active_emb[cls_mask].detach().cpu()]
    #                 class_embeddings[cls.item()].extend(emb_list)

    #     return class_embeddings


    # def split_embeddings(self, class_embeddings, threshold=0.1):
    #     """
    #     对每个样本进行后门检测：使用其他类别主动嵌入拼接被动嵌入，计算平均logits并求熵。
        
    #     参数:
    #         class_embeddings: dict, key=类别, value=list of torch.Tensor (主动嵌入)
    #         threshold: float, 熵阈值
    #     """
    #     self.model.eval()
    #     active_idx = 1
    #     attacker = self.args.attack_id

    #     all_embeddings = [[] for _ in range(self.args.num_passive)]
    #     all_labels, all_entropy, all_indices = [], [], []

    #     for inputs, labels, indices in self.train_loader:
    #         labels = labels.to(self.device)
    #         indices = indices.to(self.device)

    #         # 数据划分
    #         if self.args.dataset == "cdc":
    #             data = [inputs[:, :8].to(self.device), inputs[:, 8:].to(self.device)]
    #         elif self.args.dataset == "aids":
    #             data = [inputs[:, :12].to(self.device), inputs[:, 12:].to(self.device)]
    #         else:
    #             data = [x.to(self.device) for x in torch.chunk(inputs, self.args.num_passive, dim=2)]

    #         # 被动方嵌入
    #         embeddings = [self.model.passive[i](data[i]) for i in range(self.args.num_passive)]

    #         # 批量植入触发器（仅攻击者）
    #         aux_mask = torch.tensor([idx.item() in self.auxiliary_index for idx in indices], device=self.device)
    #         if aux_mask.any():
    #             embeddings[attacker][aux_mask] = self.trigger

    #         batch_size = labels.size(0)
    #         batch_entropy = torch.zeros(batch_size, device=self.device)

    #         # 对每个样本处理
    #         for i in range(batch_size):
    #             cls = labels[i].item()
    #             agg_emb_list = []

    #             # 对每个其他类别随机选择一个主动嵌入，拼接被动嵌入
    #             for other_cls in range(self.args.num_classes):
    #                 if other_cls == cls: continue
    #                 # 从 class_embeddings 中随机选一个
    #                 other_emb_list = class_embeddings[other_cls]
    #                 idx_choice = np.random.randint(len(other_emb_list))
    #                 active_emb = other_emb_list[idx_choice].to(self.device).unsqueeze(0)  # [1, D]
                    
    #                 # 拼接当前样本的被动嵌入
    #                 passive_embs = [embeddings[j][i:i+1] for j in range(self.args.num_passive)]
    #                 passive_embs[active_idx] = active_emb  # 替换主动嵌入
    #                 agg_emb = torch.cat(passive_embs, dim=1)  # [1, num_passive * D]
    #                 agg_emb_list.append(agg_emb)

    #             agg_emb_batch = torch.cat(agg_emb_list, dim=0)  # [C-1, num_passive*D]

    #             with torch.no_grad():
    #                 logits = self.model.active(None, None, agged_inputs=agg_emb_batch, agged=True)
    #                 avg_logits = logits.mean(dim=0, keepdim=True)
    #                 probs = torch.softmax(avg_logits, dim=1)
    #                 entropy = -(probs * torch.log(torch.clamp(probs, min=1e-10))).sum(dim=1)
    #                 batch_entropy[i] = entropy

    #         # 保存嵌入和标签
    #         for j in range(self.args.num_passive):
    #             all_embeddings[j].append(embeddings[j].cpu())
    #         all_labels.append(labels.cpu())
    #         all_entropy.append(batch_entropy.cpu())
    #         all_indices.append(indices.cpu())

    #     # 拼接
    #     all_embeddings = [torch.cat(lst, dim=0) for lst in all_embeddings]
    #     all_labels = torch.cat(all_labels, dim=0)
    #     all_entropy = torch.cat(all_entropy, dim=0)
    #     all_indices = torch.cat(all_indices, dim=0)

    #     # 根据阈值划分
    #     malicious_mask = all_entropy < threshold
    #     y_true = torch.tensor([1 if idx.item() in self.auxiliary_index else 0 for idx in all_indices])
    #     y_pred = malicious_mask.int()

    #     # 检测指标
    #     precision = precision_score(y_true, y_pred, zero_division=0)
    #     recall = recall_score(y_true, y_pred, zero_division=0)
    #     f1 = f1_score(y_true, y_pred, zero_division=0)
    #     print(f"[Detection Metrics] Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    #     embeddings_benign    = [emb[~malicious_mask].detach() for emb in all_embeddings]
    #     embeddings_malicious = [emb[malicious_mask].detach() for emb in all_embeddings]
    #     labels_benign = all_labels[~malicious_mask]
    #     labels_malicious = all_labels[malicious_mask]

    #     # 绘图
    #     all_entropy_np = all_entropy.numpy()
    #     y_true_np = y_true.numpy()
    #     y_pred_np = y_pred.numpy()
    #     fig, axes = plt.subplots(1, 2, figsize=(14,5))
    #     axes[0].scatter(np.where(y_true_np==0)[0], all_entropy_np[y_true_np==0], color='green', s=2, label='Benign')
    #     axes[0].scatter(np.where(y_true_np==1)[0], all_entropy_np[y_true_np==1], color='red', s=2, label='Backdoor')
    #     axes[0].set_title("Ground Truth"); axes[0].set_xlabel("Sample Index"); axes[0].set_ylabel("Entropy"); axes[0].legend()
    #     axes[1].scatter(np.where(y_pred_np==0)[0], all_entropy_np[y_pred_np==0], color='green', s=2, label='Benign')
    #     axes[1].scatter(np.where(y_pred_np==1)[0], all_entropy_np[y_pred_np==1], color='red', s=2, label='Backdoor')
    #     axes[1].set_title("Prediction"); axes[1].set_xlabel("Sample Index"); axes[1].set_ylabel("Entropy"); axes[1].legend()
    #     plt.tight_layout()
    #     plt.savefig(f'entropy_distribution_dataset={self.args.dataset}.pdf')

    #     return embeddings_benign, embeddings_malicious, labels_benign, labels_malicious


    # def finetune_top_model(self, embeddings_benign, embeddings_malicious, labels_benign, labels_malicious, lr=0.001, fine_tune_epochs=10):
    #     """
    #     微调 top 模型
    #     - embeddings_benign: list，每个元素是 [N_b, D] 的 tensor，对应每个 passive party
    #     - embeddings_malicious: list，每个元素是 [N_m, D] 的 tensor，对应每个 passive party
    #     - labels_benign, labels_malicious: tensor
    #     """
    #     attacker = self.args.attack_id
    #     num_passive = self.args.num_passive

    #     # ========== 构造良性数据 ==========
    #     benign_emb_all = torch.cat(embeddings_benign, dim=1)  # [N_b, D_total]
    #     benign_labels_all = labels_benign

    #     # ========== 构造 hybrid 数据 ==========
    #     hybrid_parts_per_passive = [[] for _ in range(num_passive)]
    #     hybrid_label_list = []

    #     labels_b_np = labels_benign.cpu().numpy()
    #     N_mal = labels_malicious.size(0)

    #     for m_idx in range(N_mal):
    #         mal_label = int(labels_malicious[m_idx].item())

    #         # 找标签不相同的良性样本
    #         pool = np.where(labels_b_np != mal_label)[0]
    #         if pool.size == 0:
    #             continue
    #         b_idx = int(np.random.choice(pool))

    #         # 对每个 passive party 构造 hybrid embedding
    #         for j in range(num_passive):
    #             if j == attacker:
    #                 # 恶意嵌入
    #                 part = embeddings_malicious[j][m_idx].unsqueeze(0).detach()
    #             else:
    #                 # 良性嵌入
    #                 part = embeddings_benign[j][b_idx].unsqueeze(0).detach()
    #             hybrid_parts_per_passive[j].append(part)

    #         # hybrid label 用良性标签
    #         hybrid_label_list.append(labels_benign[b_idx].unsqueeze(0).detach())

    #     # 拼接每个 passive party 的 hybrid embedding
    #     hybrid_tensors = [torch.cat(parts, dim=0) if len(parts) > 0 else torch.empty(0, embeddings_benign[0].size(1)) for parts in hybrid_parts_per_passive]
    #     hybrid_emb_all = torch.cat(hybrid_tensors, dim=1) if len(hybrid_tensors[0]) > 0 else torch.empty(0, benign_emb_all.size(1))
    #     hybrid_labels_all = torch.cat(hybrid_label_list, dim=0) if len(hybrid_label_list) > 0 else torch.empty(0, dtype=torch.long)

    #     # ========== 合并良性和 hybrid ==========
    #     final_embeddings = torch.cat([benign_emb_all, hybrid_emb_all], dim=0) if hybrid_emb_all.size(0) > 0 else benign_emb_all
    #     final_labels     = torch.cat([benign_labels_all, hybrid_labels_all], dim=0) if hybrid_labels_all.size(0) > 0 else benign_labels_all

    #     # ========== 微调 top 模型 ==========
    #     dataset = EmbeddingDataset(final_embeddings.to(self.device), final_labels.to(self.device))
    #     loader  = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    #     criterion = torch.nn.CrossEntropyLoss()
    #     optimizer = torch.optim.Adam(self.model.active.parameters(), lr=lr)

    #     self.model.active.train()
    #     for epoch in range(fine_tune_epochs):
    #         total_loss, total_correct, total_num = 0.0, 0, 0
    #         for batch_idx, (emb_batch, label_batch) in enumerate(loader):
    #             optimizer.zero_grad()
    #             logits = self.model.active(None, None, agged_inputs=emb_batch, agged=True)
    #             loss = criterion(logits, label_batch)
    #             loss.backward()
    #             optimizer.step()

    #             total_loss += loss.item() * emb_batch.size(0)
    #             total_correct += (logits.argmax(dim=1) == label_batch).sum().item()
    #             total_num += emb_batch.size(0)

    #         avg_loss = total_loss / total_num
    #         acc = total_correct / total_num
    #         print(f"Epoch [{epoch+1}/{fine_tune_epochs}] Loss={avg_loss:.6f}, Acc={acc:.4f}")



    # def evaluate_deft_defense(self, lr=0.001, fine_tune_epochs=30):
    #     class_embeddings = self.obtain_active_embeddings()
    #     embeddings_benign, embeddings_malicious, labels_benign, labels_malicious = self.split_embeddings(class_embeddings)
    #     self.finetune_top_model(embeddings_benign, embeddings_malicious, labels_benign, labels_malicious, lr=lr, fine_tune_epochs=fine_tune_epochs)
    #     return















