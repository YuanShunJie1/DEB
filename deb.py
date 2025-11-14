import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import copy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels, weights):
        self.embeddings = embeddings
        self.labels = labels
        self.weights = weights
        # self.indicators = indicators

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.weights[idx]

def safe_cat(tensor_list):
    """把一个tensor list安全拼成 batch tensor，即使只有一个元素"""
    if len(tensor_list) == 1:
        return tensor_list[0].unsqueeze(0)
    else:
        return torch.cat(tensor_list, dim=0)

def draw_entropy_distribution(args, all_entropy_np, y_true_np, y_pred_np):
    fig_file = f'/home/shunjie/codes/DEFT/basl/results_images/entropy_distribution_dataset={args.dataset}_tau={args.tau}.pdf'
    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    axes[0].scatter(np.where(y_true_np==0)[0], all_entropy_np[y_true_np==0], color='green', s=2, label='Benign')
    axes[0].scatter(np.where(y_true_np==1)[0], all_entropy_np[y_true_np==1], color='red', s=2, label='Backdoor')
    axes[0].set_title("Ground Truth"); axes[0].set_xlabel("Sample Index"); axes[0].set_ylabel("Entropy"); axes[0].legend()
    axes[1].scatter(np.where(y_pred_np==0)[0], all_entropy_np[y_pred_np==0], color='green', s=2, label='Benign')
    axes[1].scatter(np.where(y_pred_np==1)[0], all_entropy_np[y_pred_np==1], color='red', s=2, label='Backdoor')
    axes[1].set_title("Prediction"); axes[1].set_xlabel("Sample Index"); axes[1].set_ylabel("Entropy"); axes[1].legend()
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(fig_file)

# , fontsize=26
def draw_embedding_distribution(args, all_embeddings, y_pred_np, y_true_np, fig_file=None):
    # ====== t-SNE降维 ======
    combined_embs = torch.cat(all_embeddings, dim=1)
    tsne = TSNE(n_components=2, perplexity=100, max_iter=1000, random_state=42)
    emb_2d = tsne.fit_transform(combined_embs.numpy())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # # 左图：聚类结果
    axes[0].scatter(emb_2d[y_pred_np==0, 0], emb_2d[y_pred_np==0, 1], s=10, c='#5f9e6e', label='Benign', rasterized=True)
    axes[0].scatter(emb_2d[y_pred_np==1, 0], emb_2d[y_pred_np==1, 1], s=10, c='#cc8963', label='Backdoor', rasterized=True)
    axes[0].set_title("Detection", fontsize=20)
    axes[0].tick_params(axis='both', labelsize=16)
    axes[0].legend(fontsize=24,loc='lower right')

    # 右图：真实后门 vs 良性
    axes[1].scatter(emb_2d[y_true_np==0, 0], emb_2d[y_true_np==0, 1], s=10, c='#5f9e6e', label='Benign', rasterized=True)
    axes[1].scatter(emb_2d[y_true_np==1, 0], emb_2d[y_true_np==1, 1], s=10, c='#cc8963', label='Backdoor', rasterized=True)
    axes[1].set_title("Ground Truth", fontsize=20)
    axes[1].tick_params(axis='both', labelsize=16)
    axes[1].legend(fontsize=24,loc='lower right')

    plt.tight_layout()
    plt.savefig(fig_file, dpi=200)
    plt.close()
    
    
def draw_embedding_distribution_super(args, all_embeddings, y_pred_ent, y_pred_dis, y_true_np, fig_file=None):
    # ====== t-SNE降维 ======
    combined_embs = torch.cat(all_embeddings, dim=1)
    tsne = TSNE(n_components=2, perplexity=100, max_iter=1000, random_state=42)
    emb_2d = tsne.fit_transform(combined_embs.numpy())

    # ====== 创建3行1列布局 ======
    fig, axes = plt.subplots(3, 1, figsize=(7, 15))

    # ====== 第一行：Entropy-based Detection ======
    axes[0].scatter(emb_2d[y_pred_ent == 0, 0], emb_2d[y_pred_ent == 0, 1], s=10, c='#5f9e6e', label='Benign', rasterized=True)
    axes[0].scatter(emb_2d[y_pred_ent == 1, 0], emb_2d[y_pred_ent == 1, 1], s=10, c='#cc8963', label='Backdoor', rasterized=True)
    axes[0].set_title("Entropy-based Detection", fontsize=26)
    axes[0].tick_params(axis='both', labelsize=24)
    axes[0].legend(fontsize=26, loc='lower right')

    # ====== 第二行：Distance-based Detection ======
    axes[1].scatter(emb_2d[y_pred_dis == 0, 0], emb_2d[y_pred_dis == 0, 1], s=10, c='#5f9e6e', label='Benign', rasterized=True)
    axes[1].scatter(emb_2d[y_pred_dis == 1, 0], emb_2d[y_pred_dis == 1, 1], s=10, c='#cc8963', label='Backdoor', rasterized=True)
    axes[1].set_title("Distance-based Detection", fontsize=26)
    axes[1].tick_params(axis='both', labelsize=24)
    axes[1].legend(fontsize=26, loc='lower right')

    # ====== 第三行：Ground Truth ======
    axes[2].scatter(emb_2d[y_true_np == 0, 0], emb_2d[y_true_np == 0, 1], s=10, c='#5f9e6e', label='Benign', rasterized=True)
    axes[2].scatter(emb_2d[y_true_np == 1, 0], emb_2d[y_true_np == 1, 1], s=10, c='#cc8963', label='Backdoor', rasterized=True)
    axes[2].set_title("Ground Truth", fontsize=26)
    axes[2].tick_params(axis='both', labelsize=24)
    axes[2].legend(fontsize=26, loc='lower right')

    plt.tight_layout()
    if fig_file:
        plt.savefig(fig_file, dpi=200)
    plt.close()
    
    

class DEB(object):
    def __init__(self, args, model, train_loader, test_loader, device, trigger, auxiliary_index, threshold):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.trigger = trigger
        self.auxiliary_index  = auxiliary_index
        self.threshold = threshold
        self.lambda_w = args.lambda_w

    def obtain_active_embeddings(self):
        self.model.eval()
        active_idx = 1
        class_embeddings = {cls: [] for cls in range(self.args.num_classes)}

        for inputs, labels, _ in self.train_loader:
            labels = labels.to(self.device)

            # 数据划分
            if self.args.dataset == "cdc":
                data = [inputs[:, :8].to(self.device), inputs[:, 8:].to(self.device)]
            elif self.args.dataset == "aids":
                data = [inputs[:, :12].to(self.device), inputs[:, 12:].to(self.device)]
            elif self.args.dataset == 'ucihar':
                data = [inputs[:, :281].to(self.device), inputs[:, 281:].to(self.device)]
            elif self.args.dataset == 'phishing':
                data = [inputs[:, :15].to(self.device), inputs[:, 15:].to(self.device)]
            elif self.args.dataset == 'letter':
                data = [inputs[:, :8].to(self.device), inputs[:, 8:].to(self.device)]
            else:
                data = [x.to(self.device) for x in torch.chunk(inputs, self.args.num_passive, dim=2)]

            # 批量计算主动嵌入
            active_emb = self.model.passive[active_idx](data[active_idx])  # [B, D]

            # 批量收集每个类别的嵌入到 list
            for cls in torch.unique(labels):
                cls_mask = labels == cls
                if cls_mask.any():
                    emb_list = [emb for emb in active_emb[cls_mask].detach().cpu()]
                    class_embeddings[cls.item()].extend(emb_list)

        return class_embeddings

    def entropy_based_detection(self, class_embeddings):
        """
        对每个样本进行后门检测：使用其他类别主动嵌入拼接被动嵌入，计算平均logits并求熵。
        
        参数:
            class_embeddings: dict, key=类别, value=list of torch.Tensor (主动嵌入)
        """
        self.model.eval()
        active_idx = 1
        attacker = self.args.attack_id

        all_embeddings = [[] for _ in range(self.args.num_passive)]
        all_labels, all_entropy, all_indices = [], [], []


        # _temp_indice2idx = {}
        
        for inputs, labels, indices in self.train_loader:
            labels = labels.to(self.device)
            indices = indices.to(self.device)

            # 数据划分
            if self.args.dataset == "cdc":
                data = [inputs[:, :8].to(self.device), inputs[:, 8:].to(self.device)]
            elif self.args.dataset == "aids":
                data = [inputs[:, :12].to(self.device), inputs[:, 12:].to(self.device)]
            elif self.args.dataset == 'letter':
                data = [inputs[:, :8].to(self.device), inputs[:, 8:].to(self.device)]
            else:
                data = [x.to(self.device) for x in torch.chunk(inputs, self.args.num_passive, dim=2)]

            # 被动方嵌入 —— 每个 emb 立刻搬到 CPU
            embeddings = []
            with torch.no_grad():
                for i in range(self.args.num_passive):
                    emb = self.model.passive[i](data[i])
                    embeddings.append(emb.detach().cpu())

            # 批量植入触发器（仅攻击者）
            aux_mask = torch.tensor([idx.item() in self.auxiliary_index for idx in indices])
            if aux_mask.any():
                embeddings[attacker][aux_mask] = self.trigger.cpu()

            batch_size = labels.size(0)
            batch_entropy = torch.zeros(batch_size)

            # 对每个样本处理
            for i in range(batch_size):
                cls = labels[i].item()
                agg_emb_list = []

                # 对每个其他类别随机选择一个主动嵌入，拼接被动嵌入
                for other_cls in range(self.args.num_classes):
                    if other_cls == cls:
                        continue
                    other_emb_list = class_embeddings[other_cls]
                    idx_choice = np.random.randint(len(other_emb_list))
                    active_emb = other_emb_list[idx_choice].unsqueeze(0).to(self.device)

                    # 拼接当前样本的被动嵌入（临时搬到 GPU）
                    passive_embs = [embeddings[j][i:i+1].to(self.device) for j in range(self.args.num_passive)]
                    passive_embs[active_idx] = active_emb
                    agg_emb = torch.cat(passive_embs, dim=1)
                    agg_emb_list.append(agg_emb)

                agg_emb_batch = torch.cat(agg_emb_list, dim=0)

                # 计算 logits 和 entropy
                with torch.no_grad():
                    logits = self.model.active(None, None, agged_inputs=agg_emb_batch, agged=True)
                    avg_logits = logits.mean(dim=0, keepdim=True)
                    probs = torch.softmax(avg_logits, dim=1)
                    entropy = -(probs * torch.log(torch.clamp(probs, min=1e-10))).sum(dim=1)
                    batch_entropy[i] = entropy.cpu()

            # 保存嵌入和标签（都在 CPU）
            for j in range(self.args.num_passive):
                all_embeddings[j].append(embeddings[j])
                
            all_labels.append(labels.cpu())
            all_entropy.append(batch_entropy)
            all_indices.append(indices.cpu())

            # 主动释放 GPU 显存
            del data, embeddings, passive_embs, agg_emb_batch, logits, avg_logits, probs, entropy
            torch.cuda.empty_cache()

        # 拼接
        all_embeddings = [torch.cat(lst, dim=0) for lst in all_embeddings]
        all_labels = torch.cat(all_labels, dim=0)

        all_entropy = torch.cat(all_entropy, dim=0)
        all_indices = torch.cat(all_indices, dim=0)
        
        # malicious_mask = all_entropy < self.threshold
         # ======= 这里改为按熵排序并取最小的 k 个作为后门样本 =======
        N = all_entropy.numel()
        thr = float(self.threshold)
        k = int(np.ceil(thr * N))
        # 获取最小 k 个熵值的索引
        # torch.topk 返回最大的 k 个，使用 largest=False 返回最小的 k 个
        k = max(1, min(N, k))
        smallest_vals, smallest_idx = torch.topk(all_entropy, k=k, largest=False)
        # 构造 malicious_mask：被选中的最小熵索引视为 malicious（后门）
        malicious_mask = torch.zeros(N, dtype=torch.bool)
        malicious_mask[smallest_idx] = True
        # y_pred 对应所有样本（1 表示检测为后门，0 表示检测为良性）
        y_pred = malicious_mask.int()
        y_true = torch.tensor([1 if idx.item() in self.auxiliary_index else 0 for idx in all_indices])

        # 检测指标
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"[Entropy-Based Backdoor Detection] Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

        metrics = [precision, recall, f1]
        
        embeddings_benign    = [emb[~malicious_mask].detach() for emb in all_embeddings]
        embeddings_malicious = [emb[malicious_mask].detach() for emb in all_embeddings]
        
        indices_benign    = all_indices[~malicious_mask]
        indices_malicious = all_indices[malicious_mask]
        
        labels_benign = all_labels[~malicious_mask]
        labels_malicious = all_labels[malicious_mask]

        all_entropy_np, y_true_np, y_pred_np = all_entropy.numpy(), y_true.numpy(), y_pred.numpy()
        # draw_entropy_distribution(self.args, all_entropy_np, y_true_np, y_pred_np)
        # draw_embedding_distribution(self.args, all_embeddings, y_pred_np, y_true_np, fig_file=f'/home/shunjie/codes/DEFT/basl/results_images/embedding_distribution_entropy_dataset={self.args.dataset}_tau={self.args.tau}.pdf')

        return embeddings_benign, embeddings_malicious, labels_benign, labels_malicious, indices_benign, indices_malicious, metrics, [all_embeddings, y_pred_np, y_true_np]


    def distance_based_detection(self, embeddings_benign, embeddings_malicious, labels_benign, labels_malicious, indices_benign, indices_malicious):

        attack_id = self.args.attack_id
        
        # 返回检测得到的良性与后门嵌入
        num_parties = len(embeddings_benign)
        
        # 首先对 benign embeddings 按照类别labels_benign进行分组
        benign_class_embeddings = {i: [] for i in range(self.args.num_classes)}
        for emb, label in zip(embeddings_benign[attack_id], labels_benign):
            benign_class_embeddings[label.item()].append(emb)

        # 计算每个类别的中心点
        class_centroids = {}
        for cls, embeddings in benign_class_embeddings.items():
            class_centroids[cls] = torch.mean(torch.stack(embeddings), dim=0).cpu().numpy()

        # 计算embeddings_malicious中每个样本到对应类别中心点的距离
        malicious_distances = []
        for emb, label in zip(embeddings_malicious[attack_id], labels_malicious):
            centroid = class_centroids[label.item()]
            distance = torch.norm(emb.cpu() - torch.tensor(centroid)).item()
            malicious_distances.append(distance)

        # 使用GMM进行聚类，进一步在embeddings_malicious的基础上区分benign和malicious
        gmm = GaussianMixture(n_components=2, random_state=42)
        malicious_distances = np.array(malicious_distances).reshape(-1, 1)  # 需要将距离转为列向量
        gmm.fit(malicious_distances)
        gmm_labels = gmm.predict(malicious_distances)

        # 计算每个簇的均值和样本到簇心的距离
        cluster_means = gmm.means_.flatten()  # 获取簇的均值
        # 基于距离确定后门簇，后门簇离均值较远
        malicious_cluster = np.argmax(cluster_means)  # 获取离均值较远的簇
        malicious_mask = (gmm_labels == malicious_cluster)
        y_pred_malicious = torch.tensor(malicious_mask.astype(int), dtype=torch.int)
        # 根据计算得到的 mask 来划分恶意和良性样本
        benign_mask = ~malicious_mask

        # 构造y_pred，良性样本的标签都为0
        y_pred_benign = torch.zeros(len(labels_benign), dtype=torch.int)

        # 合并y_pred_benign和y_pred_malicious
        y_pred = torch.cat([y_pred_benign, y_pred_malicious])

        # 根据 indices_benign 和 indices_malicious 构造 y_true
        all_indices = torch.cat([indices_benign, indices_malicious])
        y_true = torch.tensor([1 if idx.item() in self.auxiliary_index else 0 for idx in all_indices])

        # 计算检测指标
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"[Distance-Based Backdoor Detection] Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

        metrics = [precision, recall, f1]
        
        _all_embeddings = []
        for i in range(num_parties):
            _all_embeddings.append(torch.cat([embeddings_benign[i], embeddings_malicious[i]], dim=0))
            
        # draw_embedding_distribution(self.args, _all_embeddings, y_pred.numpy(), y_true.numpy(), fig_file=f'/home/shunjie/codes/DEFT/basl/results_images/embedding_distribution_distance_dataset={self.args.dataset}_tau={self.args.tau}.pdf')

        embeddings_benign_detected = []
        for i in range(num_parties):
            embeddings_benign_detected.append(torch.cat([embeddings_benign[i], embeddings_malicious[i][benign_mask]], dim=0))
        
        labels_benign_detected = torch.cat([labels_benign, labels_malicious[benign_mask]], dim=0)
        
        # 返回检测出的恶意样本
        embeddings_malicious_detected = []
        for i in range(num_parties):
            embeddings_malicious_detected.append(embeddings_malicious[i][malicious_mask])
        labels_malicious_detected = labels_malicious[malicious_mask]
        
        all_embeddings = []
        for i in range(num_parties):
            all_embeddings.append(torch.cat([embeddings_benign_detected[i], embeddings_malicious_detected[i]], dim=0))
        
        y_pred = torch.cat([y_pred_benign, y_pred_malicious])
        all_indices = torch.cat([indices_benign, indices_malicious])
        sorted_indices, sort_order = torch.sort(all_indices)
        out_pred = y_pred[sort_order]
        out_pred = out_pred.numpy()
        
        return embeddings_benign_detected, embeddings_malicious_detected, labels_benign_detected, labels_malicious_detected, metrics, out_pred


    def finetune_top_model(self, embeddings_benign, embeddings_malicious, labels_benign, labels_malicious, lr=0.001, fine_tune_epochs=10):
        """
        微调 top 模型
        - embeddings_benign: list，每个元素是 [N_b, D] 的 tensor，对应每个 passive party
        - embeddings_malicious: list，每个元素是 [N_m, D] 的 tensor，对应每个 passive party
        - labels_benign, labels_malicious: tensor
        """
        attacker = self.args.attack_id
        num_passive = self.args.num_passive

        # ========== 构造良性数据 ==========
        benign_emb_all = torch.cat(embeddings_benign, dim=1)  # [N_b, D_total]
        benign_labels_all = labels_benign

        # ========== 构造 hybrid 数据 ==========
        hybrid_parts_per_passive = [[] for _ in range(num_passive)]
        hybrid_label_list = []

        labels_b_np = labels_benign.cpu().numpy()
        N_mal = labels_malicious.size(0)

        for m_idx in range(N_mal):
            mal_label = int(labels_malicious[m_idx].item())

            # 找标签不相同的良性样本
            pool = np.where(labels_b_np != mal_label)[0]
            if pool.size == 0:
                continue
            b_idx = int(np.random.choice(pool))

            # 对每个 passive party 构造 hybrid embedding
            for j in range(num_passive):
                if j == attacker:
                    part = embeddings_malicious[j][m_idx].unsqueeze(0).detach()
                else:
                    part = embeddings_benign[j][b_idx].unsqueeze(0).detach()
                hybrid_parts_per_passive[j].append(part)

            # hybrid label 用良性标签
            hybrid_label_list.append(labels_benign[b_idx].unsqueeze(0).detach())

        # 拼接每个 passive party 的 hybrid embedding
        hybrid_tensors = [torch.cat(parts, dim=0) if len(parts) > 0 else torch.empty(0, embeddings_benign[0].size(1)) for parts in hybrid_parts_per_passive]
        hybrid_emb_all = torch.cat(hybrid_tensors, dim=1) if len(hybrid_tensors[0]) > 0 else torch.empty(0, benign_emb_all.size(1))
        hybrid_labels_all = torch.cat(hybrid_label_list, dim=0) if len(hybrid_label_list) > 0 else torch.empty(0, dtype=torch.long)
        
        final_embeddings = torch.cat([benign_emb_all, hybrid_emb_all], dim=0)
        final_labels     = torch.cat([benign_labels_all , hybrid_labels_all], dim=0)
        
        benign_weights = torch.ones(len(benign_labels_all), device=self.device)
        hybrid_weights = torch.ones(len(hybrid_labels_all), device=self.device) * self.lambda_w
        final_weights = torch.cat([benign_weights, hybrid_weights], dim=0)
                        
        # ========== 微调 top 模型 ==========
        dataset = EmbeddingDataset(final_embeddings.to(self.device), final_labels.to(self.device), final_weights.to(self.device))
        loader  = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam(self.model.active.parameters(), lr=lr)

        self.model.active.train()
        for epoch in range(fine_tune_epochs):
            for batch_idx, (emb_batch, label_batch, weight_bacth) in enumerate(loader):
                optimizer.zero_grad()
                
                # ====== 1. 分类交叉熵损失 ======
                logits = self.model.active(None, None, agged_inputs=emb_batch, agged=True)
                loss1 = criterion(logits, label_batch)
                loss1 = (loss1 * weight_bacth).mean()
                loss = loss1
                loss.backward()
                optimizer.step()
            
            print(f"Epoch [{epoch+1}/{fine_tune_epochs}] Loss={loss.item():.6f}")

        return self.model.active


    def evaluate_defense(self, lr=0.001, fine_tune_epochs=30, top_tunned_name='tunned_top.pth'):
        print("\n============== DEFT Defense Evaluation ==============")
        class_embeddings = self.obtain_active_embeddings()
        
        embs_b1, embs_m1, labs_b1, labs_m1, inds_b1, inds_m1, m1, image_material1 = self.entropy_based_detection(class_embeddings)
        embs_b2, embs_m2, labs_b2, labs_m2, m2, out_pred_dis = self.distance_based_detection(embs_b1, embs_m1, labs_b1, labs_m1, inds_b1, inds_m1)

        top_tuned = self.finetune_top_model(embs_b2, embs_m2, labs_b2, labs_m2, lr=lr, fine_tune_epochs=fine_tune_epochs)

        metrics = [m1, m2]
        
        # 保存 finetuned 模型
        torch.save(top_tuned, top_tunned_name)
        print(f"Saved finetuned top model to {top_tunned_name}")
        return top_tuned, metrics


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
                    data = [inputs[:, :8].to(self.device), inputs[:, 8:].to(self.device)]
                elif self.args.dataset == 'aids':
                    data = [inputs[:, :12].to(self.device), inputs[:, 12:].to(self.device)]
                elif self.args.dataset == 'letter':
                    data = [inputs[:, :8].to(self.device), inputs[:, 8:].to(self.device)]
                else:
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
        # print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(self.test_loader.dataset), test_acc))
        print('ACC: {:.2f}%, {}/{}\n'.format(test_acc, correct, len(self.test_loader.dataset)))
        return test_acc



    def backdoor(self, top_tuned=None):
        if top_tuned is not None:
            self.model.active = top_tuned
        
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (inputs, labels, indices) in enumerate(self.test_loader):
                
                if self.args.dataset == 'cdc':
                    # 这里的CDC其实是Letter数据集
                    # CDC 数据集有 16 个特征，前 8 个和后 8 个拆分
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
                elif self.args.dataset == 'letter':
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

