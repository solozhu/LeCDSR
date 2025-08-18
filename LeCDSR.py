import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np


def load_embeddings(domains):
    item_embeddings = np.load(f'./dataset/{domains}/item_embeddings.npy')
    user_embeddings = np.load(f'./dataset/{domains}/reasoning_user_embeddings.npy')
    return item_embeddings, user_embeddings


class DTRLayer(nn.Module):
    """Distinguishable Textual Representations Layer

    """
    def __init__(self, input_size, output_size, dropout=0.0, max_seq_length=50):
        super(DTRLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(1, max_seq_length, input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        """前向传播

        Args:
            x: 输入

        Returns:
            output: 输出
        """
        return self.lin(self.dropout(x) - self.bias)

class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor
    """
    def __init__(self, n_exps, layers, dropout=0.0, max_seq_length=15, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([DTRLayer(layers[0], layers[1], dropout, max_seq_length) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        """前向传播

        Args:
            x: 输入(batch, max_seq_length, input_size)

        Returns:
            output: 输出(batch, max_seq_length, output_size)
        """
        gates = self.noisy_top_k_gating(x, self.training) # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)


class Contrastive_Loss(nn.Module):
    def __init__(self, tau=1) -> None:
        super().__init__()

        self.temperature = tau

    def forward(self, X, Y):
        """
        Args:
            X: (batch_size, num_samples, feat_dim)
            Y: (batch_size, num_samples, feat_dim)
        """
        batch_size, num_samples, feat_dim = X.shape

        # Compute logits: (batch_size, num_samples, num_samples)
        logits = torch.bmm(X, Y.transpose(1, 2)) / self.temperature

        # Compute similarity matrices
        X_similarity = torch.bmm(Y, Y.transpose(1, 2))  # (batch_size, num_samples, num_samples)
        Y_similarity = torch.bmm(X, X.transpose(1, 2))  # (batch_size, num_samples, num_samples)

        # Compute targets
        targets = F.softmax(
            (X_similarity + Y_similarity) / (2 * self.temperature),
            dim=-1
        )
        # Compute losses
        X_loss = -torch.sum(targets * F.log_softmax(logits, dim=-1), dim=-1)  # (batch_size, num_samples)
        Y_loss = -torch.sum(targets * F.log_softmax(logits.transpose(1, 2), dim=-1),
                            dim=-1)  # (batch_size, num_samples)

        # Average across samples and batch
        loss = (X_loss + Y_loss) / 2.0  # (batch_size, num_samples)
        return loss.mean()


class LeCDSR(nn.Module):
    def __init__(self,
                 model_args):
        super(LeCDSR, self).__init__()

        self.args = model_args

        # init args
        L = self.args.L
        dims = self.args.d
        item_num = self.args.item_num
        user_num = self.args.user_num
        domains = self.args.data_dir
        self.use_llm_emb = self.args.use_llm_emb
        self.use_llm_user_emb = self.args.use_llm_user_emb
        self.use_context_conv = self.args.use_context_conv

        # user and item embeddings
        text_embs, user_embs = load_embeddings(domains)
        if self.use_llm_emb:
            self.llm_item = nn.Embedding.from_pretrained(torch.tensor(text_embs, dtype=torch.float32),
                                                         freeze=True)
        if self.use_llm_user_emb:
            self.llm_user = nn.Embedding.from_pretrained(torch.tensor(user_embs, dtype=torch.float32),
                                                         freeze=True)

        self.user_embeddings = nn.Embedding(user_num, dims)
        self.item_embeddings = nn.Embedding(item_num, dims, padding_idx=item_num - 1)
        self.position_embeddings = nn.Embedding(L + 1, dims, padding_idx=0)

        self.feature_gating = nn.Sequential(
            nn.Sigmoid()
        )
        self.instance_gating = nn.Sequential(
            nn.Sigmoid()
        )
        # feature gate
        self.feature_gate_user = nn.Linear(dims, dims)  # 对应论文中的参数 Wg1
        self.feature_gate_item = nn.Linear(dims, dims)  # 对应论文中的参数 Wg4
        # self.feature_gate_embedding = nn.Linear(dims, dims)  # 特征门控用来捕获物品特征中的重点，在此基础上额外捕获LLM

        # instance gate
        # 对应论文中的参数 Wg5
        self.instance_gate_user = nn.Parameter(torch.zeros(dims, L, dtype=torch.float32, requires_grad=True))
        # 对应论文中的参数 wg6
        self.instance_gate_position = nn.Parameter(torch.zeros(dims, 1, dtype=torch.float32, requires_grad=True))
        # 对应论文中的参数 wg9
        self.instance_gate_item = nn.Parameter(torch.zeros(dims, 1, dtype=torch.float32, requires_grad=True))

        self.instance_gate_user = torch.nn.init.xavier_uniform_(self.instance_gate_user)
        self.instance_gate_item = torch.nn.init.xavier_uniform_(self.instance_gate_item)
        self.instance_gate_position = torch.nn.init.xavier_uniform_(self.instance_gate_position)

        # items_to_predict embeddings
        self.W2 = nn.Embedding(item_num, dims, padding_idx=item_num - 1)
        self.b2 = nn.Embedding(item_num, 1, padding_idx=item_num - 1)

        # weight initialization

        self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        self.position_embeddings.weight.data.normal_(0, 1.0 / self.position_embeddings.embedding_dim)

        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()
        if self.use_llm_emb:
            # llm嵌入对齐层
            self.adapter_text = nn.Sequential(
                nn.Linear(1024, 768),
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Linear(768, dims)
                # nn.Dropout(0.2)
            )
        if self.use_llm_user_emb:
            self.adapter_user = nn.Sequential(
                nn.Linear(1024, 768),
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Linear(768, dims)
                # nn.Dropout(0.2)
            )
        if self.use_context_conv:
            # 上下文卷积参数
            self.complex_weight = nn.Parameter(
                torch.randn(1, L // 2 + 1, dims, 2, dtype=torch.float32) * 0.02)
            self.item_gating = nn.Linear(dims, 1)
            self.fusion_gating = nn.Linear(dims, 1)
        self.align_loss = Contrastive_Loss(self.args.tau2)

    def forward(self, seq, position, user_ids, items_to_predict):
        item_embs = self.item_embeddings(seq)
        position_embs = self.position_embeddings(position)
        if self.use_llm_emb:
            llm_embs = self.adapter_text(self.llm_item(seq))
        if self.use_llm_user_emb:
            user_embs = self.adapter_user(self.llm_user(user_ids))
        else:
            user_embs = self.user_embeddings(user_ids)
        # feature gating
        if self.use_llm_emb:
            if self.use_context_conv:
                item_embs = self.contextual_convolution(item_embs, llm_embs)
                gate = self.feature_gating(self.feature_gate_item(item_embs)
                                           + self.feature_gate_user(user_embs)
                                           )
            else:
                gate = self.feature_gating(self.feature_gate_item(item_embs)
                                           + self.feature_gate_item(llm_embs)
                                           + self.feature_gate_user(user_embs)
                                           )
        else:
            gate = self.feature_gating(self.feature_gate_item(item_embs)
                                       + self.feature_gate_user(user_embs)
                                       )
        gated_item = item_embs * gate

        # instance gating
        # instance score (batch, L)
        instance_score = self.instance_gating(
            torch.matmul(gated_item, self.instance_gate_item.unsqueeze(0)).squeeze()
            + torch.matmul(user_embs, self.instance_gate_user.unsqueeze(0)).squeeze()
            + torch.matmul(position_embs, self.instance_gate_position.unsqueeze(0)).squeeze()
        )
        #           (B, L, d) * (B, L, )
        union_out = gated_item * instance_score.unsqueeze(2)

        # if self.use_context_conv:
        #     union_out = self.contextual_convolution(union_out, llm_embs)

        # 在L上做加权平均
        union_out = torch.sum(union_out, dim=1)
        union_out = union_out / torch.sum(instance_score, dim=1, keepdim=True)

        w2 = self.W2(items_to_predict)  # 此处引入了候选推荐矩阵
        b2 = self.b2(items_to_predict)

        # MF
        res = torch.baddbmm(b2, w2, user_embs.permute(0, 2, 1)).squeeze()  # (B, 100, 1)  (B, 100, dim) @ (B, 1, dim)

        # union level
        res += torch.bmm(union_out.unsqueeze(1), w2.permute(0, 2, 1)).squeeze()

        # item-item product
        rel_score = item_embs.bmm(w2.permute(0, 2, 1))
        rel_score = torch.mean(rel_score, dim=1)
        res += rel_score

        return res

    def contextual_convolution(self, item_emb, feature_emb):
        """Sequence-Level Representation Fusion
        """
        feature_fft = torch.fft.rfft(feature_emb, dim=1, norm='ortho')
        item_fft = torch.fft.rfft(item_emb, dim=1, norm='ortho')

        complext_weight = torch.view_as_complex(self.complex_weight)
        item_conv = torch.fft.irfft(item_fft * complext_weight, n=feature_emb.shape[1], dim=1, norm='ortho')
        fusion_conv = torch.fft.irfft(feature_fft * item_fft, n=feature_emb.shape[1], dim=1, norm='ortho')

        item_gate_w = self.item_gating(item_conv)
        fusion_gate_w = self.fusion_gating(fusion_conv)

        contextual_emb = (item_conv * torch.sigmoid(item_gate_w) + fusion_conv * torch.sigmoid(fusion_gate_w))
        return contextual_emb

    def cal_align_loss(self, seq):
        llm_embs = self.adapter_text(self.llm_item(seq))
        item_embs = self.item_embeddings(seq)
        cl1 = self.align_loss(item_embs, llm_embs)
        cl2 = self.align_loss(llm_embs, item_embs)
        return cl1 + cl2
