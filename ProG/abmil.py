import math
import os

import torch
from torch import nn, Tensor
import torch.nn.functional as F
# from architecture.network import Classifier_1fc, DimReduction, DimReduction1
from einops import repeat
# from .nystrom_attention import NystromAttention
try:
    from emb_position import *
except Exception:
    from ProG.emb_position import *

class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        if self.droprate != 0.0:
            self.dropout = torch.nn.Dropout(p=self.droprate)

    def forward(self, x):

        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        return x

class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x



class DimReduction1(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction1, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):
        x_ = x
        x = self.fc1(x)
        x = self.relu1(x+x_)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x


class NystromAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.,
        n_token = 1
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head
        self.n_token = n_token

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, mask = None, return_attn = False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value = 0)

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value = False)

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # set masked positions to 0 in queries, keys, values

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l = l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l = l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l = l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)

        q_landmarks /= divisor
        k_landmarks /= divisor

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        attn1 = einsum(einops_eq, q, k_landmarks)
        attn2 = einsum(einops_eq, q_landmarks, k_landmarks)
        attn3 = einsum(einops_eq, q_landmarks, k)

        # masking

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (attn1, attn2, attn3))
        attn2 = moore_penrose_iter_pinv(attn2, iters)
        out = (attn1 @ attn2) @ (attn3 @ v)

        # add depth-wise conv residual of values
        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        out = out[:, -n:]
        if return_attn:
            attn1 = attn1[:,:,:self.n_token] @ attn2
            attn1 = (attn1 @ attn3)
        
            return out, attn1.mean(1)

        return out


def pos_enc_1d(D, len_seq):
    
    if D % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(D))
    pe = torch.zeros(len_seq, D)
    position = torch.arange(0, len_seq).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, D, 2, dtype=torch.float) *
                         -(math.log(10000.0) / D)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MLP_single_layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_single_layer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x

class TransformWrapper1(nn.Module):
    def __init__(self, conf):
        super(TransformWrapper1, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = MutiHeadAttention2(conf.D_inner, 8)
        self.q = nn.Parameter(torch.zeros((1, conf.n_token, conf.D_inner)))
        nn.init.normal_(self.q, std=1e-6)
        self.n_class = conf.n_class

        self.classifier = nn.ModuleList()
        for i in range(conf.n_token):
            self.classifier.append(Classifier_1fc(conf.D_inner, conf.n_class, 0.0))
        self.n_token = conf.n_token
        self.Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, 0.0)

    def forward(self, input, use_attention_mask=True):
        input = self.dimreduction(input)
        q = self.q
        k = input
        v = input
        outputs = []
        attns = []
        for i in range(self.n_token):
            feat_i, attn_i = self.sub_attention[i](q[:, i].unsqueeze(0), k, v, use_attention_mask=use_attention_mask)
            outputs.append(self.classifier[i](feat_i))
            attns.append(attn_i)

        attns = torch.cat(attns, 1)
        feat_bag = self.bag_attention(v, attns.softmax(dim=-1).mean(1, keepdim=True))

        return torch.cat(outputs, dim=0), self.Slide_classifier(feat_bag), attns
class TransformWrapper(nn.Module):
    def __init__(self, conf):
        super(TransformWrapper, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.sub_attention = nn.ModuleList()
        for i in range(conf.n_token):
            self.sub_attention.append(MutiHeadAttention(conf.D_inner, 8, n_masked_patch=conf.n_masked_patch, mask_drop=conf.mask_drop))
        self.bag_attention = MutiHeadAttention1(conf.D_inner, 8)
        self.q = nn.Parameter(torch.zeros((1, conf.n_token, conf.D_inner)))
        nn.init.normal_(self.q, std=1e-6)
        self.n_class = conf.n_class

        self.classifier = nn.ModuleList()
        for i in range(conf.n_token):
            self.classifier.append(Classifier_1fc(conf.D_inner, conf.n_class, 0.0))
        self.n_token = conf.n_token
        self.Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, 0.0)

    def forward(self, input, use_attention_mask=True):
        input = self.dimreduction(input)
        q = self.q
        k = input
        v = input
        outputs = []
        attns = []
        for i in range(self.n_token):
            feat_i, attn_i = self.sub_attention[i](q[:, i].unsqueeze(0), k, v, use_attention_mask=use_attention_mask)
            outputs.append(self.classifier[i](feat_i))
            attns.append(attn_i)

        attns = torch.cat(attns, 1)
        feat_bag = self.bag_attention(v, attns.softmax(dim=-1).mean(1, keepdim=True))

        return torch.cat(outputs, dim=0), self.Slide_classifier(feat_bag), attns


class MutiHeadAttention2(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.1,
        n_masked_patch: int = 0,
        mask_drop: float = 0.0
    ) -> None:
        super().__init__()
        self.n_masked_patch = n_masked_patch
        self.mask_drop = mask_drop
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj1 = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self.out_proj1 = nn.Linear(self.internal_dim, embedding_dim)

        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor, use_attention_mask=False) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        v1 = self.v_proj1(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)

        if self.n_masked_patch > 0 and use_attention_mask:
            # Get the indices of the top-k largest values
            b, h, q, c = attn.shape
            n_masked_patch = min(self.n_masked_patch, c)
            _, indices = torch.topk(attn, n_masked_patch, dim=-1)
            indices = indices.reshape(b * h * q, -1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(b*h*q, c).to(attn.device)
            random_mask.scatter_(-1, masked_indices, 0)
            attn = attn.masked_fill(random_mask.reshape(b, h, q, -1) == 0, -1e9)

        attn_out = attn
        attn = torch.softmax(attn, dim=-1)
        # Get output
        out1 = attn @ v
        out1 = self._recombine_heads(out1)
        out1 = self.out_proj(out1)
        out1 = self.layer_norm(out1)

        return out1[0], attn_out[0]

class MutiHeadAttention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.1,
        n_masked_patch: int = 0,
        mask_drop: float = 0.0
    ) -> None:
        super().__init__()
        self.n_masked_patch = n_masked_patch
        self.mask_drop = mask_drop
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor, use_attention_mask=False) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)

        if self.n_masked_patch > 0 and use_attention_mask:
            # Get the indices of the top-k largest values
            b, h, q, c = attn.shape
            n_masked_patch = min(self.n_masked_patch, c)
            _, indices = torch.topk(attn, n_masked_patch, dim=-1)
            indices = indices.reshape(b * h * q, -1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(b*h*q, c).to(attn.device)
            random_mask.scatter_(-1, masked_indices, 0)
            attn = attn.masked_fill(random_mask.reshape(b, h, q, -1) == 0, -1e9)

        attn_out = attn
        attn = torch.softmax(attn, dim=-1)
        # Get output
        out1 = attn @ v
        out1 = self._recombine_heads(out1)
        out1 = self.out_proj(out1)
        out1 = self.dropout(out1)
        out1 = self.layer_norm(out1)

        return out1[0], attn_out[0]

class MutiHeadAttention1(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            downsample_rate: int = 1,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, v: Tensor, attn: Tensor) -> Tensor:
        # Input projections
        v = self.v_proj(v)

        # Separate into heads
        v = self._separate_heads(v, self.num_heads)

        # Get output
        out1 = attn @ v
        out1 = self._recombine_heads(out1)
        out1 = self.out_proj(out1)
        out1 = self.dropout(out1)
        out1 = self.layer_norm(out1)

        return out1[0]



class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN


        return A  ### K x N


class AttnMIL(nn.Module):
    def __init__(self, conf, D=128, droprate=0):
        super(AttnMIL, self).__init__()
        self.dimreduction = DimReduction(conf.feat_d, conf.D_inner)
        self.attention = Attention_Gated(conf.D_inner, D, 1)
        self.classifier = Classifier_1fc(conf.D_inner, conf.n_class, droprate)

    def forward(self, x): ## x: N x L
        x = x[0]
        med_feat = self.dimreduction(x)
        A = self.attention(med_feat)  ## K x N

        A_out = A
        A = F.softmax(A, dim=1)  # softmax over N
        afeat = torch.mm(A, med_feat) ## K x L
        outputs = self.classifier(afeat)
        return outputs, A_out.unsqueeze(0)



class AttnMIL1(nn.Module):
    def __init__(self, conf, D=128, droprate=0):
        super(AttnMIL1, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = Attention_Gated(conf.D_inner, D, conf.n_token)
        self.classifier = nn.ModuleList()
        for i in range(conf.n_token):
            self.classifier.append(Classifier_1fc(conf.D_inner, conf.n_class, droprate))
        self.n_masked_patch = conf.n_masked_patch
        self.n_token = conf.n_token
        self.Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, droprate)
        self.mask_drop = conf.mask_drop


    def forward(self, x, use_attention_mask=False): ## x: N x L
        x = x[0]
        x = self.dimreduction(x)
        A = self.attention(x)  ## K x N


        if self.n_masked_patch > 0 and use_attention_mask:
            # Get the indices of the top-k largest values
            k, n = A.shape
            n_masked_patch = min(self.n_masked_patch, n)
            _, indices = torch.topk(A, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(random_mask == 0, -1e9)

        # mask_drop = torch.zeros([0])
        # if self.n_masked_patch > 0 and use_attention_mask:
        #     # Get the indices of the top-k largest values
        #     k, n = A.shape
        #     n_masked_patch = min(self.n_masked_patch, n)
        #     _, indices = torch.topk(A, n_masked_patch, dim=-1)
        #     confidence_values = torch.topk(torch.softmax(A, dim=1), n_masked_patch, dim=-1)[0]
        #     mask_drop = torch.clamp((confidence_values[:,:1].sum(dim=-1) / confidence_values[:,:2].sum(dim=-1) - 0.5) * 2, min=0.2, max=0.9)
        #     shuffled_index = torch.argsort(torch.rand(*indices.shape), dim=-1)
        #     for i in range(self.n_token):
        #         masked_indices = indices[i][shuffled_index[i][:int(mask_drop[i] * n_masked_patch)]]
        #         A[i,masked_indices] = -1e9

        A_out = A
        A = F.softmax(A, dim=1)  # softmax over N
        afeat = torch.mm(A, x) ## K x L
        outputs = []
        for i, head in enumerate(self.classifier):
            outputs.append(head(afeat[i]))
        return torch.stack(outputs, dim=0), self.Slide_classifier(afeat.mean(dim=0, keepdim=True)), A_out.unsqueeze(0)


class AttnMIL4(nn.Module):
    def __init__(self, conf, D=128, droprate=0):
        super(AttnMIL4, self).__init__()
        self.dimreduction = DimReduction(conf.feat_d, conf.D_inner)
        self.attention = Attention_Gated(conf.D_inner, D, conf.n_token)
        self.classifier = nn.ModuleList()
        for i in range(conf.n_token):
            self.classifier.append(Classifier_1fc(conf.D_inner, conf.n_class, droprate))
        self.n_masked_patch = conf.n_masked_patch
        self.n_token = conf.n_token
        self.mask_drop = conf.mask_drop

    def forward(self, x, is_train=True): ## x: N x L
        x = x[0]
        med_feat = self.dimreduction(x)
        A = self.attention(med_feat)  ## K x N

        if self.n_masked_patch > 0 and is_train:
            # Get the indices of the top-k largest values
            k, n = A.shape
            n_masked_patch = min(self.n_masked_patch, n)
            _, indices = torch.topk(A, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(random_mask == 0, -1e9)

        A_out = A
        A = F.softmax(A, dim=1)  # softmax over N
        afeat = torch.mm(A, med_feat) ## K x L
        outputs = []
        # max_conf = []
        for i, head in enumerate(self.classifier):
            output = head(afeat[i])
            outputs.append(output)
            # max_conf.append(torch.softmax(output, dim=0).amax())
        outputs = torch.stack(outputs)
        return outputs, outputs.mean(dim=0, keepdim=True), A_out.unsqueeze(0)

class AttnMIL3(nn.Module):
    def __init__(self, conf, D=128, droprate=0):
        super(AttnMIL3, self).__init__()
        self.dimreduction = DimReduction(conf.feat_d, conf.D_inner)
        self.attention = Attention_Gated(conf.D_inner, D, conf.n_token)
        self.classifier = nn.ModuleList()
        for i in range(conf.n_token):
            self.classifier.append(Classifier_1fc(conf.D_inner, conf.n_class, droprate))
        self.n_masked_patch = conf.n_masked_patch
        self.n_token = conf.n_token
        # self.Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, droprate)
        self.mask_drop = conf.mask_drop

    def forward(self, x, is_train=True): ## x: N x L
        x = x[0]
        med_feat = self.dimreduction(x)
        A = self.attention(med_feat)  ## K x N

        if self.n_masked_patch > 0 and is_train:
            # Get the indices of the top-k largest values
            k, n = A.shape
            n_masked_patch = min(self.n_masked_patch, n)
            _, indices = torch.topk(A, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(random_mask == 0, -1e9)

        A_out = A
        A = F.softmax(A, dim=1)  # softmax over N
        afeat = torch.mm(A, med_feat) ## K x L
        outputs = []
        # max_conf = []
        for i, head in enumerate(self.classifier):
            output = head(afeat[i])
            outputs.append(output)
            # max_conf.append(torch.softmax(output, dim=0).amax())
        outputs = torch.stack(outputs)
        return outputs, outputs.max(axis=0)[0].unsqueeze(0), A_out.unsqueeze(0)



# AttnMIL5 基本没啥用
class AttnMIL5(nn.Module):
    def __init__(self, conf, D=128, droprate=0):
        super(AttnMIL5, self).__init__()
        self.dimreduction = DimReduction(conf.feat_d, conf.D_inner)
        self.attention = Attention_Gated(conf.D_inner, D, conf.n_token)
        self.classifier = nn.ModuleList()
        for i in range(conf.n_token):
            self.classifier.append(Classifier_1fc(conf.D_inner, conf.n_class, droprate))
        self.n_masked_patch = conf.n_masked_patch
        self.n_token = conf.n_token
        self.Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, droprate)
        self.mask_drop = conf.mask_drop

    def forward(self, x, is_train=True): ## x: N x L
        x = x[0]
        med_feat = self.dimreduction(x)
        A = self.attention(med_feat)  ## K x N

        if self.n_masked_patch > 0 and is_train:
            # Get the indices of the top-k largest values
            k, n = A.shape
            n_masked_patch = min(self.n_masked_patch, n)
            _, indices = torch.topk(A, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(random_mask == 0, -1e9)

        A_out = A
        A = F.softmax(A, dim=1)  # softmax over N
        afeat = torch.mm(A, med_feat) ## K x L
        outputs = []
        # max_conf = []
        for i, head in enumerate(self.classifier):
            output = head(afeat[i])
            outputs.append(output)
            # max_conf.append(torch.softmax(output, dim=0).amax())
        outputs = torch.stack(outputs)
        return outputs, self.Slide_classifier(afeat).amax(dim=0).unsqueeze(0), A_out.unsqueeze(0)

class AttnMIL2(nn.Module):
    def __init__(self, conf, D=128, droprate=0):
        super(AttnMIL2, self).__init__()
        self.dimreduction = DimReduction(conf.feat_d, conf.D_inner)
        self.attention1 = Attention_Gated(conf.D_inner, D, conf.n_token)
        self.attention2 = Attention_Gated(conf.D_inner, D, 1)
        self.classifier = nn.ModuleList()
        for i in range(conf.n_token):
            self.classifier.append(Classifier_1fc(conf.D_inner, conf.n_class, droprate))
        self.n_masked_patch = conf.n_masked_patch
        self.n_token = conf.n_token
        self.Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, droprate)

    def forward(self, x, is_train=True): ## x: N x L
        x = x[0]
        med_feat = self.dimreduction(x)
        A = self.attention1(med_feat)  ## K x N

        if self.n_masked_patch > 0 and is_train:
            # Get the indices of the top-k largest values
            k, n = A.shape
            n_masked_patch = min(self.n_masked_patch, n)
            _, indices = torch.topk(A, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * 0.2)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(random_mask == 0, -1e9)

        A_out = A
        A = F.softmax(A, dim=1)  # softmax over N
        afeat = torch.mm(A, med_feat) ## K x L
        outputs = []
        for i, head in enumerate(self.classifier):
            outputs.append(head(afeat[i]))
        A2 = self.attention2(afeat)
        A2 = F.softmax(A2, dim=1)
        afeat = torch.mm(A2, afeat)
        return torch.stack(outputs, dim=0), self.Slide_classifier(afeat), A_out.unsqueeze(0)


class AttnMIL6(nn.Module):
    def __init__(self, conf, D=128, droprate=0):
        super(AttnMIL6, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = Attention_Gated(conf.D_inner, D, conf.n_token)
        self.classifier = nn.ModuleList()
        for i in range(conf.n_token):
            self.classifier.append(Classifier_1fc(conf.D_inner, conf.n_class, droprate))
        self.n_masked_patch = conf.n_masked_patch
        self.n_token = conf.n_token
        self.Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, droprate)
        self.mask_drop = conf.mask_drop


    def forward(self, x, use_attention_mask=False): ## x: N x L
        x = x[0]
        x = self.dimreduction(x)
        A = self.attention(x)  ## K x N


        if self.n_masked_patch > 0 and use_attention_mask:
            # Get the indices of the top-k largest values
            k, n = A.shape
            n_masked_patch = min(self.n_masked_patch, n)
            _, indices = torch.topk(A, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(random_mask == 0, -1e9)

        A_out = A
        A = F.softmax(A, dim=1)  # softmax over N
        afeat = torch.mm(A, x) ## K x L
        # outputs = []
        # for i, head in enumerate(self.classifier):
        #     outputs.append(head(afeat[i]))
        bag_A = F.softmax(A_out, dim=1).mean(0, keepdim=True)
        bag_feat = torch.mm(bag_A, x)
        # return torch.stack(outputs, dim=0), self.Slide_classifier(bag_feat), A_out.unsqueeze(0)
        return bag_feat

    def forward_feature(self, x, use_attention_mask=False): ## x: N x L
        x = x[0]
        x = self.dimreduction(x)
        A = self.attention(x)  ## K x N


        if self.n_masked_patch > 0 and use_attention_mask:
            # Get the indices of the top-k largest values
            k, n = A.shape
            n_masked_patch = min(self.n_masked_patch, n)
            _, indices = torch.topk(A, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(random_mask == 0, -1e9)

        A_out = A
        bag_A = F.softmax(A_out, dim=1).mean(0, keepdim=True)
        bag_feat = torch.mm(bag_A, x)
        return bag_feat



class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, head=8, n_token=1):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            din_tokem=dim,
            dim_head=dim // 8,
            heads=head,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
            n_token=n_token,
        )

    def forward(self, x, need_attn=False):
        if need_attn:
            z, attn = self.attn(self.norm(x), return_attn=need_attn)
            x = x + z
            return x, attn
        else:
            x = x + self.attn(self.norm(x))
            return x


if __name__ == '__main__':
    class Config:
        D_feat = 768  # Number of features in the input instances
        D_inner = 128  # Reduced feature dimension
        n_token = 1    # Number of attention branches (or tokens)
        n_class = 2    # Number of classes for classification
        n_masked_patch = 0  # Number of masked patches, set to 0 if not used
        mask_drop = 0.5  # Proportion of masks to drop if n_masked_patch > 0

    conf = Config()

    # Create an instance of the model
    model = AttnMIL6(conf)

    # Set the model to evaluation mode
    model.eval()

    # Define a random input tensor representing a batch of bags with instances
    # For example, a batch of 5 bags, each containing 10 instances, where each instance has 1024 features
    batch_size = 2
    num_instances = 10
    num_features = conf.D_feat
    input_data = torch.randn(batch_size, num_instances, num_features)

    # Forward pass through the model
    outputs = []
    for i in range(batch_size):
        bag = input_data[i].unsqueeze(0)  # Add batch dimension
        class_output, slide_output, attention_weights = model(bag)
        outputs.append((class_output, slide_output, attention_weights))

    # Print results for the first bag in the batch
    print("Class Output for the first bag:", outputs[0][0])
    print("Slide Classifier Output for the first bag:", outputs[0][1])
    print("Attention Weights for the first bag:", outputs[0][2])