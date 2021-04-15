"""
Original repo: https://github.com/artest08/LateTemporalModeling3DCNN
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import SPATIAL_TEMPORAL_MODULES
from ...core.ops import conv_1x1x1_bn


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _gelu_activation(x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def forward(self, x):
        return self.w_2(self.dropout(self._gelu_activation(self.w_1(x))))


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()

        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        y = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, mask=mask))
        y = self.output_sublayer(y, self.feed_forward)
        y = self.dropout(y)
        return y


class BERTEmbedding(nn.Module):
    def __init__(self, input_dim, max_len, dropout=0.1):
        super().__init__()

        self.max_len = max_len

        # Compute the positional encodings once in log space.
        self.pe = nn.Parameter(torch.Tensor(1, max_len, input_dim))
        self.pe.data.normal_(std=0.02)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence):
        y = self.pe + sequence
        y = self.dropout(y)
        return y


class BERT(nn.Module):
    def __init__(self, input_dim, max_len, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, mask_prob=0.8):
        super().__init__()

        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.max_len = max_len
        self.input_dim = input_dim
        self.mask_prob = mask_prob

        self.cls_token = nn.Parameter(torch.Tensor(1, 1, self.input_dim))
        self.cls_token.data.normal_(std=0.02)

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(input_dim=input_dim, max_len=max_len + 1)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden, attn_heads, self.feed_forward_hidden, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, input_vectors):
        # attention masking for padded token
        batch_size = input_vectors.shape[0]
        if self.training:
            bernoulli_matrix = torch.cat((torch.tensor([1]).float().cuda(),
                                         (torch.tensor([self.mask_prob]).float().cuda()).repeat(self.max_len)),
                                         0).unsqueeze(0).repeat([batch_size, 1])
            bernoulli_distributor = torch.distributions.Bernoulli(bernoulli_matrix)
            sample = bernoulli_distributor.sample()
            mask = (sample > 0).unsqueeze(1).repeat(1, sample.size(1), 1).unsqueeze(1)
        else:
            mask = torch.ones(batch_size, 1, self.max_len + 1, self.max_len + 1).cuda()

        # embedding the indexed sequence to sequence of vectors
        y = torch.cat((self.cls_token.repeat(batch_size, 1, 1), input_vectors), 1)
        y = self.embedding(y)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            y = transformer.forward(y, mask)

        return y


@SPATIAL_TEMPORAL_MODULES.register_module()
class BERTSpatialTemporalModule(nn.Module):
    def __init__(self, in_channels, spatial_size=7, temporal_size=1, hidden_size=256, num_layers=1, num_heads=8):
        super().__init__()

        self.in_channels = in_channels
        self.spatial_size = spatial_size if not isinstance(spatial_size, int) else (spatial_size, spatial_size)
        self.temporal_size = temporal_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.mapper = conv_1x1x1_bn(self.in_channels, self.hidden_size, as_list=False)
        self.spatial_pool = nn.AvgPool3d((1,) + self.spatial_size, stride=1, padding=0)
        self.bert = BERT(self.hidden_size, self.temporal_size,
                         hidden=self.hidden_size, n_layers=self.num_layers, attn_heads=self.num_heads)

    def init_weights(self):
        pass

    def forward(self, x, return_extra_data=False):
        y = self.mapper(x)
        y = self.spatial_pool(y)
        input_vectors = y.view(-1, self.hidden_size, self.temporal_size).transpose(1, 2)

        outputs = self.bert(input_vectors)
        output = outputs[:, 0].view(-1, self.hidden_size, 1, 1, 1)

        if return_extra_data:
            return output, dict()
        else:
            return output
