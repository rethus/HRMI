import copy
import math

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

# 标准化
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out


# 残差 + 标准化
class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        sublayer_x = sublayer(x).squeeze()
        # print("----")
        # print(x.shape, sublayer_x.shape)
        # print("----")
        return self.dropout(self.layer_norm(x + sublayer_x))


#
def self_attention(query, key, value, dropout=None, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.cuda()  # 将mask移到这一行
        scores = scores.masked_fill(mask == 0, -1e9)
    self_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        self_attn = dropout(self_attn)
    return torch.matmul(self_attn, value), self_attn

# def self_attention(query, key, value, dropout=None, mask=None):
#     device = ("cuda" if (torch.cuda.is_available()) else "cpu")
#     d_k = query.size(-1)  # d_k 8
#
#     layerNorm = LayerNorm(d_k).to(device)
#     fc = nn.Linear(d_k, d_k).to(device)
#     LkRelu = nn.LeakyReLU()
#     # torch.Size([64, 4, 1, 8])
#     query, key, value = LkRelu(fc(layerNorm(query))), LkRelu(fc(layerNorm(key))), LkRelu(fc(layerNorm(value)))
#
#     pool = nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)).to(device)
#
#     key = pool(key)  # key torch.Size([64, 4, 1, 8])
#
#     scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # scores torch.Size([64, 4, 1, 1])
#
#     self_attn = F.softmax(scores, dim=-1)  # self_attn torch.Size([64, 4, 1, 1])
#     fc2 = nn.Linear(1, d_k).to(device)
#     # self_attn = self_attn.expand(-1, -1, -1, d_k)  # self_attn torch.Size([64, 4, 1, 8])
#     self_attn = fc2(self_attn)  # self_attn torch.Size([64, 4, 1, 8])
#
#     if dropout is not None:
#         self_attn = dropout(self_attn)
#
#     value = pool(value)  # value torch.Size([64, 4, 1, 8])
#
#     scores = torch.matmul(self_attn, value.transpose(-2, -1))  # scores torch.Size([64, 4, 1, 1])
#
#     scores = F.softmax(scores, dim=-1)  # scores torch.Size([64, 4, 1, 1])
#     scores = scores.expand(-1, -1, -1, d_k)  # scores torch.Size([64, 4, 1, 8])
#
#     return scores, self_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, head, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert (d_model % head == 0)
        self.d_k = d_model // head
        self.head = head
        self.d_model = d_model
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batch = query.size(0)

        query = self.linear_query(query).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)
        key = self.linear_key(key).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)
        value = self.linear_value(value).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)

        x, self.attn = self_attention(query, key, value, dropout=self.dropout, mask=None)

        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k)

        return self.linear_out(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:batch_size, :seq_len]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        # self.layer_norm = LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        out = self.dropout_2(self.w_2(inter))
        return out


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, attn, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x, mask=None):
        x = self.sublayer_connection[0](x, lambda x: self.attn(x, x, x, mask))
        return self.sublayer_connection[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, d_model, d_out, n, encoder_layer, EnablePositionalEncoding):
        super(Encoder, self).__init__()
        self.layerNorm = LayerNorm(d_model)

        self.encoder_layer = clones(encoder_layer, n)

        self.generator = Generator(d_model, d_out)

        self.positionalEncoding = PositionalEncoding(d_model, 64, "cuda" if (torch.cuda.is_available()) else "cpu")
        self.enablePositionalEncoding = EnablePositionalEncoding

    def forward(self, x, src_mask=None):
        if self.enablePositionalEncoding:
            # print(x.size(), print(self.positionalEncoding(x).size()))
            x = x + self.positionalEncoding(x)

        for layer in self.encoder_layer:
            x = layer(x, src_mask)
            # x = F.relu(x)

        return self.generator(x)

