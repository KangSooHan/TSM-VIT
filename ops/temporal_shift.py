# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch
import torch.nn as nn
import torch.nn.functional as F
from indrnn import *

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.num_attention_heads = 12
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = torch.nn.Linear(hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(hidden_size, self.all_head_size)
        self.out = torch.nn.Linear(hidden_size, hidden_size)
        self.softmax = torch.nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        return attention_output, attention_probs

class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=16, n_div=8, dim=256, inplace=False, pool=False):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        self.w1 = torch.nn.Parameter(torch.rand(1, 1, dim)*2-1)
        self.att = Attention(dim)
        self.att_norm = torch.nn.LayerNorm(dim)
        self.pool = pool
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.net(x)
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace, weight=(self.att, self.att_norm, self.w1), pool=self.pool)
        return x

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False, weight=None, pool=False):
        att, att_norm, w1 = weight
        nt, p, c = x.size()
        x = x.view(-1, n_segment, p, c)

        if inplace:
            # Due to some out of order error when performing parallel computing. 
            # May need to write a CUDA kernel.
            raise NotImplementedError
            # out = InplaceShift.apply(x, fold)
        else:
            out = x.clone()

            tokens = x[:, :, 0]

            h = tokens

            past = tokens.clone()

            tokens[:, 1:] += past[:, :-1] * w1

            tokens = att_norm(tokens)
            tokens, _ = att(tokens)
            tokens = tokens + h

            out[:, :, 0] = tokens

        if pool:
            return torch.cat((tokens, out[:, -1, 1:]), 1)

        else:
            return out.view(-1, p, c)


def make_temporal_shift(net, n_segment, num_layers=6):
    def make_block_temporal(stage, n_segment, pool=False):
        stage.identity = TemporalShift(stage.identity, n_segment = n_segment, dim=768, pool=pool)
        return stage

    layers = list(net.transformer.encoder.layer.children())
    for i in range(num_layers):
        layers[i] = make_block_temporal(layers[i], n_segment)
        if i==(num_layers-1):
            layers[i] = make_block_temporal(layers[i], n_segment, pool=True)
