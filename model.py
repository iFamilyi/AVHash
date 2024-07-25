import os
from transformers import BertModel, BertConfig
import json
from math import sqrt
import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-3):
        super(LayerNormalization, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z

        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)

        return ln_out

class MMVH(nn.Module):

    def __init__(self, model_config_path, cls_path, hashcode_size):
        super(MMVH, self).__init__()

        self.model_config_path = model_config_path
        self.cls_path = cls_path
        with open(model_config_path + '/config.json') as json_file:
            data = json.load(json_file)
        self.hidden_size = data['hidden_size']
        self.hashcode_size = hashcode_size

        # 音频，图片，视频编码器
        self.AE = EncoderModel(self.model_config_path, self.cls_path)
        self.IE = EncoderModel(self.model_config_path, self.cls_path)
        self.VE = EncoderModel(self.model_config_path, self.cls_path)

        # 交叉注意力
        self.AqI = Multi_CrossAttention(self.model_config_path) # A 作为 Query
        self.IqA = Multi_CrossAttention(self.model_config_path) # I 作为 Query

        self.Audio_project = nn.Linear(768,768)

        # 各个模态模态内的加权残差权重
        self.A_W = nn.Linear(2*self.hidden_size, 2)
        self.I_W = nn.Linear(2*self.hidden_size, 2)

        # 不同模态的加权
        self.AI_W = nn.Linear(2*self.hidden_size, 2)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmiod = nn.Sigmoid()

        self.hash = HashLayer(hidden_size=self.hidden_size, hashcode_size=self.hashcode_size)

    def forward(self, I_input, A_input):

        A_enc_pooler_output, A_enc_last_hidden_state = self.AE(A_input)
        I_enc_pooler_output, I_enc_last_hidden_state = self.IE(I_input)

        A_enc_25frames = A_enc_last_hidden_state[:,1:]
        I_enc_25frames = I_enc_last_hidden_state[:,1:]
        A_enc_25frames_ = A_enc_25frames + A_input
        I_enc_25frames_ = I_enc_25frames + I_input

        AqI = self.AqI(A_enc_25frames_, I_enc_25frames_)
        IqA = self.IqA(I_enc_25frames_, A_enc_25frames_)

        AeAc = torch.cat((A_enc_25frames_,AqI),dim=-1)
        IeIc = torch.cat((I_enc_25frames_,IqA),dim=-1)
        r_A = self.tanh(self.A_W(AeAc))
        r_I = self.tanh(self.I_W(IeIc))
        w_A = self.softmax(r_A)
        w_I = self.softmax(r_I)
        A_fuse =  w_A[:, :, 0].unsqueeze(-1) * A_enc_25frames_ +  w_A[:, :, 1].unsqueeze(-1) * AqI
        I_fuse =  w_I[:, :, 0].unsqueeze(-1) * I_enc_25frames_ +  w_I[:, :, 1].unsqueeze(-1) * IqA

        IfAf = torch.cat((I_fuse, A_fuse),dim=-1)
        IA_w = self.softmax(self.tanh(self.AI_W(IfAf)))
        IA_fuse = IA_w[:, :, 0].unsqueeze(-1) * I_fuse + IA_w[:, :, 1].unsqueeze(-1) * A_fuse

        v_enc_pooler_output, v_enc_last_hidden_state = self.VE(IA_fuse)

        vh = self.hash(v_enc_pooler_output)

        return A_enc_pooler_output, I_enc_pooler_output, vh


class EncoderModel(nn.Module):
    """ text processed by bert model encode and get cls vector for multi classification
    """

    def __init__(self, model_config_path, cls_path):
        super(EncoderModel, self).__init__()
        self.encoder = BertModel.from_pretrained(model_config_path)
        self.cls = torch.load(cls_path).float()

    def forward(self, x):

        cls = self.cls.to(x.device)
        batch = x.shape[0]
        expanded_cls = cls.expand(batch, -1)
        input = torch.cat((expanded_cls.unsqueeze(1), x), dim=1)

        output = self.encoder(inputs_embeds = input)

        return output.pooler_output, output.last_hidden_state



class CalculateAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        attention = torch.matmul(Q, torch.transpose(K, -1, -2))
        # use mask
        attention = torch.softmax(attention / torch.sqrt(torch.tensor(K.size(-1))), dim=-1)
        attention = torch.matmul(attention, V)
        return attention


class Multi_CrossAttention(nn.Module):
    """
    forward时，第一个参数用于计算query，第二个参数用于计算key和value
    """
    def __init__(self, model_config_path):
        super().__init__()

        with open(model_config_path+'/config.json', 'r') as f:
            config = json.load(f)

        self.hidden_size = config['hidden_size']  # 输入维度
        self.all_head_size = config['hidden_size']  # 输出维度
        self.num_heads = config['num_attention_heads']  # 注意头的数量
        self.h_size = self.all_head_size // self.num_heads

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.linear_k = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.linear_v = nn.Linear(self.hidden_size, self.all_head_size, bias=False)
        self.linear_output = nn.Linear(self.all_head_size, self.hidden_size)

        # normalization
        self.norm = sqrt(self.all_head_size)

    def print(self):
        print(self.hidden_size, self.all_head_size)
        print(self.linear_k, self.linear_q, self.linear_v)

    def forward(self, x, y):

        batch_size = x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # q_s: [batch_size, num_heads, seq_length, h_size]
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        attention = CalculateAttention()(q_s, k_s, v_s)
        # attention : [batch_size , seq_length , num_heads * h_size]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)

        # output : [batch_size , seq_length , hidden_size]
        output = self.linear_output(attention)

        return output



class HashLayer(nn.Module):

    def __init__(self, hidden_size, hashcode_size):
        super(HashLayer, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hashcode_size)
        self.hashac = nn.Tanh()

    def forward(self, x):
        x = self.hashac(self.linear1(x))
        return x




