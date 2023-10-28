"""
 -*- coding: utf-8 -*-
time: 2023/10/27 15:11
author: suyunsen
email: suyunsen2023@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from typing import  List,Optional,Tuple
from torch import Tensor
import numpy as np
import logging

from torch.nn.init import xavier_uniform_, constant_

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



# 用于生成层数
def clones(model:nn.Module,N:int)->nn.Module:
    return nn.ModuleList([copy.deepcopy(model) for _ in range(N)])

#把词变为Embedding
class Embeddings(nn.Module):
    max_len:int = 512

    def __init__(self,vocab:int=3000,embed_dim:int=768,max_len:int=512,dropout_p:float = 0.1,training=True):
        super(Embeddings,self).__init__()
        if max_len > self.max_len:
            logging.error("超出最大长度")
        else:
            self.max_len = max_len
        self.embed_dim = embed_dim

        if training is False:
            dropout_p = 0.0

        self.dropout = nn.Dropout(dropout_p)

        # 定义embedding矩阵 其中vocab是词典大小比如bert就是三万多,embed_dim就是每个词被映射的维度
        self.embedding = nn.Embedding(vocab,embed_dim)

    def forward(self,X:Tensor):
        X = self.embedding(X)
        return self.positional_encoding(X)

    def positional_encoding(self,X) -> Tensor:

        P = torch.zeros((1, self.max_len, self.embed_dim))
        X_ = torch.arange(self.max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000,
            torch.arange(0, self.embed_dim, 2, dtype=torch.float32) / self.embed_dim)
        P[:, :, 0::2] = torch.sin(X_)
        P[:, :, 1::2] = torch.cos(X_)
        X = X + P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

#主要用于注意力计算
def multi_head_attention_forward(query:Tensor,
                                 key:Tensor,
                                 value:Tensor,
                                 in_proj_weight:Tensor,
                                 in_proj_bias:Optional[Tensor],
                                 out_proj_weight:Tensor,
                                 out_proj_bias:Optional[Tensor],
                                 attention_mask:Optional[bool],
                                 n_head
                                 )->Tensor:
    """
    注意这里的[s,b,m]顺序是序列长度，bach_size和embed_dim ,中是简单实现我们默认所有bach_size的序列长度一样
    注意query和key和value并不是用于计算的，这个对于新学时肯定有很大疑问
    其实对于encoding 他们三者是同一个值且都是上一层的输出
    :param query: 上一层的输出
    :param key: 上一层的输出
    :param value: 上一层输出
    :param in_proj_weight: K,Q,V权重矩阵的主要来源
    :param in_proj_bias:
    :param out_proj_weight:
    :param out_proj_bias:
    :param attention_mask: mask_attention
    :param n_head: 几个头
    :return attention 结果:
    """
    # 注意对于transformer 在attention中 在解码时q 和 v , k 序列长度可以不一样（翻译任务）
    tar_len,bsize,embed_dim = query.shape
    src_len,_,_ = key.shape
    head_embed_dim = embed_dim // n_head

    # 防止不能整除
    assert head_embed_dim * n_head == embed_dim

    q,k,v = get_qkv_weight(key,query,value,in_proj_weight,in_proj_bias)

    #做反转变为(bach_size,seq,embed_dim) 为什么刚开始要把bach_size放中间是为了多头好变换注意看下面代码

    q = q.contiguous().view(tar_len,bsize * n_head,head_embed_dim).transpose(0,1)
    k = k.contiguous().view(-1,bsize * n_head,head_embed_dim).transpose(0,1)
    v = v.contiguous().view(-1,bsize * n_head,head_embed_dim).transpose(0,1)

    attn_mask = None
    if attention_mask is not None:
        attn_mask = subsequent_mask(bsize*n_head,tar_len,src_len)

    attn_output = calculative_dot_attention(k,q,v,attn_mask)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tar_len, bsize, embed_dim)

    # 得到的结果在做一次线线性变换
    attn_output = F.linear(attn_output,out_proj_weight,out_proj_bias)

    return attn_output

#得到mask矩阵
def subsequent_mask(bs,tar_len,scr_len):
    "Mask out subsequent positions."
    attn_shape = (bs, tar_len, scr_len)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def get_qkv_weight(k,q,v,w,w_bias)->List[Tensor]:
    """

    :param k:
    :param q:
    :param v:
    :param w: 权重大矩阵[3*embed_dim,embed_dim]
    :param w_bais:
    :return:
    """
    # 最后一维是embed_dim
    E = q.size(-1)
    # 主要判段是自主意还是seq2seq
    if k is v and q is v:
        #此时是自主意力,(注意真实的k,q,v,是怎么变来的)
        return F.linear(q,w,w_bias).chunk(3,dim=-1)

    wq,wkv = w.split([E,E*2])
    if w_bias is None:
        b_q = b_kv = None
    else:
        b_q, b_kv = w_bias.split([E, E * 2])
    return (F.linear(q,wq,b_q),) + F.linear(k, wkv, b_kv).chunk(2, dim=-1)

def calculative_dot_attention(k,q,v,attn_mask)->Tensor:
    """
    :param k:
    :param q:
    :param v:
    :param attn_mask: attn_mask 主要用于decode的mask_attention
    :return:
    """
    E = q.size(-1)
    q = q/math.sqrt(E)

    attn_v = torch.bmm(q ,k.transpose(-2,-1))

    if attn_mask is not None:
        attn_v = attn_v.masked_fill(attn_mask == 0, -1e9)

    #这样就可以后可以使没关注的位置置为0
    attn_v = F.softmax(attn_v,dim=-1)

    attn_v = torch.bmm(attn_v,v)
    return attn_v

class MultiheadAttention(nn.Module):

    def __init__(self,d_model,n_head,attention_mask=None,bias=True):
        """

        :param d_model: 模型维度如bert 768
        :param n_head:  几个头
        :param attention_mask:
        :param bias:
        """
        super(MultiheadAttention,self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.attention_mask = attention_mask

        self.in_proj_weight = nn.Parameter(torch.empty((3*d_model,d_model)))

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3*d_model))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj_weight = nn.Linear(d_model,d_model,bias=bias)

        self._reset_parameters()
    def _reset_parameters(self):
        #初始化参数
        xavier_uniform_(self.in_proj_weight)

        #初始化偏执
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj_weight.bias, 0.)

    def forward(self,query:Tensor,key:Tensor,value:Tensor,attn_mask:Optional[Tensor]=None):

       return multi_head_attention_forward(query,key,value
                                           ,self.in_proj_weight,self.in_proj_bias
                                           ,self.out_proj_weight.weight,self.out_proj_weight.bias
                                           ,self.attention_mask,self.n_head)

class TransformerEncodeLayer(nn.Module):

    def __init__(self,d_model,n_head,dim_feedforwad=2048,activate=F.relu,is_attn=False,
                 dropout_d:int = 0.1 , training =True,layer_norm_eps=1e-5):
        """

        :param d_model: 模型维度
        :param n_head: 几个关注头
        :param dim_feedforwad: 前反馈层的输出
        :param activate:
        :param is_attn: 是否自我关注
        :param dropout_d:
        :param training: 是否训练，不训练dropout就为0
        :param layer_norm_eps: 主要防止除数为0
        """
        super(TransformerEncodeLayer,self).__init__()

        if training is False:
            dropout_d = 0.0

        self.dropout1 = nn.Dropout(dropout_d)
        self.dropout2 = nn.Dropout(dropout_d)

        self.norm1 = nn.LayerNorm(d_model,eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model,eps=layer_norm_eps)

        self.line1 = nn.Linear(d_model,dim_feedforwad)
        self.line2 = nn.Linear(dim_feedforwad,d_model)

        self.activate = activate

        self.attn = MultiheadAttention(d_model,n_head)

    def forward(self,X:Tensor):
        out = self.attn(X,X,X)
        out = X + self.dropout1(out)
        out = self.norm1(out)
        new_out = self.line1(out)
        new_out = self.line2(new_out)

        new_out = out + self.dropout2(new_out)

        new_out = self.norm2(new_out)

        #可以看自己是否使用激活函数 , 这里我不使用激活函数了

        return new_out


class TransformerEncoder(nn.Module):

    def __init__(self,d_modle,n_head,num_layer:int = 6,vocab:int=300,dropout_d:int=0.1):

        super(TransformerEncoder,self).__init__()
        self.d_modle = d_modle
        self._embeddings = Embeddings()
        self.training = True
        self.n_head = n_head
        self.encoder = clones(TransformerEncodeLayer(self.d_modle,self.n_head, dropout_d=dropout_d),num_layer)

    def forward(self,X):
        X = self._embeddings(X)
        for fu in self.encoder:
            X = fu(X)
        return X

    def trains(self):
        self.training = True
    def eval(self):
        self.training = False


class TransformerDeconderLayer(nn.Module):

    def __init__(self,d_model,n_head,dim_feedforwad=2048,activate=F.relu,is_attn=False,
                 dropout_d:int = 0.1 , training =True,layer_norm_eps=1e-5):
        """

        :param d_model:
        :param n_head:
        :param dim_feedforwad:
        :param activate:
        :param is_attn:
        :param dropout_d:
        :param training:
        :param layer_norm_eps:
        """
        super(TransformerDeconderLayer,self).__init__()

        if training is False:
            dropout_d = 0.0

        self.attn_self = MultiheadAttention(d_model,n_head)
        self.attn_self_mask = MultiheadAttention(d_model,n_head,attention_mask=True)

        self.dropout1 = nn.Dropout(dropout_d)
        self.dropout2 = nn.Dropout(dropout_d)
        self.dropout3 = nn.Dropout(dropout_d)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.line1 = nn.Linear(d_model, dim_feedforwad)
        self.line2 = nn.Linear(dim_feedforwad, d_model)

        self.activate = activate

    def forward(self,tar_X:Tensor,src_X:Tensor):

        attn_self_mask_out = self.attn_self_mask(tar_X,tar_X,tar_X)
        attn_self_mask_out = tar_X + self.dropout1(attn_self_mask_out)
        attn_self_mask_out = self.norm1(attn_self_mask_out)

        attn_seq2seq = self.attn_self(attn_self_mask_out,src_X,src_X)
        attn_seq2seq = attn_self_mask_out + self.dropout2(attn_seq2seq)
        attn_seq2seq = self.norm2(attn_seq2seq)

        line_out = self.line2(self.line1(attn_seq2seq))
        line_out = attn_seq2seq + self.dropout3(line_out)
        line_out = self.norm3(line_out)

        #使用激活函数
        return line_out


class TransformerDecoder(nn.Module):

    def __init__(self,d_modle,n_head,num_layer:int = 6,vocab:int=300,dropout_d:int=0.1):
        """

        :param d_modle:
        :param n_head:
        :param num_layer:
        :param vocab: 字典大小难得写了默认它300
        :param dropout_d:
        """
        super(TransformerDecoder,self).__init__()
        self.d_modle = d_modle
        self._embeddings = Embeddings()
        self.training = True
        self.n_head = n_head
        self.decoder = clones(TransformerDeconderLayer(self.d_modle, self.n_head, dropout_d=dropout_d), num_layer)

    def forward(self,tar_X:Tensor,src_X:Tensor):
        tar_X = self._embeddings(tar_X)

        for model in  self.decoder:
            tar_X = model(tar_X,src_X)

        return tar_X
    def trains(self):
        self.training = True
    def eval(self):
        self.training = False

#定义模型的encode和deconde
class Transformer(nn.Module):
    """
    基础的Encoder-Decoder结构。
    """

    def __init__(self,en_d_model=768,en_n_head=8,de_d_model=768,de_n_head=8,dropout_d=0.1,training=True):
        """

        :param en_d_model: 编码器每个词维度
        :param en_n_head:  编码器自主意头数目
        :param de_d_model: 解码器每个词维度
        :param de_n_head: 解码器自主意头数目
        :param dropout_d: dropout率
        :param training: 是否训练
        """
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(en_d_model,en_n_head,dropout_d=dropout_d)
        self.decoder = TransformerDecoder(de_d_model,de_n_head,dropout_d=dropout_d)
        self.training = training

    def forward(self, src_X:Tensor, tar_X:Tensor,):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src_X),tar_X )

    def encode(self, src_X):
        return self.encoder(src_X)

    def decode(self, memory, tar_X):
        return self.decoder( tar_X,memory)

    def trains(self):
        self.decoder.trains()
        self.encoder.trains()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

if __name__ == '__main__':

    #测试encoder
    # enmodel = TransformerEncoder(768,12)
    # et = torch.rand((12,3)).to(torch.long)
    # print(enmodel(et))

    #测试decoder
    # em = Embeddings()
    # denmodel = TransformerDecoder(768, 12)
    # dt = torch.rand((12,3)).to(torch.long)
    # src = em(torch.rand((12,3)).to(torch.long))
    # print(denmodel(dt,src))

    #transfoermer测试
    model = Transformer()

    # 记住第二维是bach_size , 第一维是序列长度
    src_x = torch.rand((12,3)).to(torch.long)
    tar_x = torch.rand((16,3)).to(torch.long)

    res = model(src_x,tar_x)

    print(res.shape)