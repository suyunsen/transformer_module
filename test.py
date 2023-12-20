"""
 -*- coding: utf-8 -*-
time: 2023/10/26 21:24
author: suyunsen
email: suyunsen2023@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
from typing import  List,Optional,Tuple
from torch import Tensor
import numpy as np

def clone(model,N):
    return nn.ModuleList([copy.deepcopy(model) for _ in range(N)])
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule,self).__init__()
        # self.linears = nn.ModuleList([nn.Linear(10, 20) for i in range(10)])
        self.linears = nn.Linear(10,10)
    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        return self.linears(x)

class all(nn.Module):
    def __init__(self,modle):
        super(all,self).__init__()
        self.item = 6
        self.modle = clone(modle,4)

    def forward(self,x):
        x = self.modle(x)

def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    attn = torch.bmm(q, k.transpose(-2, -1))
    attn = F.softmax(attn, dim=-1)
    output = torch.bmm(attn, v)
    return output,attn
def subsequent_mask(bs,tar_len,scr_len):
    "Mask out subsequent positions."
    attn_shape = (bs, tar_len, tar_len)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

if __name__ == '__main__':
    # module = MyModule()
    # model = all(module)
    # a = torch.rand(10)
    # res = module(a)
    # print(res)
    # print(subsequent_mask(3))


    # torch.nn.BCELoss

    #线下变换

    q = torch.rand((5,2,100))
    print(q.size(-1))
    b = torch.rand((300,100))
    c = torch.rand((300))

    res = F.linear(q,b,c)

    print(res.shape)
    # tem = res.chunk(3,dim=-1)[0]
    #
    # t = tem.contiguous().view(-1,2*4,25)
    # print(t)
    # print(t.shape)
    # print(t.transpose(0, 1).shape)
    # tes = True
    # if not tes:
    #     print("hha")
    # else:
    #     print("nihap")
    # print(type(module.parameters()))
    # idx = 1
    # print(sum(p.numel() for p in module.parameters()))
    # cnt = sum(p.numel() for p in model.parameters())
    # print(cnt)
    # q = torch.rand((2,3,100))
    # v = torch.rand((2,3,100))
    # k = torch.rand((2,3,100))
    # q = q.contiguous().view(3, 2 * 4, 25).transpose(0, 1)
    # k = k.contiguous().view(-1, 2 * 4, 25).transpose(0, 1)
    # v = v.contiguous().view(-1, 2 * 4, 25).transpose(0, 1)
    # print(q.shape)
    # print(k.shape)
    # print(v.shape)
    # print(_scaled_dot_product_attention(q,k,v)[0].shape)
    # r = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])
    # mask = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]])
    # print(r.masked_fill(mask == 0, -1e9))
    # attn_shape = (5, 5)
    # print(np.ones(attn_shape))
    # res = np.triu(np.ones(attn_shape),k=1)
    # print(res)
    # v = torch.rand((2,12,100))
    # res = subsequent_mask(2,3,3)
    # print(res.shape)
    # a = torch.randn((2,3,3))
    # print(a)
    # t = a.masked_fill(res == 0 , -1e9)
    # print(t)
    # print(F.softmax(t,dim=-1))
    #
    # # 将输入张量与掩码相乘，实现注意力掩码
    # masked_input = input_tensor * attention_mask.unsqueeze(-1)
    #
    # print(masked_input)
    # P = torch.zeros((1, 512, 768))
    # print(P.shape[1])

    # t = torch.randn((3,5))
    # print(t)
    # t = t.unsqueeze(dim=1)
    # print(t.shape)
    # print(t)
    # t = t.repeat((1,5,1))
    # print(t.shape)
    # print(t)
    # print(t.contiguous().view(-1,5,3))


    # print(t)
    # print(t.shape)




