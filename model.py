# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter,LayerNorm,InstanceNorm2d

from utils import ST_BLOCK_0 #ASTGCN
from utils import ST_BLOCK_1 #TSADGCN
from utils import ST_BLOCK_4 #STGCN


"""
the parameters:
x-> [batch_num,in_channels,num_nodes,tem_size],
"""

class ASTGCN_block(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,K,Kt):
        super(ASTGCN_block,self).__init__()
        self.block1=ST_BLOCK_0(c_in,c_out,num_nodes,tem_size,K,Kt)
        self.block2=ST_BLOCK_0(c_out,c_out,num_nodes,tem_size,K,Kt)
        self.final_conv=Conv2d(tem_size,12,kernel_size=(1, c_out),padding=(0,0),
                          stride=(1,1), bias=True)
        self.w=Parameter(torch.zeros(num_nodes,12), requires_grad=True)
        nn.init.xavier_uniform_(self.w)
    def forward(self,x,supports):
        x,_,_ = self.block1(x,supports)
        x,d_adj,t_adj = self.block2(x,supports)
        x = x.permute(0,3,2,1)
        x = self.final_conv(x).squeeze().permute(0,2,1)#b,n,12
        x = x*self.w
        return x,d_adj,t_adj
    
class ASTGCN(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,week,day,recent,K,Kt): 
        super(ASTGCN,self).__init__()
        self.block_w=ASTGCN_block(c_in,c_out,num_nodes,week,K,Kt)
        self.block_d=ASTGCN_block(c_in,c_out,num_nodes,day,K,Kt)
        self.block_r=ASTGCN_block(c_in,c_out,num_nodes,recent,K,Kt)
        self.bn=BatchNorm2d(c_in,affine=False)
        
    def forward(self,x_w,x_d,x_r,supports):
        x_w=self.bn(x_w)
        x_d=self.bn(x_d)
        x_r=self.bn(x_r)
        x_w,_,_=self.block_w(x_w,supports)
        x_d,_,_=self.block_d(x_d,supports)
        x_r,d_adj_r,t_adj_r=self.block_r(x_r,supports)
        out=x_w+x_d+x_r
        return out,d_adj_r,t_adj_r
    
class TSADGCN(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,week,day,recent,K,Kt): 
        super(TSADGCN,self).__init__()
        tem_size=week+day+recent
        self.block1=ST_BLOCK_1(c_in,c_out,num_nodes,tem_size,K,Kt)
        self.block2=ST_BLOCK_1(c_out,c_out,num_nodes,tem_size,K,Kt)
        self.bn=BatchNorm2d(c_in,affine=False)
        self.conv1=Conv2d(c_out,1,kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=True)
        self.conv2=Conv2d(c_out,1,kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=True)
        self.conv3=Conv2d(c_out,1,kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=True)
        self.conv4=Conv2d(c_out,1,kernel_size=(1, 2),padding=(0,0),
                          stride=(1,2), bias=True)
        
        self.h=Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        nn.init.uniform_(self.h, a=0, b=0.0001)
        
    def forward(self,x_w,x_d,x_r,supports):
        x_w=self.bn(x_w)
        x_d=self.bn(x_d)
        x_r=self.bn(x_r)
        x=torch.cat((x_w,x_d,x_r),-1)
        A=self.h+supports
        d=1/(torch.sum(A,-1)+0.0001)
        D=torch.diag_embed(d)
        A=torch.matmul(D,A)
        A1=F.dropout(A,0.5,self.training)
        
        x,_,_=self.block1(x,A1)
        x,d_adj,t_adj=self.block2(x,A1)
    
        x1=x[:,:,:,0:12]
        x2=x[:,:,:,12:24]
        x3=x[:,:,:,24:36]
        x4=x[:,:,:,36:60]
        
        x1=self.conv1(x1).squeeze()
        x2=self.conv2(x2).squeeze()
        x3=self.conv3(x3).squeeze()
        x4=self.conv4(x4).squeeze()#b,n,l
        x=x1+x2+x3+x4
        return x,d_adj,A
    


    
        
class STGCN(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,week,day,recent,K,Kt):
        super(STGCN,self).__init__()
        tem_size=week+day+recent
        self.block1=ST_BLOCK_4(c_in,c_out,num_nodes,tem_size,K,Kt)
        self.block2=ST_BLOCK_4(c_out,c_out,num_nodes,tem_size,K,Kt)
        self.block3=ST_BLOCK_4(c_out,c_out,num_nodes,tem_size,K,Kt)
        
        self.bn=BatchNorm2d(c_in,affine=False)
        self.conv1=Conv2d(c_out,12,kernel_size=(1, recent),padding=(0,0),
                          stride=(1,1), bias=True)
        self.c_out=c_out
        
    def forward(self,x_w,x_d,x_r,supports):
        x=self.bn(x_r)
        shape = x.shape
        
        x=self.block1(x,supports)
        x=self.block2(x,supports)
        x=self.block3(x,supports)
        x=self.conv1(x).squeeze().permute(0,2,1).contiguous()#b,n,l
        return x,supports,supports 

