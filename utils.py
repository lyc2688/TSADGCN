# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter, LayerNorm, BatchNorm1d

"""
x-> [batch_num,in_channels,num_nodes,tem_size],
"""

class TATT(nn.Module):
    def __init__(self,c_in,num_nodes,tem_size):
        super(TATT,self).__init__()
        self.conv1=Conv2d(c_in, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.conv2=Conv2d(num_nodes, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.w=nn.Parameter(torch.rand(num_nodes,c_in), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w)
        self.b=nn.Parameter(torch.zeros(tem_size,tem_size), requires_grad=True)
        
        self.v=nn.Parameter(torch.rand(tem_size,tem_size), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.v)
        
    def forward(self,seq):
        c1 = seq.permute(0,1,3,2)#b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze(1)#b,l,n
        
        c2 = seq.permute(0,2,1,3)#b,c,n,l->b,n,c,l
        f2 = self.conv2(c2).squeeze(1)#b,c,l
     
        logits = torch.sigmoid(torch.matmul(torch.matmul(f1,self.w),f2)+self.b)
        logits = torch.matmul(self.v,logits)
        ##normalization
        a,_ = torch.max(logits, 1, True)
        logits = logits - a
        coefs = torch.softmax(logits,-1)
        return coefs
    
class SATT(nn.Module):
    def __init__(self,c_in,num_nodes,tem_size):
        super(SATT,self).__init__()
        self.conv1=Conv2d(c_in, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.conv2=Conv2d(tem_size, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.w=nn.Parameter(torch.rand(tem_size,c_in), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w)
        self.b=nn.Parameter(torch.zeros(num_nodes,num_nodes), requires_grad=True)
        
        self.v=nn.Parameter(torch.rand(num_nodes,num_nodes), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.v)
        
    def forward(self,seq):
        c1 = seq
        f1 = self.conv1(c1).squeeze(1)#b,n,l
        
        c2 = seq.permute(0,3,1,2)#b,c,n,l->b,l,n,c
        f2 = self.conv2(c2).squeeze(1)#b,c,n
     
        logits = torch.sigmoid(torch.matmul(torch.matmul(f1,self.w),f2)+self.b)
        logits = torch.matmul(self.v,logits)
        ##normalization
        a,_ = torch.max(logits, 1, True)
        logits = logits - a
        coefs = torch.softmax(logits,-1)
        return coefs

class cheby_conv_ds(nn.Module):
    def __init__(self,c_in,c_out,K):
        super(cheby_conv_ds,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv2d(c_in_new, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        self.K=K
        
    def forward(self,x,adj,ds):
        nSample, feat_in,nNode, length  = x.shape
        Ls = []
        L0 = torch.eye(nNode).cuda()
        L1 = adj
    
        L = ds*adj
        I = ds*torch.eye(nNode).cuda()
        Ls.append(I)
        Ls.append(L)
        for k in range(2, self.K):
            L2 = 2 *torch.matmul( adj, L1) - L0
            L0, L1 = L1, L2
            L3 =ds*L2
            Ls.append(L3)

        Lap = torch.stack(Ls, 1)  # [B, K,nNode, nNode]
        #print(Lap)
        Lap = Lap.transpose(-1,-2)
        x = torch.einsum('bcnl,bknq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out  

    
###ASTGCN_block
class ST_BLOCK_0(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,K,Kt):
        super(ST_BLOCK_0,self).__init__()
        
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        self.TATT=TATT(c_in,num_nodes,tem_size)
        self.SATT=SATT(c_in,num_nodes,tem_size)
        self.dynamic_gcn=cheby_conv_ds(c_in,c_out,K)
        self.K=K
        
        self.time_conv=Conv2d(c_out, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        #self.bn=BatchNorm2d(c_out)
        self.bn=LayerNorm([c_out,num_nodes,tem_size])
        
    def forward(self,x,supports):
        x_input=self.conv1(x)
        T_coef=self.TATT(x)
        T_coef=T_coef.transpose(-1,-2)
        x_TAt=torch.einsum('bcnl,blq->bcnq',x,T_coef)
        S_coef=self.SATT(x)#B x N x N
        
        spatial_gcn=self.dynamic_gcn(x_TAt,supports,S_coef)
        spatial_gcn=torch.relu(spatial_gcn)
        time_conv_output=self.time_conv(spatial_gcn)
        out=self.bn(torch.relu(time_conv_output+x_input))
        
        return  out,S_coef,T_coef    
     


###1###
class T_cheby_conv(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''
    def __init__(self,c_in,c_out,K,Kt):
        super(T_cheby_conv,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv2d(c_in_new, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        self.K=K
        
        
    def forward(self,x,adj):
        nSample, feat_in, nNode, length  = x.shape
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 *torch.matmul( adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 0)  # [K,nNode, nNode]
        Lap = Lap.transpose(-1,-2)
        #print(Lap)
        x = torch.einsum('bcnl,knq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out 

class ST_BLOCK_1(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,K,Kt):
        super(ST_BLOCK_1,self).__init__()
        
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        self.TATT_1=TATT_1(c_out,num_nodes,tem_size)
        self.dynamic_gcn=T_cheby_conv(c_out,2*c_out,K,Kt)
        self.K=K
        self.time_conv=Conv2d(c_in, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        #self.bn=BatchNorm2d(c_out)
        self.c_out=c_out
        self.bn=LayerNorm([c_out,num_nodes,tem_size])
    def forward(self,x,supports):
        x_input=self.conv1(x)
        x_1=self.time_conv(x)
        x_1=F.leaky_relu(x_1)
        x_1=F.dropout(x_1,0.5,self.training)
        x_1=self.dynamic_gcn(x_1,supports)
        filter,gate=torch.split(x_1,[self.c_out,self.c_out],1)
        x_1=torch.sigmoid(gate)*F.leaky_relu(filter)
        x_1=F.dropout(x_1,0.5,self.training)
        T_coef=self.TATT_1(x_1)
        T_coef=T_coef.transpose(-1,-2)
        x_1=torch.einsum('bcnl,blq->bcnq',x_1,T_coef)
        out=self.bn(F.leaky_relu(x_1)+x_input)
        return out,supports,T_coef
        
    


class SATT_3(nn.Module):
    def __init__(self,c_in,num_nodes):
        super(SATT_3,self).__init__()
        self.conv1=Conv2d(c_in*12, c_in, kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=False)
        self.conv2=Conv2d(c_in*12, c_in, kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=False)
        self.bn=LayerNorm([num_nodes,num_nodes,4])
        self.c_in=c_in
    def forward(self,seq):
        shape = seq.shape
        seq=seq.permute(0,1,3,2).contiguous().view(shape[0],shape[1]*12,shape[3]//12,shape[2])
        seq=seq.permute(0,1,3,2)
        shape = seq.shape
        #b,c*12,n,l//12
        f1 = self.conv1(seq).view(shape[0],self.c_in//4,4,shape[2],shape[3]).permute(0,3,1,4,2).contiguous()
        f2 = self.conv2(seq).view(shape[0],self.c_in//4,4,shape[2],shape[3]).permute(0,1,3,4,2).contiguous()
        
        logits = torch.einsum('bnclm,bcqlm->bnqlm',f1,f2)
        #a,_ = torch.max(logits, -1, True)
        #logits = logits - a
        #logits = logits.permute(0,2,1,3).contiguous()
        #logits=self.bn(logits).permute(0,3,2,1).contiguous()
        logits=logits.permute(0,3,1,2,4).contiguous()
        logits = torch.sigmoid(logits)
        logits = torch.mean(logits,-1)
        return logits
    
class SATT_2(nn.Module):
    def __init__(self,c_in,num_nodes):
        super(SATT_2,self).__init__()
        self.conv1=Conv2d(c_in, c_in, kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=False)
        self.conv2=Conv2d(c_in, c_in, kernel_size=(1, 1),padding=(0,0),
                          stride=(1,1), bias=False)
        self.bn=LayerNorm([num_nodes,num_nodes,12])
        self.c_in=c_in
    def forward(self,seq):
        shape = seq.shape
        f1 = self.conv1(seq).view(shape[0],self.c_in//4,4,shape[2],shape[3]).permute(0,3,1,4,2).contiguous()
        f2 = self.conv2(seq).view(shape[0],self.c_in//4,4,shape[2],shape[3]).permute(0,1,3,4,2).contiguous()
        
        logits = torch.einsum('bnclm,bcqlm->bnqlm',f1,f2)
        #a,_ = torch.max(logits, -1, True)
        #logits = logits - a
        #logits = logits.permute(0,2,1,3).contiguous()
        #logits=self.bn(logits).permute(0,3,2,1).contiguous()
        logits=logits.permute(0,3,1,2,4).contiguous()
        logits = torch.sigmoid(logits)
        logits = torch.mean(logits,-1)
        return logits

A=np.zeros((60,60))
for i in range(12):
    for j in range(12):
        A[i,j]=1
        A[i+12,j+12]=1
        A[i+24,j+24]=1
for i in range(24):
    for j in range(24):        
        A[i+36,j+36]=1
B=(-1e13)*(1-A)  
B=(torch.tensor(B)).type(torch.float32).cuda()   
class TATT_1(nn.Module):
    def __init__(self,c_in,num_nodes,tem_size):
        super(TATT_1,self).__init__()
        self.conv1=Conv2d(c_in, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.conv2=Conv2d(num_nodes, 1, kernel_size=(1, 1),
                          stride=(1,1), bias=False)
        self.w=nn.Parameter(torch.rand(num_nodes,c_in), requires_grad=True)
        nn.init.xavier_uniform_(self.w)
        self.b=nn.Parameter(torch.zeros(tem_size,tem_size), requires_grad=True)
        
        self.v=nn.Parameter(torch.rand(tem_size,tem_size), requires_grad=True)
        nn.init.xavier_uniform_(self.v)
        self.bn=BatchNorm1d(tem_size)
        
    def forward(self,seq):
        c1 = seq.permute(0,1,3,2)#b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze()#b,l,n
        
        c2 = seq.permute(0,2,1,3)#b,c,n,l->b,n,c,l
        f2 = self.conv2(c2).squeeze()#b,c,n
     
        logits = torch.sigmoid(torch.matmul(torch.matmul(f1,self.w),f2)+self.b)
        logits = torch.matmul(self.v,logits)
        ##normalization
        #logits=tf_util.batch_norm_for_conv1d(logits, is_training=training,
            #                                   bn_decay=bn_decay, scope='bn')
        #a,_ = torch.max(logits, 1, True)
        #logits = logits - a
        
        logits = logits.permute(0,2,1).contiguous()
        logits=self.bn(logits).permute(0,2,1).contiguous()
        coefs = torch.softmax(logits+B,-1)
        return coefs

    
    
class ST_BLOCK_2(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,K,Kt):
        super(ST_BLOCK_2,self).__init__()
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        self.TATT_1=TATT_1(c_out,num_nodes,tem_size)
        self.SATT_3=SATT_3(c_out,num_nodes)
        self.SATT_2=SATT_2(c_out,num_nodes)
        self.dynamic_gcn=T_cheby_conv_ds(c_out,2*c_out,K,Kt)
        self.LSTM=nn.LSTM(num_nodes,num_nodes,batch_first=True)#b*n,l,c
        self.K=K
        self.tem_size=tem_size
        self.time_conv=Conv2d(c_in, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        #self.bn=BatchNorm2d(c_out)
        self.c_out=c_out
        self.bn=LayerNorm([c_out,num_nodes,tem_size])
        
        
    def forward(self,x,supports):
        x_input=self.conv1(x)
        x_1=self.time_conv(x)
        x_1=F.leaky_relu(x_1)
        x_tem1=x_1[:,:,:,0:48]
        x_tem2=x_1[:,:,:,48:60]
        S_coef1=self.SATT_3(x_tem1)
        #print(S_coef1.shape)
        S_coef2=self.SATT_2(x_tem2)
        #print(S_coef2.shape)
        S_coef=torch.cat((S_coef1,S_coef2),1)#b,l,n,c
        shape=S_coef.shape
        #print(S_coef.shape)
        h = Variable(torch.zeros((1,shape[0]*shape[2],shape[3]))).cuda()
        c=Variable(torch.zeros((1,shape[0]*shape[2],shape[3]))).cuda()
        hidden=(h,c)
        S_coef=S_coef.permute(0,2,1,3).contiguous().view(shape[0]*shape[2],shape[1],shape[3])
        S_coef=F.dropout(S_coef,0.5,self.training) #2020/3/28/22:17,试验下效果
        _,hidden=self.LSTM(S_coef,hidden)
        adj_out=hidden[0].squeeze().view(shape[0],shape[2],shape[3]).contiguous()
        adj_out1=(adj_out)*supports
        x_1=F.dropout(x_1,0.5,self.training)
        x_1=self.dynamic_gcn(x_1,adj_out1)
        filter,gate=torch.split(x_1,[self.c_out,self.c_out],1)
        x_1=torch.sigmoid(gate)*F.leaky_relu(filter)
        x_1=F.dropout(x_1,0.5,self.training)
        T_coef=self.TATT_1(x_1)
        T_coef=T_coef.transpose(-1,-2)
        x_1=torch.einsum('bcnl,blq->bcnq',x_1,T_coef)
        out=self.bn(F.leaky_relu(x_1)+x_input)
        return out,adj_out,T_coef


class ST_BLOCK_3(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,K,Kt):
        super(ST_BLOCK_3,self).__init__()
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        self.TATT_1=TATT_1(c_out,num_nodes,tem_size)
        self.dynamic_gcn=GAT(c_out,c_out//1,0.5,0.3,1,tem_size,Kt)
        self.dynamic_gcn1=GAT(c_out,c_out//1,0.5,0.3,1,tem_size,Kt)
        self.K=K
        
        self.time_conv=Conv2d(c_in, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        #self.bn=BatchNorm2d(c_out)
        self.c_out=c_out
        self.bn=LayerNorm([c_out,num_nodes,tem_size])
        
    def forward(self,x,supports):
        x_input=self.conv1(x)
        x_1=self.time_conv(x)
        x_1=F.leaky_relu(x_1)
        x_1=F.dropout(x_1,0.5,self.training)
        filter,S_coef=self.dynamic_gcn(x_1,supports)
        gate,_=self.dynamic_gcn1(x_1,supports)
        x_1=torch.sigmoid(gate)*F.leaky_relu(filter)
        x_1=F.dropout(x_1,0.5,self.training)
        T_coef=self.TATT_1(x_1)
        T_coef=T_coef.transpose(-1,-2)
        x_1=torch.einsum('bcnl,blq->bcnq',x_1,T_coef)
        out=self.bn(F.leaky_relu(x_1)+x_input)
        return out,S_coef,T_coef

###STGCN
class cheby_conv(nn.Module):
    '''
    x : [batch_size, feat_in, num_node ,tem_size] - input of all time step
    nSample : number of samples = batch_size
    nNode : number of node in graph
    tem_size: length of temporal feature
    c_in : number of input feature
    c_out : number of output feature
    adj : laplacian
    K : size of kernel(number of cheby coefficients)
    W : cheby_conv weight [K * feat_in, feat_out]
    '''
    def __init__(self,c_in,c_out,K,Kt):
        super(cheby_conv,self).__init__()
        c_in_new=(K)*c_in
        self.conv1=Conv2d(c_in_new, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        self.K=K
        
        
    def forward(self,x,adj):
        nSample, feat_in, nNode, length  = x.shape
        Ls = []
        L1 = adj
        L0 = torch.eye(nNode).cuda()
        Ls.append(L0)
        Ls.append(L1)
        for k in range(2, self.K):
            L2 = 2 *torch.matmul( adj, L1) - L0
            L0, L1 = L1, L2
            Ls.append(L2)

        Lap = torch.stack(Ls, 0)  # [K,nNode, nNode]
        #print(Lap)
        Lap = Lap.transpose(-1,-2)
        x = torch.einsum('bcnl,knq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        return out 

class ST_BLOCK_4(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,K,Kt):
        super(ST_BLOCK_4,self).__init__()
        self.conv1=Conv2d(c_in, c_out, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        self.gcn=cheby_conv(c_out//2,c_out,K,1)
        self.conv2=Conv2d(c_out, c_out*2, kernel_size=(1, Kt),padding=(0,1),
                          stride=(1,1), bias=True)
        self.c_out=c_out
        self.conv_1=Conv2d(c_in, c_out, kernel_size=(1, 1),
                          stride=(1,1), bias=True)
        #self.conv_2=Conv2d(c_out//2, c_out, kernel_size=(1, 1),
          #                stride=(1,1), bias=True)

    def forward(self,x,supports):
        x_input1=self.conv_1(x)
        x1=self.conv1(x)
        filter1,gate1=torch.split(x1,[self.c_out//2,self.c_out//2],1)
        x1=(filter1)*torch.sigmoid(gate1)
        x2=self.gcn(x1,supports)
        x2=torch.relu(x2)
        #x_input2=self.conv_2(x2)
        x3=self.conv2(x2)
        filter2,gate2=torch.split(x3,[self.c_out,self.c_out],1)
        x=(filter2+x_input1)*torch.sigmoid(gate2)
        return x





class ST_BLOCK_5(nn.Module):
    def __init__(self,c_in,c_out,num_nodes,tem_size,K,Kt):
        super(ST_BLOCK_5,self).__init__()
        self.gcn_conv=gcn_conv_hop(c_out+c_in,c_out*4,K,1)
        self.c_out=c_out
        self.tem_size=tem_size
        
        
    def forward(self,x,supports):
        shape = x.shape
        h = Variable(torch.zeros((shape[0],self.c_out,shape[2]))).cuda()
        c = Variable(torch.zeros((shape[0],self.c_out,shape[2]))).cuda()
        out=[]
        
        for k in range(self.tem_size):
            input1=x[:,:,:,k]
            tem1=torch.cat((input1,h),1)
            fea1=self.gcn_conv(tem1,supports)
            i,j,f,o = torch.split(fea1, [self.c_out, self.c_out, self.c_out, self.c_out], 1)
            new_c=c*torch.sigmoid(f)+torch.sigmoid(i)*torch.tanh(j)
            new_h=torch.tanh(new_c)*(torch.sigmoid(o))
            c=new_c
            h=new_h
            out.append(new_h)
        x=torch.stack(out,-1)
        return x 