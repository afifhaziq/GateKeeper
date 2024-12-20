# -*- coding: utf-8 -*-
# @Author: xiegr
# @Date:   2020-08-30 15:58:51
# @Last Modified by:   xiegr
# @Last Modified time: 2020-09-18 14:22:48
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

class Config(object):

    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'Fnet'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单          -
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'      # 模型训练结果

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.learning_rate = 0.0039
        # 若超过1000batch效果还没提升，则提前结束训练
        self.require_improvement = 300000
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 10                                          # epoch数
        self.batch_size = 163                                          # mini-batch大小
        
        self.max_byte_len = 50
        self.d_dim = 39
        self.dropout = 0.1
        self.hidden_size = 81

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        
        self.byteembedding = nn.Embedding(num_embeddings=256, embedding_dim=config.d_dim)
        #self.posembedding =  nn.Embedding(num_embeddings=50, embedding_dim=config.d_dim)
        
        self.flat = nn.Flatten(start_dim=1, end_dim=2)
        self.fc1 = nn.Linear(in_features=config.max_byte_len*config.d_dim ,out_features=config.hidden_size)
        #self.bn = nn.BatchNorm1d(config.hidden_size,affine = True)
        self.layer_norm = nn.LayerNorm(config.d_dim)
        self.fc2 = nn.Linear(in_features=config.hidden_size,
                             out_features=config.num_classes)
        #self.dropout = nn.Dropout(p=config.dropout)
        
    def forward(self, x, pos):
        out = self.byteembedding(x)
        y = out
        #out = out + self.posembedding(pos)
        # 2次傅立叶
        out = torch.fft.fft(torch.fft.fft(out, dim=-1), dim=-2).real
        # 2维傅立叶
        #out =  torch.fft.fft2(out).real
        score = 0
        #out,score = self.attention(out,out,out)
        #out = out + y 
        out = self.layer_norm(out + y) 
        #out = out + y
       

        out = self.flat(out)
        out = self.fc1(out)
        out = F.gelu(out)
        out = self.fc2(out)
    
        return out,score


