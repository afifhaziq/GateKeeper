import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

class Config(object):
    """Configuration parameters"""

    def __init__(self, dataset):
        self.model_name = 'Base'
        self.train_path = dataset + '\\data\\train.txt'                                 # Training set
        self.dev_path = dataset + '\\data\\dev.txt'                                    # Validation set  
        self.test_path = dataset + '\\data\\test.txt'                                   # Test set
        self.class_list = [x.strip() for x in open(
            dataset + '\\data\\class.txt', encoding='utf-8').readlines()]              # Class list  C:\Users\afif\Documents\Master\Code\benchmark_ntc\GateKeeper\dataset\ISCXVPN2016\data\data\class.txt
        self.save_path = dataset + '\\saved_dict\\' + self.model_name + '.ckpt'        # Model checkpoint

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # Device
        self.learning_rate = 0.00197
        self.require_improvement = 300000                                            # Early stopping if no improvement after this many batches
        self.num_classes = len(self.class_list)                             #len(self.class_list)                                      # Number of classes
        self.num_epochs = 10                                                         # Number of epochs
        self.batch_size = 163                                                        # Batch size
        
        self.max_byte_len = 50                                                      # Maximum byte length
        self.d_dim = 39                                                             # Embedding dimension
        self.dropout = 0.1                                                          # Dropout rate
        self.hidden_size = 81                                                       # Hidden layer size


class SelfAttention(nn.Module):
    """Self attention module"""
    def __init__(self, d_dim=256, dropout=0.3):
        super().__init__()
        self.dim = d_dim
        self.query = nn.Linear(d_dim, d_dim)
        self.key = nn.Linear(d_dim, d_dim) 
        self.value = nn.Linear(d_dim, d_dim)
        self.out = nn.Linear(d_dim, d_dim)

    def scaled_dot_product(self, query, key, value):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.dim)
        attention_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, value), attention_weights

    def forward(self, x, y, z):
        query = self.query(x)
        key = self.key(y)
        value = self.value(z)
        
        context, attention = self.scaled_dot_product(query, key, value)
        output = self.out(context)
        
        return output, torch.mean(attention, dim=-2)


class Model(nn.Module):
    """GateKeeper model for traffic classification"""
    def __init__(self, config):
        super().__init__()
        
        # Embedding layers
        self.byte_embedding = nn.Embedding(256, config.d_dim)
        self.pos_embedding = nn.Embedding(50, config.d_dim)

        # Attention layer
        self.attention = SelfAttention(config.d_dim, config.dropout)
        
        # Normalization and regularization
        self.layer_norm = nn.LayerNorm(config.d_dim)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=2),
            nn.Linear(config.max_byte_len * config.d_dim, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.num_classes)
        )
        
    def forward(self, x, pos):
        # Embedding
        embedded = self.byte_embedding(x)
        residual = embedded
        
        # Self attention
        attended, attention_scores = self.attention(embedded, embedded, embedded)
        
        # Residual connection and layer norm
        output = self.layer_norm(attended + residual)
        
        # Classification
        out = self.classifier(output)
    
        return out, attention_scores