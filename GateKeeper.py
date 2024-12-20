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
        self.train_path = dataset + '/data/train.txt'                                # Training set
        self.dev_path = dataset + '/data/dev.txt'                                    # Validation set  
        self.test_path = dataset + '/data/test.txt'                                  # Test set
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # Class list
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # Model checkpoint

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # Device
        self.learning_rate = 0.00197
        self.require_improvement = 300000                                            # Early stopping if no improvement after this many batches
        self.num_classes = len(self.class_list)                                      # Number of classes
        self.num_epochs = 10                                                         # Number of epochs
        self.batch_size = 163                                                        # Batch size
        
        self.byte_len_withKBS = 20                                                      # Maximum byte length
        self.d_dim = 39                                                             # Embedding dimension
        self.dropout = 0.1                                                          # Dropout rate
        self.hidden_size = 81                                                       # Hidden layer size


class Model(nn.Module):
    """GateKeeper model for traffic classification"""
    def __init__(self, config):
        super().__init__()
        
        # Embedding layers
        self.byte_embedding = nn.Embedding(256, config.d_dim)
    
        # Normalization and regularization
        self.layer_norm = nn.LayerNorm(config.d_dim)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=2),
            nn.Linear(config.byte_len_withKBS * config.d_dim, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.num_classes)
        )
        
    def forward(self, x, pos):
        # Embedding
        embedded = self.byte_embedding(x)
        out = torch.fft.fft(torch.fft.fft(embedded, dim=-1), dim=-2).real
        residual = embedded
     
        # Residual connection and layer norm
        out = self.layer_norm(out + residual)
        
        # Classification
        out = self.classifier(out)
        
        
        return out,0
