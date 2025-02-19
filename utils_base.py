import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import random

def Dec(content):
    new = [float(i.strip("\n")) for i in content]
    return new


def build_dataset(config):
   
    tokenizer = lambda x: x.split(' ')                                         # Split the sentence by space
    def load_dataset(path):
        contents = []
        
        pos = [i for i in range(config.max_byte_len)]                       # Get the position of the word (byte)
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()                                        # Remove the space at the beginning and end of the line                                       
                if not lin:                                               # If the line is empty, skip to next iterations
                    continue
                content, label = lin.split('\t')
                token = tokenizer(content)
                token = [token[i] for i in pos]                            # assign byte encoding to token based on position
                contents.append((Dec(token),pos,int(float(label))))     # Append the token, position and label to the contents list
                
        return contents  # [([...], 0), ([...], 1), ...]

    train = load_dataset(config.train_path)
    dev = load_dataset(config.dev_path)
    test = load_dataset(config.test_path)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        #x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        #pos = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        #y = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)             # Convert the token to tensor and move it to the device
        pos = torch.LongTensor([_[1] for _ in datas]).to(self.device)           # Convert the position to tensor and move it to the device
        y = torch.LongTensor([_[2] for _ in datas]).to(self.device)             # Convert the label to tensor and move it to the device
        
        #print(x.shape)
        #x = torch.reshape(x,(x.shape[0],50))
        
        
        return x,pos,y
        
        
    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    
    return iter


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

if __name__ == '__main__':
    test_list = ["ff" for i in range(50)]
    
