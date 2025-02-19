import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import random

def Dec(content):
    new = [int(i.strip("\n")) for i in content]
    return new


def build_dataset(config):
   
    tokenizer = lambda x: x.split(' ')  
   
    def load_dataset(path):
        contents = []
        
        # Example IoT-23 KBS result 
        pos = [33, 35, 32, 27, 24, 20, 25, 23, 22, 26, 4, 5, 11, 3, 9, 36, 10, 37, 34, 41, 21, 8, 42, 49, 29, 30, 31, 48, 44, 17, 45, 46, 2, 47, 28, 43, 40, 6, 13, 0, 1, 15, 12, 14, 19, 39, 7, 16, 18, 38]
        # Randomly select 20 bytes
        # pos = random.sample(range(0, 50), config.byte_len_withKBS)
        pos = sorted(pos[:config.byte_len_withKBS])
        
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')

                token = tokenizer(content)
      
                token = [token[i] for i in pos] 
                contents.append((Dec(token),pos,int(label)))
                
        return contents  

    train = load_dataset(config.train_path)
    dev = load_dataset(config.dev_path)
    test = load_dataset(config.test_path)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        #x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        #pos = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        #y = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        pos = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        
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
    
