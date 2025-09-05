import os
import tiktoken
import torch
import numpy as np


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.uint32)
    ptt = torch.tensor(npt,dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self,batch,context,rank,world,file_path):
        self.B = batch
        self.T = context
        self.rank = rank
        self.world = world
        self.cur_pos = self.B*self.T*self.rank
        with open(file=file_path,mode="r",encoding="utf-8") as f:
            data = f.read()
        enc = tiktoken.get_encoding("gpt2")
        encoded_tokens = enc.encode(data,allowed_special="all")
        self.tokens = torch.tensor(data=encoded_tokens)
        if torch.distributed.get_rank() == 0:
            print(f"Total Tokens : {len(self.tokens)}")
    def len(self):
        return self.B,self.T        
    def next_batch(self):
        buff = self.tokens[self.cur_pos:self.cur_pos+(self.B * self.T+1)]
        x = buff[:-1].view(self.B,self.T)
        y = buff[1:].view(self.B,self.T)
        self.cur_pos += (self.B * self.T * self.world)
        if self.cur_pos + (self.B * self.T * self.world)+1 > len(self.tokens):
            self.cur_pos = self.B*self.T*self.rank
        return x,y

class ShardedDataLoaderLite:
    def __init__(self,batch,context,rank,world,dir):
        self.B = batch
        self.T = context
        self.rank = rank
        self.world = world
        self.cur_pos = self.B*self.T*self.rank
        shards = sorted(os.listdir(dir))
        self.shards = [os.path.join(dir,file) for file in shards]
        self.cur_shard = 0
        self.tokens = load_tokens(self.shards[self.cur_shard])
        if torch.distributed.get_rank() == 0:
            print(f"Total Tokens {len(self.tokens)} of shard {self.shards[self.cur_shard]}")
    def reset(self):
        self.cur_shard = 0
        self.tokens = load_tokens(self.shards[self.cur_shard])
        self.cur_pos = self.B*self.T*self.rank
    def len(self):
        return self.B,self.T        
    def next_batch(self):
        buff = self.tokens[self.cur_pos:self.cur_pos+(self.B * self.T+1)]
        idx = torch.randint(0,len(buff) - self.T,size=(self.B,))
        x = torch.stack([buff[i:i+self.T] for i in idx])
        y = torch.stack([buff[i+1:i+self.T+1] for i in idx])
        #x = buff[:-1].view(self.B,self.T)
        #y = buff[1:].view(self.B,self.T)
        self.cur_pos += (self.B * self.T * self.world)
        if self.cur_pos + (self.B * self.T * self.world)+1 > len(self.tokens):
            self.cur_shard = (self.cur_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.cur_shard])
            self.cur_pos = self.B*self.T*self.rank
        return x,y
        
        
        