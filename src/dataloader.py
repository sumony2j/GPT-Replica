import tiktoken
import torch
import numpy as np
from torch.utils.data import Dataset

def load_tokens(filename):
    npt = np.load(file=filename)
    npt = npt.astype(np.uint32)
    ptt = torch.tensor(npt,dtype=torch.long)
    return ptt

class TrainingDataset(Dataset):
    def __init__(self,Block_Size,file_path):
        super().__init__()
        self.block_size = Block_Size
        with open(file=file_path,mode="r",encoding="utf-8") as f:
            data = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(data)
        self.tokens = torch.tensor(tokens,dtype=torch.long)
    def __len__(self):
        return len(self.tokens) - self.block_size - 1
    def __getitem__(self, index):
        data_chunk = self.tokens[index:(index+self.block_size+1)] 
        x = data_chunk[:-1]
        y = data_chunk[1:]
        return x,y

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
        
        
        