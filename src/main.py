from model import SeedGPT,GPTConfig
import tiktoken
import torch
import os
from dataloader import DataLoaderLite,TrainingDataset
from train import  train
from train_other import train_other
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group,destroy_process_group,get_rank,get_world_size,is_initialized


def setup_ddp():
    ddp = int(os.environ.get("RANK",-1)) != -1
    if ddp:
        assert torch.cuda.is_available(), "DDP works with multipe GPU only"
        init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(device=local_rank)
        device = torch.device(f"cuda:{local_rank}")
        world_rank = get_world_size()
        rank = get_rank()
    else:
        local_rank = 0
        world_rank = 1
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using Device : {device}")
    if local_rank == 0:
        print("Setup Done \n")
    return ddp,device,local_rank,world_rank,rank

def cleanup_ddp():
    if is_initialized():
        destroy_process_group()

if __name__ == "__main__":
    ddp,device,local_rank,world_rank,rank = setup_ddp()
    Dataset = TrainingDataset(Block_Size=256,file_path="./shakespeare.txt")
    Sampler = DistributedSampler(Dataset,num_replicas=world_rank,
                                 rank=local_rank,shuffle=True)
    Data = DataLoader(dataset=Dataset,sampler=Sampler,batch_size=256)
    #Data = DataLoaderLite(batch=32,context=64,rank=local_rank,world=world_rank,file_path="./shakespeare.txt")
    model = SeedGPT.from_pretrained()
    #model = SeedGPT(GPTConfig(vocab_size=50304,block_size=1024,
    #                          n_embd=768,n_head=12,n_layer=12))
    torch.manual_seed(42)   
    
    model.to(device=device)
    model = torch.compile(model=model,backend="inductor", mode="max-autotune")
    if ddp:
        model = DDP(model,device_ids=[local_rank],output_device=local_rank)
    #train(model=model,epochs=10,data=Data)
    train_other(model=model,epochs=10,data=Data)
    cleanup_ddp()
    
    torch.save(model.state_dict(),"SeedGPT.pt")
    """
    encoder = tiktoken.get_encoding("gpt2")
    encoded_txt = encoder.encode("The little boy")
    tokens = torch.tensor(encoded_txt,dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(5,1)
    x = tokens.to(device)
    output = model.generate(x,max_token=30)
    for i in range(5):
        out_tokens = output[i].tolist()
        decoded = encoder.decode(out_tokens)
        print(f">> {decoded}")
    """