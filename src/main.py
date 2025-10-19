from src.model import SeedGPT,GPTConfig
import torch
import os
from src.dataloader import ShardedDataLoaderLite
from src.train import  train
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
    train_data = ShardedDataLoaderLite(batch=128,context=256,rank=local_rank,world=world_rank,dir="/WSpace/dataset/fineweb-edu-dedup-10b/fineweb-edu-10b/train")
    val_data = ShardedDataLoaderLite(batch=128,context=256,rank=local_rank,world=world_rank,dir="/WSpace/dataset/fineweb-edu-dedup-10b/fineweb-edu-10b/val")
    model = SeedGPT.from_pretrained(model_type="Tiny")
    #model = SeedGPT(GPTConfig(vocab_size=50304,block_size=1024,n_embd=768,n_head=12,n_layer=12))
    torch.manual_seed(42)   
    
    torch.set_float32_matmul_precision("high")
    model.to(device=device)
    opt = model.configure_optimizer(weight_decay=0.1,lr=6e-4)
    model = torch.compile(model=model,backend="inductor", mode="max-autotune")
    if ddp:
        model = DDP(model,device_ids=[local_rank],output_device=local_rank)
    
    train(model=model,steps=10,opt=opt,lr_rate=1e-4,train_data=train_data,
          val_data=val_data,device=device.type)
    cleanup_ddp()
    torch.save(model.state_dict(),"SeedGPT.pt")
    