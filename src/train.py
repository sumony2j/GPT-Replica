import os
import time
import math
import torch
from dataloader import ShardedDataLoaderLite
from torch.distributed import get_rank,all_reduce,ReduceOp
from torch.optim.lr_scheduler import LambdaLR

def get_lr(opt,max_step,warmup_step=1000,max_lr=6e-4):
    def lr_lambda(iter):
        min_lr = max_lr*0.1
        if iter < warmup_step:
            return max_lr * (iter+1)/warmup_step
        elif iter > max_step:
            return min_lr
        else:
            decay_ration = (iter - warmup_step) / (max_step - warmup_step)
            coff = 0.5 * (1.0 + math.cos(math.pi * decay_ration))
            return min_lr + coff * (max_lr - min_lr)
    return LambdaLR(optimizer=opt,lr_lambda=lr_lambda)
 
def train(model, steps, opt, train_data:ShardedDataLoaderLite, val_data:ShardedDataLoaderLite, lr_rate, device="cpu"):
    
    scheduler = get_lr(opt=opt,max_step=200000,warmup_step=10000)
    
    ## Validation Eval
    for i in range(steps):
        t0 = time.time()
        if i % 100 == 0:
            model.eval()
            val_data.reset()
            with torch.no_grad():
                val_step = 20
                total_loss = 0
                for _ in range(val_step):
                    val_x,val_y = val_data.next_batch()
                    val_x,val_y = val_x.to(device), val_y.to(device)
                    with torch.autocast(device_type=device,dtype=torch.bfloat16):
                        _,val_loss = model(val_x,val_y)
                    total_loss += val_loss.detach()
                validation_loss = total_loss/val_step
            ddp = int(os.environ.get("RANK",-1)) != -1
            if ddp:
                all_reduce(validation_loss,op=ReduceOp.AVG)
            if get_rank() == 0:
                print(f"validation loss : {validation_loss.item():.4f}")   

        model.train()
        opt.zero_grad()
        
        x,y = train_data.next_batch()
        x,y = x.to(device), y.to(device)       
        with torch.autocast(device_type=device,dtype=torch.bfloat16):
            _,loss = model(x,y)
        
        loss.backward()
        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        
        opt.step()
        scheduler.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1-t0)*1000
        B,T = train_data.len()
        tokens_per_sec = (B*T)/(t1-t0)
        current_lr = scheduler.get_last_lr()[0]
        if get_rank() == 0:
            print(f"step : {i}, train loss : {loss.item():.2f}, learning rate : {current_lr:.2e}, norm : {norm.item():.2f}, time : {dt:.2f}ms, tok/sec : {tokens_per_sec:.2f}")
        