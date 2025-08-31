import time
import torch
from dataloader import DataLoaderLite
from torch.distributed import get_rank

def train(model, epochs, data:DataLoaderLite, device="cpu"):
    lr_rate = 5e-3
    opt = torch.optim.AdamW(model.parameters(),lr=lr_rate,betas=(0.9,0.95))
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=opt,
        T_max=500,
        eta_min=lr_rate*0.1
    )
    
    for i in range(epochs):
        t0 = time.time()
        x,y = data.next_batch()
        x,y = x.to(device), y.to(device)
        
        opt.zero_grad()
        with torch.autocast(device_type=device,dtype=torch.bfloat16):
            _,loss = model(x,y)
        
        loss.backward()
        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        
        opt.step()
        scheduler.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1-t0)*1000
        B,T = data.len()
        tokens_per_sec = (B*T)/(t1-t0)
        if get_rank() == 0:
            print(f"step : {i}, loss : {loss.item():.2f}, norm : {norm:.2f}, time : {dt:.2f}ms, tok/sec : {tokens_per_sec:.2f}")
        