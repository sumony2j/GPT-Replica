import os
import math
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
        
@dataclass
class GPTConfig:
    block_size : int = 1024
    vocab_size : int = 50257
    n_layer : int = 6
    n_head : int =  12
    n_embd : int = 768

class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd,4*config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4*config.n_embd,config.n_embd)
        self.c_proj.WEIGHT_FLAG = 1
    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
        
class Attention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = self.n_embd//self.n_head
        self.c_attn = nn.Linear(self.n_embd,3*self.n_embd)
        self.c_proj = nn.Linear(self.n_embd,self.n_embd)
        self.c_proj.WEIGHT_FLAG = 1
        self.flash = hasattr(F,"scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,
                                                                                                     config.block_size,config.block_size))
    def forward(self,x):
        B,T,C = x.size() # (B,T,C)
        qkv = self.c_attn(x) # (B,T,3C)
        q,k,v = qkv.split(self.n_embd,dim=2) # (B,T,C)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # (B,T,NH,HS) --> (B,NH,T,HS)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # (B,T,NH,HS) --> (B,NH,T,HS)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2) # (B,T,NH,HS) --> (B,NH,T,HS)
        if self.flash:
            out = F.scaled_dot_product_attention(q,k,v,is_causal=True)
        else:
            wei = (q@k.transpose(-2,-1))/(math.sqrt(k.size(-1))) # (B,NH,T,HS) * (B,NH,HS,T) --> (B,NH,T,T)
            wei = wei.masked_fill(self.bias[:,:,:T,:T]==0,float("-inf"))
            wei = F.softmax(wei,dim=-1)
            out = wei @ v # (B,NH,T,T) * (B,T,NH,HS) --> (B,NH,T,HS)
        out = out.transpose(1,2).contiguous().view(B,T,C) # (B,T,C)
        out = self.c_proj(out) 
        return out

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x+ self.mlp(self.ln_2(x))
        return x       
         
class SeedGPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.n_embd),
            wpe = nn.Embedding(config.block_size,config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
    def forward(self,x,targets=None):
        B,T = x.size()
        assert T <= self.config.block_size, f"Maxiumum Sequence len reached"
        tok_emb = self.transformer.wte(x)
        pos = torch.arange(0,T,dtype=torch.long,device=x.device)
        pos_emb = self.transformer.wpe(pos)
        data = tok_emb + pos_emb
        for b in self.transformer.h:
            data = b(data)
        data = self.transformer.ln_f(data)
        logits = self.lm_head(data)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))
        return logits,loss
    def generate(self,ix,max_token,temp=1.0,top_k=50):
        for _ in range(max_token):
            x = ix[:,-self.config.block_size:]
            logits,_ = self(x)
            logit = logits[:,-1,:]
            logit = logit/temp
            probs = F.softmax(logit,dim=-1)
            topk_probs,idx = torch.topk(probs,k=top_k,dim=-1)
            next_ix = torch.multinomial(topk_probs,num_samples=1)
            next = torch.gather(idx,-1,next_ix)
            ix = torch.cat((x,next),dim=-1)
        return ix
    def _init_weights(self,module):
        if isinstance(module,nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module,"WEIGHT_FLAG"):
                std *= math.sqrt(2*self.config.n_layer)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    def configure_optimizer(self,weight_decay,lr):
        param_dict = {name:param for name,param in self.named_parameters()}
        param_dict = {pn:p for pn,p in param_dict.items() if p.requires_grad}
        decay_param = [p for pn,p in param_dict.items() if p.dim()>=2]
        nondecay_param = [p for pn,p in param_dict.items() if p.dim()<2]
        optim_param = [
            {"params" : decay_param, "weight_decay" : weight_decay},
            {"params" : nondecay_param, "weight_decay" : 0.0}
        ]
        opt = torch.optim.AdamW(optim_param,lr=lr,betas=(0.9,0.95),eps=1e-8,fused=True)
        return opt
    @classmethod
    def from_pretrained(cls,model_type,weight_path=None):
        if model_type not in ["Lite","Tiny","Mini"]:
            return "Provide a valid model Lite/Tiny/Mini"
        if model_type == "Lite":
            config_args = {"n_layer" : 6,"n_head" : 6,"n_embd" : 768}
        elif model_type == "Tiny":
            config_args = {"n_layer" : 12,"n_head" : 12,"n_embd" : 768}
        elif model_type == "Mini":
            config_args = {"n_layer" : 24,"n_head" : 16,"n_embd" : 1024}
        config_args["block_size"] = 1024
        config_args["vocab_size"] = 50257
        
        config = GPTConfig(**config_args)
        model = cls(config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if weight_path is not None:
            weights = torch.load(weight_path, map_location=torch.device(device=device))
            weights_ = {k.replace("module._orig_mod.", ""): v for k, v in weights.items()}
            model_weights = {k.replace("module.", ""): v for k, v in weights_.items()}
            model.load_state_dict(model_weights)
            return model
        model_weights = model.state_dict()
        model_weights_keys = model_weights.keys()
        model_weights_keys = [k for k in model_weights_keys if not k.endswith(".attn.bias")]
        
        print(f"Loading weights of {model_type} \n") 

        if model_type == "Lite":
            gpt_hf_weights = torch.load("./weights/Lite.bin", map_location=torch.device(device=device))
        elif model_type == "Tiny":
            gpt_hf_weights = torch.load("./weights/Tiny.bin", map_location=torch.device(device=device))
        elif model_type == "Mini":
            gpt_hf_weights = torch.load("./weights/Mini.bin", map_location=torch.device(device=device))
    
        gpt_hf_weights_keys = gpt_hf_weights.keys()
        gpt_hf_weights_keys = [k for k in gpt_hf_weights_keys if not k.endswith(".attn.bias")]
        gpt_hf_weights_keys = [k for k in gpt_hf_weights_keys if not k.endswith(".attn.masked_bias")]
        
        
        assert len(model_weights_keys) == len(gpt_hf_weights_keys), f"Mismatch Keys {len(model_weights_keys)} != {len(gpt_hf_weights_keys)}"
        
        transposed_modules = ["attn.c_attn.weight","attn.c_proj.weight","mlp.c_fc.weight","mlp.c_proj.weight"]
        
        for k in gpt_hf_weights_keys:
            if any(k.endswith(w) for w in transposed_modules):
                assert gpt_hf_weights[k].shape[::-1] == model_weights[k].shape
                with torch.no_grad():
                    model_weights[k].copy_(gpt_hf_weights[k].t())
            else:
                assert gpt_hf_weights[k].shape == model_weights[k].shape
                with torch.no_grad():
                    model_weights[k].copy_(gpt_hf_weights[k])
        return model

if __name__=="__main__":
    model = SeedGPT.from_pretrained(model_type="Lite")
    print(model)