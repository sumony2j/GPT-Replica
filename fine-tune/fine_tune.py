from transformers import Trainer,TrainingArguments
from src.model import SeedGPT
import torch

weights_path = ""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SeedGPT.from_pretrained(model_type="Tiny")
weights = torch.load(weights_path,map_location="cuda")
mod_weights = {k.replace("module.", ""): v for k, v in weights.items()}
mod_weights = {k.replace("module._orig_mod.", ""): v for k, v in weights.items()}
model.load_state_dict(mod_weights)
model.to(device=device)

Args = TrainingArguments()

train = Trainer(model=model,args=Args)



