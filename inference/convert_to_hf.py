import torch
from src.model import SeedGPT
from transformers import GPT2Config,AutoTokenizer,AutoModelForCausalLM

SAVE_DIR = "Tiny"
pt_weights = "./weights/AI_StoryTeller_Final.pt"

config = GPT2Config(vocab_size=50257,n_embd=768,n_head=12,n_layer=12,block_size=1024)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

model = SeedGPT.from_pretrained("Tiny")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight = torch.load(pt_weights,map_location=device)

modified_weights = {k.replace("module._orig_mod.", ""): v for k, v in weight.items()}
model.load_state_dict(modified_weights)

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"Model converted and saved at {SAVE_DIR}")