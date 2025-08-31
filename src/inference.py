import torch
import tiktoken
from model import SeedGPT

weights = torch.load("",map_location="cuda")
new_weights = {k.replace("module.", ""): v for k, v in weights.items()}

model = SeedGPT.from_pretrained()
model.load_state_dict(new_weights)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

encoder = tiktoken.get_encoding("gpt2")
encoded_txt = encoder.encode("Love")
tokens = torch.tensor(encoded_txt,dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(5,1)
x = tokens.to(device)
output = model.generate(x,max_token=30)
for i in range(5):
    out_tokens = output[i].tolist()
    decoded = encoder.decode(out_tokens)
    print(f">> {decoded}")