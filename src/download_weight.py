from transformers import GPT2LMHeadModel

model_name = "gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_name)

save_dir = "./weights"

model.save_pretrained(save_dir,safe_serialization=False)       # saves weights + config

print(f"Model and tokenizer saved to {save_dir}")
