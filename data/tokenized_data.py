import tiktoken
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp


encoder = tiktoken.get_encoding("gpt2")
end_tok = encoder._special_tokens["<|endoftext|>"]

def tokenizer(data):
    tokens = encoder.encode_ordinary(text=data)
    tokens.append(end_tok)
    return np.array(tokens,dtype=np.uint16)

def write_file(filename,tokens):
    buffer = np.array(tokens,dtype=np.uint32)
    with open(file=filename,mode="wb") as f:
        buffer.tofile(f)
    print(f"[INFO] {len(tokens)} tokens have been saved in {filename}\n")

def load_text_from_parquet(DIR):
    for file in os.listdir(DIR):
        filename = f"{DIR}/{file}"
        if filename.endswith(".parquet"):
            print(f"Processing file : {filename}")
            df = pd.read_parquet(filename,columns=["text"])
            for txt in df["text"]:
                yield txt

DATASET_DIR = "/WSpace/dataset/fineweb-edu-dedup-10b"
LOCAL_DIR = "fineweb-edu-10b"
BASE_NAME = "fineweb"
num_shard = 0
count = 0
shard_size = 2000000
all_tokens = np.empty(shape=(shard_size,),dtype=np.uint16)
os.makedirs(LOCAL_DIR,exist_ok=True)

dataset = load_text_from_parquet(DATASET_DIR)
nprocs = max(1,os.cpu_count()-5)
print(f"Using {nprocs} processes for tokenization...")

with mp.Pool(nprocs) as pool:
    progress_bar = None
    for tokens in pool.imap(tokenizer,dataset,chunksize=16):
        if count + len(tokens) < shard_size:
            all_tokens[count:count+len(tokens)] = tokens
            count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size,unit="token",
                                    desc=f"Shard {num_shard}")
            progress_bar.update(len(tokens))
        else:
            extra = shard_size - count
            all_tokens[count:count+extra] = tokens[:extra]
            
            # Save
            filename = os.path.join(LOCAL_DIR,f"{BASE_NAME}_train_{num_shard:06d}.bin")
            write_file(filename=filename,tokens=all_tokens.tolist())
            
            # Next Shard
            num_shard += 1
            progress_bar = None
            leftover = tokens[extra:]
            all_tokens[0:len(leftover)] = leftover
            count = len(leftover)
    if count > 0:
        filename = os.path.join(LOCAL_DIR,f"{BASE_NAME}_train_{num_shard:06d}.bin")
        write_file(filename=filename,tokens=all_tokens[:count].tolist())

print(f"✅ Finished tokenizing FineWeb 10B. Total shards created: {num_shard + 1}")