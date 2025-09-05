import tiktoken
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import re
import unicodedata
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException

def clean_text(text, min_length=50, lang='en'):
    """
    Clean a single text string from web dataset.

    Args:
        text (str): Raw input text.
        min_length (int): Minimum character length to keep text.
        lang (str): Language to filter (e.g., 'en' for English).

    Returns:
        str or None: Cleaned text, or None if filtered out.
    """
    if not isinstance(text, str):
        return None

    # 1. Strip HTML tags
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")

    # 2. Normalize Unicode
    text = unicodedata.normalize("NFC", text)

    # 3. Remove zero-width and control characters
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    text = re.sub(r'[\r\n\t]+', ' ', text)

    # 4. Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # 5. Minimum length filter
    if len(text) < min_length:
        return None

    # 6. Language detection
    try:
        detected_lang = detect(text)
        if detected_lang != lang:
            return None
    except LangDetectException:
        return None

    return text

encoder = tiktoken.get_encoding("gpt2")
end_tok = encoder._special_tokens["<|endoftext|>"]

def tokenizer(data):
    tokens = encoder.encode_ordinary(text=data)
    tokens.append(end_tok)
    return np.array(tokens,dtype=np.uint16)

def write_file(filename,tokens):
    buffer = np.array(tokens,dtype=np.uint32)
    np.save(filename,buffer)
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