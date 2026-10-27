"""
Tokenize TinyStories for training.
- Trains a BPE tokenizer on the first ~50 MB of training text.
- Encodes the full train and val sets.
- Saves as uint16 .npy files (vocab fits in 65535).
"""
import os
import sys
import numpy as np

# Make sure cs336_basics is importable from this directory
sys.path.insert(0, os.path.dirname(__file__))
from cs336_basics.tokenizer import BPETokenizer

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
TRAIN_TXT = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-train.txt")
VAL_TXT   = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-valid.txt")
TRAIN_NPY = os.path.join(DATA_DIR, "train_tokens.npy")
VAL_NPY   = os.path.join(DATA_DIR, "val_tokens.npy")

VOCAB_SIZE    = 10_000
SAMPLE_BYTES  = 50 * 1024 * 1024   # 50 MB for BPE training
SPECIAL       = ["<|endoftext|>"]

print(f"Reading {SAMPLE_BYTES // 1_000_000} MB sample for BPE training...")
with open(TRAIN_TXT, "r", encoding="utf-8") as f:
    sample = f.read(SAMPLE_BYTES)

print(f"Training BPE (vocab_size={VOCAB_SIZE})...")
vocab, merges = BPETokenizer.train(sample, VOCAB_SIZE)
tokenizer = BPETokenizer(vocab, merges, special_tokens=SPECIAL)
print(f"Vocab size: {len(tokenizer.vocab)}  (with special tokens)")

# ── 2. Tokenize helper ────────────────────────────────────────────────────────
def tokenize_file(path, npy_path, label):
    print(f"Tokenizing {label}...")
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            ids.extend(tokenizer.encode(line))
            if (i + 1) % 100_000 == 0:
                print(f"  {i+1:,} lines  {len(ids):,} tokens", end="\r")
    print(f"  {len(ids):,} tokens total")
    arr = np.array(ids, dtype=np.uint16)
    np.save(npy_path, arr)
    print(f"  Saved → {npy_path}")

# ── 3. Encode both splits ─────────────────────────────────────────────────────
tokenize_file(TRAIN_TXT, TRAIN_NPY, "train")
tokenize_file(VAL_TXT,   VAL_NPY,   "val")
print("Done.")
