# Generated using Chat-GPT

"""
Masking: choose ~15% of real, non-special tokens → 80% <mask>, 10% random, 10% keep.

Labels: original token IDs only at masked positions; -100 elsewhere.

Loss: model predicts those hidden tokens; gradients update the encoder to understand polymer “language.”

Outcome: a stronger text encoder you can fine-tune for property prediction.

Why this helps your downstream task:
The encoder learns chemistry context: rings, branches, functional groups, polymer markers, etc.
These embeddings become good starting points for your supervised training (Tg, FFV, Tc, density, Rg).
With limited labeled data (and missing targets), starting from a self-supervised encoder usually improves accuracy and stability.

The script saves:
A directory with the MLM-pretrained encoder (config.json, pytorch_model.bin, etc.).
A small ssl_meta.pt with token IDs and paths (handy for loading later).
"""



import os, sys, math, random, argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaConfig, RobertaForMaskedLM
import pandas as pd

# ---------- Utils ----------
def set_seed(seed=1337):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_tokenized_pt(path: Path):
    blob = torch.load(path)
    return blob["input_ids"], blob["attention_mask"]

# BERT-style masking: 80% [MASK], 10% random, 10% keep
def apply_mlm_mask(input_ids, attention_mask, special_ids, mask_id, pad_id, vocab_size, p=0.15):
    B, L = input_ids.shape
    labels = input_ids.clone()

    # maskable = real tokens AND not special tokens
    maskable = attention_mask.bool()
    for sid in special_ids:
        maskable &= (input_ids != sid)
    # sample positions
    probs = torch.rand_like(input_ids.float())
    mask_pos = (probs < p) & maskable

    # 80% -> [MASK]
    choice = torch.rand_like(input_ids.float())
    to_mask = mask_pos & (choice < 0.8)
    # 10% -> random token
    to_rand = mask_pos & (choice >= 0.8) & (choice < 0.9)
    # 10% -> keep original (mask_pos & choice>=0.9)

    masked = input_ids.clone()
    masked[to_mask] = mask_id

    rand_tokens = torch.randint(low=0, high=vocab_size, size=input_ids.shape, device=input_ids.device)
    masked[to_rand] = rand_tokens[to_rand]

    # labels: only compute loss on masked positions
    labels[~mask_pos] = -100
    return masked, labels

# ---------- Dataset ----------
class TokenizedMLMDataset(Dataset):
    def __init__(self, tokens_pt, split="train", val_frac=0.1, seed=1337):
        tokens =Path(tokens_pt).resolve()
        ids, am = load_tokenized_pt(tokens)
        N = ids.size(0)
        # simple split
        set_seed(seed)
        idx = torch.randperm(N)
        cut = int(N * (1 - val_frac))
        self.is_train = (split == "train")
        self.input_ids = ids[idx[:cut]] if self.is_train else ids[idx[cut:]]
        self.attn_mask = am[idx[:cut]] if self.is_train else am[idx[cut:]]
        self.N = self.input_ids.size(0)

    def __len__(self): return self.N
    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attn_mask[i],
        }

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    # paths
    ap.add_argument("--tp_dir", required=True, help="Path to TransPolymer repo (contains PolymerSmilesTokenization.py)")
    ap.add_argument("--vocab", required=True, help="Path to vocab.json")
    ap.add_argument("--merges", required=True, help="Path to merges.txt")
    ap.add_argument("--tokens_pt", required=True, help="Path to *.tokenized.pt you saved earlier")
    ap.add_argument("--out_dir", default="text_ssl_ckpt", help="Where to save the SSL model")
    # model/data
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--intermediate", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load tokenizer (TransPolymer class) just to get vocab size & ids ---
    TP_DIR = Path(args.tp_dir).resolve()
    VOCAB = Path(args.vocab).resolve()
    MERGES = Path(args.merges).resolve()
    sys.path.append(str(TP_DIR))
    from PolymerSmilesTokenization import PolymerSmilesTokenizer

    tok = PolymerSmilesTokenizer(
        vocab_file=str(VOCAB),
        merges_file=str(MERGES),
        bos_token="<s>", eos_token="</s>", sep_token="</s>",
        cls_token="<s>", unk_token="<unk>", pad_token="<pad>", mask_token="<mask>",
    )

    vocab_size = tok.vocab_size
    pad_id  = tok.convert_tokens_to_ids("<pad>")
    mask_id = tok.convert_tokens_to_ids("<mask>")
    bos_id  = tok.convert_tokens_to_ids("<s>")
    eos_id  = tok.convert_tokens_to_ids("</s>")
    unk_id  = tok.convert_tokens_to_ids("<unk>")
    special_ids = {pad_id, mask_id, bos_id, eos_id, unk_id}

    # --- build model ---
    cfg = RobertaConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        intermediate_size=args.intermediate,
        max_position_embeddings=max(args.max_len + 2, 514),
        type_vocab_size=1,
        pad_token_id=pad_id,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
    )
    model = RobertaForMaskedLM(cfg).to(device)

    # --- data ---
    train_ds = TokenizedMLMDataset(args.tokens_pt, split="train", val_frac=0.1, seed=args.seed)
    val_ds   = TokenizedMLMDataset(args.tokens_pt, split="val",   val_frac=0.1, seed=args.seed)

    def collate(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
        attn_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
        # apply MLM mask on the fly
        masked_ids, labels = apply_mlm_mask(
            input_ids, attn_mask, special_ids,
            mask_id=mask_id, pad_id=pad_id,
            vocab_size=vocab_size, p=0.15
        )
        return masked_ids, attn_mask, labels

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # --- optim ---
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # --- train ---
    best_val = float("inf")
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for masked_ids, attn_mask, labels in train_dl:
            masked_ids, attn_mask, labels = masked_ids.to(device), attn_mask.to(device), labels.to(device)
            out = model(input_ids=masked_ids, attention_mask=attn_mask, labels=labels)
            loss = out.loss
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tot += loss.item() * masked_ids.size(0); n += masked_ids.size(0)
        sched.step()
        train_loss = tot / max(1, n)

        # validation
        model.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for masked_ids, attn_mask, labels in val_dl:
                masked_ids, attn_mask, labels = masked_ids.to(device), attn_mask.to(device), labels.to(device)
                out = model(input_ids=masked_ids, attention_mask=attn_mask, labels=labels)
                vtot += out.loss.item() * masked_ids.size(0); vn += masked_ids.size(0)
        val_loss = vtot / max(1, vn)

        print(f"Epoch {epoch:02d} | train_MLM {train_loss:.4f} | val_MLM {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_dir = Path(args.out_dir)
            model.save_pretrained(save_dir.as_posix())
            # also save a tiny meta so we know special ids later
            meta = {
                "vocab": args.vocab,
                "merges": args.merges,
                "pad_id": pad_id, "mask_id": mask_id,
                "bos_id": bos_id, "eos_id": eos_id, "unk_id": unk_id,
                "max_len": args.max_len,
            }
            torch.save(meta, save_dir / "ssl_meta.pt")
            print(f"  -> saved best to {save_dir} (val {best_val:.4f})")

if __name__ == "__main__":
    main()
