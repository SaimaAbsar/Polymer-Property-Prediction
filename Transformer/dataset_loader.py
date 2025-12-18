# reg_dataset.py
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DEFAULT_TARGETS = ["Tg","FFV","Tc","Density","Rg"]

class TokenizedDataset(Dataset):
    def __init__(self, csv_path, tokens_pt, targets=None, split="train", val_frac=0.15, seed=1337):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        self.targets = targets or DEFAULT_TARGETS

        blob = torch.load(tokens_pt)
        self.input_ids = blob["input_ids"].long()        # [N, L]
        self.attn_mask = blob["attention_mask"].long()   # [N, L]
        self.N = self.input_ids.size(0)

        # Y + mask (missing -> NaN -> mask False)
        Y, M = [], []
        for name in self.targets:
            if name in self.df.columns:
                col = self.df[name].to_numpy()
            else:
                col = np.full((self.N,), np.nan, dtype=float)
            mask = ~np.isnan(col)
            vals = np.nan_to_num(col, nan=0.0)
            Y.append(vals); M.append(mask)
        self.y_all = torch.tensor(np.stack(Y, axis=1), dtype=torch.float)   # [N, T]
        self.m_all = torch.tensor(np.stack(M, axis=1), dtype=torch.bool)    # [N, T]

        # split
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(self.N, generator=g)
        cut = int(self.N * (1 - val_frac))
        self.sel = idx[:cut] if split == "train" else idx[cut:]

    def __len__(self): return self.sel.numel()

    def __getitem__(self, i):
        idx = self.sel[i].item()
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attn_mask[idx],
            "y": self.y_all[idx],
            "mask": self.m_all[idx],
        }

def collate(batch):
    ids  = torch.stack([b["input_ids"] for b in batch], 0)
    am   = torch.stack([b["attention_mask"] for b in batch], 0)
    y    = torch.stack([b["y"] for b in batch], 0)
    msk  = torch.stack([b["mask"] for b in batch], 0)
    return {"input_ids": ids, "attention_mask": am, "y": y, "mask": msk}
