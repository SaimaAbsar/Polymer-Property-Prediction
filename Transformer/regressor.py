# Aug 26, 2025
"""
The main regression model for predicting polymer properties from SMILES strings.
It uses a transformer-based architecture to encode the input SMILES strings and predict the target properties.
The encoded representations capture the structural and chemical information of the polymers, enabling accurate predictions.

The encoded embeddings are obtained using the pretrained TransPolymer, which is saved as the train.tokenized.pt. 
These embeddings are used to pretrain an MLM encoder which is saved as text_ssl_ckpt.
The following supervised regressor loads the pretrained encoder and adds a regression head on top, in order to predict the target properties.

"""

import argparse, json
from pathlib import Path
import sched
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaForMaskedLM, RobertaModel, RobertaConfig

from dataset_loader import TokenizedDataset, collate, DEFAULT_TARGETS

def masked_l1(pred, target, mask):
    if not mask.any():
        return torch.tensor(0.0, device=pred.device)
    return (pred[mask] - target[mask]).abs().mean()

class Regressor(nn.Module):
    def __init__(self, encoder: RobertaModel, n_targets: int, dropout=0.1, pool="cls"):
        super().__init__()
        self.encoder = encoder
        self.pool = pool  # "cls" or "mean"
        H = encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(H, n_targets)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state  # [B, L, H]
        if self.pool == "cls":
            pooled = last[:, 0, :]  # <s> token
        else:
            m = attention_mask.unsqueeze(-1)  # [B, L, 1]
            pooled = (last * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(self.drop(pooled))  # [B, T]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_data", required=True, help="CSV with targets + smiles")
    ap.add_argument("--tokens_pt", required=True, help="*.tokenized.pt (input_ids, attention_mask)")
    ap.add_argument("--ssl_dir", required=True, help="Folder saved by ssl_text_mlm.py (contains config.json, pytorch_model.bin)")
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    ap.add_argument("--out_dir", default="regressor_ckpt")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--pool", choices=["cls","mean"], default="cls")
    ap.add_argument("--freeze_encoder_epochs", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    # --- datasets/loaders ---
    train_ds = TokenizedDataset(args.input_data, args.tokens_pt, targets, split="train", val_frac=0.15, seed=1337)
    val_ds   = TokenizedDataset(args.input_data, args.tokens_pt, targets, split="val",   val_frac=0.15, seed=1337)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # --- load MLM encoder and strip the LM head ---
    ssl_dir = Path(args.ssl_dir)
    mlm = RobertaForMaskedLM.from_pretrained(ssl_dir.as_posix())
    encoder = mlm.roberta  # RobertaModel

    model = Regressor(encoder, n_targets=len(targets), dropout=0.1, pool=args.pool).to(device)

    # --- optional warmup: freeze encoder for a few epochs ---
    frozen_params = []
    if args.freeze_encoder_epochs > 0:
        for p in model.encoder.parameters():
            p.requires_grad = False
            frozen_params.append(p)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    def run_epoch(loader, train: bool, epoch: int):
        model.train(train)
        total_loss, total_mae, n = 0.0, 0.0, 0
        per_target = {k: [] for k in targets}

        for batch in loader:
            ids  = batch["input_ids"].to(device)
            am   = batch["attention_mask"].to(device)
            y    = batch["y"].to(device)
            msk  = batch["mask"].to(device)

            with torch.set_grad_enabled(train):
                pred = model(ids, am)                # [B, T]
                loss = masked_l1(pred, y, msk)       # masked L1
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            # metrics
            total_loss += loss.item() * ids.size(0)
            # masked MAE (same as loss)
            mae = masked_l1(pred.detach(), y, msk).item()
            total_mae += mae * ids.size(0); n += ids.size(0)

            # per-target MAE
            for i, name in enumerate(targets):
                mi = msk[:, i]
                if mi.any():
                    per_target[name].append((pred[:, i][mi] - y[:, i][mi]).abs().mean().item())

        avg_loss = total_loss / max(1, n)
        avg_mae  = total_mae  / max(1, n)
        per_t = {k: (float(np.mean(v)) if v else float("nan")) for k, v in per_target.items()}
        print(f"Epoch {epoch:02d} | {'train' if train else 'val'} | loss {avg_loss:.4f} | MAE {avg_mae:.4f} | per-target {per_t}")
        return avg_mae

    for ep in range(1, args.epochs + 1):
        # unfreeze after warmup
        if ep == args.freeze_encoder_epochs + 1 and frozen_params:
            for p in model.encoder.parameters(): p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            print("[info] encoder unfrozen")

        _ = run_epoch(train_loader, train=True,  epoch=ep)
        val_mae = run_epoch(val_loader,   train=False, epoch=ep)
        scheduler.step()

        # save best
        if val_mae < best_val:
            best_val = val_mae
            torch.save(model.state_dict(), out_dir / "regressor.pt")
            # also save a small meta for inference
            meta = {
                "targets": targets,
                "pool": args.pool,
                "ssl_dir": args.ssl_dir,
            }
            with open(out_dir / "meta.json", "w") as f:
                json.dump(meta, f)
            print(f"  -> saved best (val MAE {best_val:.4f}) to {out_dir}")

    print("Done.")

if __name__ == "__main__":
    main()
