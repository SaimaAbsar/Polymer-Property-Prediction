# Script to predict polymer properties from test data
import argparse, json
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import RobertaForMaskedLM

# ---- dataset for tokenized test ----
class TokenOnlyDataset(Dataset):
    def __init__(self, tokens_pt):
        blob = torch.load(tokens_pt, map_location="cpu")
        self.input_ids = blob["input_ids"].long()
        self.attn_mask = blob["attention_mask"].long()
        # row_index optional; if missing, we synthesize 0..N-1
        self.row_index = blob.get("row_index", torch.arange(self.input_ids.size(0), dtype=torch.long))

    def __len__(self): return self.input_ids.size(0)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attn_mask[i],
            "row": int(self.row_index[i].item())
        }

def collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch], 0),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch], 0),
        "row": torch.tensor([b["row"] for b in batch], dtype=torch.long)
    }

# ---- the same regressor head used in training ----
class Regressor(nn.Module):
    def __init__(self, encoder, n_targets: int, dropout=0.1, pool="cls"):
        super().__init__()
        self.encoder = encoder
        self.pool = pool
        H = encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(H, n_targets)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state  # [B, L, H]
        if self.pool == "cls":
            pooled = last[:, 0, :]
        else:
            m = attention_mask.unsqueeze(-1)
            pooled = (last * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(self.drop(pooled))  # [B, T]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens_pt", required=True, help="test.tokenized.pt")
    ap.add_argument("--regressor_dir", required=True, help="folder with regressor.pt and meta.json")
    ap.add_argument("--test_data", required=True, help="original test.csv (to carry IDs/SMILES in output)")
    ap.add_argument("--csv_id_col", default="id", help="optional column to treat as ID in output")
    ap.add_argument("--out_csv", required=True, help="where to save predictions CSV")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load meta + targets + encoder path ---
    reg_dir = Path(args.regressor_dir)
    with open(reg_dir / "meta.json", "r") as f:
        meta = json.load(f)
    targets = meta["targets"]
    pool = meta.get("pool", "cls")
    ssl_dir = meta["ssl_dir"]

    # --- rebuild model: load SSL encoder, attach regressor head, then load head+encoder weights ---
    mlm = RobertaForMaskedLM.from_pretrained(ssl_dir)
    encoder = mlm.roberta
    model = Regressor(encoder, n_targets=len(targets), pool=pool).to(device)
    state = torch.load(reg_dir / "regressor.pt", map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    # --- data ---
    ds = TokenOnlyDataset(args.tokens_pt)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # --- forward pass ---
    rows = []
    preds = []
    with torch.no_grad():
        for batch in dl:
            ids = batch["input_ids"].to(device)
            am  = batch["attention_mask"].to(device)
            yhat = model(ids, am).cpu()  # [B, T]
            preds.append(yhat)
            rows.extend(batch["row"].tolist())

    preds = torch.cat(preds, dim=0).numpy()  # [N, T]

    # --- build output frame ---
    df_in = pd.read_csv(args.test_data)
    # Align by row index: assume tokenization kept original order
    out = pd.DataFrame()
    if args.csv_id_col and args.csv_id_col in df_in.columns:
        out[args.csv_id_col] = df_in.loc[rows, args.csv_id_col].values
    elif "id" in df_in.columns:
        out["id"] = df_in.loc[rows, "id"].values
    else: 
        out["row_index"] = rows

    if "SMILES" in df_in.columns:
        out["SMILES"] = df_in.loc[rows, "SMILES"].astype(str).values

    for i, name in enumerate(targets):
        out[name] = preds[:, i]

    out.to_csv(args.out_csv, index=False)
    print(f"Saved predictions → {args.out_csv}")
    print("Columns:", out.columns.tolist())

if __name__ == "__main__":
    main()
