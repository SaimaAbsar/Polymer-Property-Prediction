# Polymer Property Prediction 
This project builds a **Transformer-based pipeline** to predict polymer properties (e.g., Tg, FFV, Tc, Density, Rg) from polymer string data (SMILES).

---

## Workflow

### 1. Tokenization with TransPolymer
- Used the **TransPolymer tokenizer** (adapted from RoBERTa) to split SMILES strings into tokens.  
- Each tokenized sequence has:
  - `input_ids` (the token IDs)
  - `attention_mask` (marks which parts are real tokens vs padding).  
- These are saved into a `*.tokenized.pt` file inside the data directory.

---

### 2. Self-Supervised Pretraining (MLM)
- Trained a **Masked Language Model (MLM)** on the tokenized SMILES data.  
- Process:
  - Randomly hide ~15% of the tokens.
  - The encoder learns to predict the hidden tokens from the surrounding context.
- This step helps the model learn the “language” of polymers before we predicting any physical properties.  
- The pretrained encoder is saved (e.g. `saved_models/ssl_mlm_ckpt/`).

---

### 3. Supervised Regression
- Built a **regressor on top of the pretrained encoder**:
  - The MLM encoder is reused as a feature extractor.
  - A small linear layer (the “regression head”) maps the embeddings from encoder → 5 target properties.
- Training:
  - Uses the input train data CSV with labels.
  - Handles **missing values** by using a mask for the rows with missing labels.
  - Uses **masked L1 loss** (mean absolute error only where labels exist).
- The best model checkpoint is saved in `saved_models/regressor_ckpt/`.

---

## Running the Code

#### Pretrain (MLM)

python self_supervised_mlm.py \
  --tp_dir ./TransPolymer-master \
  --vocab ./TransPolymer-master/tokenizer/vocab.json \
  --merges ./TransPolymer-master/tokenizer/merges.txt \
  --tokens_pt train.tokenized.pt \
  --out_dir saved_models/ssl_mlm_ckpt

#### Fine-tune
python regressor.py \
  --input_data ./data/train.csv \
  --tokens_pt ./data/train.tokenized.pt \
  --ssl_dir ./saved_models/text_ssl_ckpt \
  --out_dir ./saved_models/regressor_ckpt

#### Prediction
python predict_test.py \
  --tokens_pt ./data/test.tokenized.pt \
  --regressor_dir ./saved_models/regressor_ckpt \
  --test_data ./data/test.csv \
  --out_csv ./results/predictions_test.csv \

## Running the model using notebook: 
1. Tokenization
2. Main
