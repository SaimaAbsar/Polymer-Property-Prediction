python regressor.py --input_data "C:\Personal\NIPS\Data\train.csv" --tokens_pt "C:\Personal\NIPS\Data\train.tokenized.pt" --ssl_dir "C:\Personal\NIPS\saved_models\text_ssl_ckpt" --out_dir "C:\Personal\NIPS\saved_models\regressor_ckpt" --epochs 20 --batch_size 32 --lr 2e-4 --freeze_encoder_epochs 3 --pool cls

