if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=60
model_name=PatchTST

root_path_name=./dataset/toydataset
data_path_name=toydata.csv
model_id_name=toy_PatchTST
data_name=custom

random_seed=2021
for pred_len in 14 28 60 120
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 0 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --label_len 0\
      --enc_in 1 \
      --e_layers 3 \
      --n_heads 16 \
      --d_model 768 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --patience 3\
      --target 'Channel 1' \
      --itr 1 --batch_size 128 --learning_rate 0.0001 --gpu 0  | tee logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done