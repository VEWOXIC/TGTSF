# export CUDA_VISIBLE_DEVICES=5

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/TGTSF_torch_weather_large" ]; then
    mkdir ./logs/TGTSF_torch_weather_large
fi
seq_len=288
model_name=TGTSF_torch

root_path_name=./dataset/Weather_captioned
data_path_name=weather_large.parquet
news_path_name=caption_emb_large
des_path_name=caption_emb_large
model_id_name=weather_large
data_name=weather_large


random_seed=2021
for bs in 16
do
    for pred_len in 720
    do
        python -u run_TGTSF_weather.py \
        --random_seed $random_seed \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --news_path $news_path_name \
        --des_path $des_path_name \
        --info_overhead 0 \
        --news_pre_embed 1 \
        --des_pre_embed 1 \
        --add_date 1 \
        --text_dim 512 \
        --cross_layers 3 \
        --mixer_self_layers 2 \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 1 \
        --e_layers 3 \
        --n_heads 16 \
        --d_model 512 \
        --d_ff 256 \
        --dropout 0.2\
        --fc_dropout 0.2\
        --head_dropout 0\
        --patch_len 6\
        --stride 3\
        --des 'Exp' \
        --train_epochs 100\
        --patience 10\
        --target 'Channel 1' \
        --revin 0\
        --itr 1 --batch_size $bs --learning_rate 0.00005  | tee logs/TGTSF_torch_weather_large/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_bs'$bs'_p6s3_zero_Pemb'.log 
    done
done