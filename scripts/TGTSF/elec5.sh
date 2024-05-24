# export CUDA_VISIBLE_DEVICES=5

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/TGTSF_torch_elec" ]; then
    mkdir ./logs/TGTSF_torch_elec
fi
seq_len=360
model_name=TGTSF_torch

root_path_name=./dataset/electricity
data_path_name=electricity_days.csv
news_path_name=Elec_news-embedding-paraphrase-MiniLM-L6-v2
des_path_name=Elec_news-embedding-paraphrase-MiniLM-L6-v2
model_id_name=elec
data_name=elec


random_seed=2021
for bs in 8
do
    for pred_len in 720
    do
        python -u run_TGTSF_elec.py \
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
        --text_dim 384 \
        --cross_layers 3 \
        --self_layers 2 \
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
        --d_model 384 \
        --d_ff 256 \
        --dropout 0.2\
        --fc_dropout 0.2\
        --head_dropout 0\
        --patch_len 24\
        --stride 12\
        --des 'Exp' \
        --train_epochs 100\
        --patience 20\
        --target 'Channel 1' \
        --lradj 'TST' \
        --pct_start 0.1\
        --revin 0\
        --itr 1 --batch_size $bs --learning_rate 0.0001  | tee logs/TGTSF_torch_elec/norin_$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_bs'$bs'_patchlen2412_zero_Pemb'.log 
    done
done