export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/TGTSF" ]; then
    mkdir ./logs/TGTSF
fi
seq_len=60
model_name=TGTSF

root_path_name=./dataset/toydataset
data_path_name=toydata.csv
news_path_name=News-embedding-paraphrase-MiniLM-L6-v2
des_path_name=des_embeddings.npy
model_id_name=toy
data_name=investing


random_seed=2021
for bs in 256
do
    for pred_len in 14 28 60 120
    do
        python -u run_TGTSF.py \
        --random_seed $random_seed \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --news_path $news_path_name \
        --des_path $des_path_name \
        --info_overhead 0 \
        --news_pre_embed 1 \
        --des_pre_embed 1 \
        --add_date 0 \
        --text_dim 384 \
        --cross_layers 3 \
        --self_layers 3 \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
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
        --patience 20\
        --target 'Channel 1' \
        --itr 1 --batch_size $bs --learning_rate 0.0001  #| tee logs/TGTSF/$model_name'_withdrop_'$model_id_name'_'$seq_len'_'$pred_len'_bs'$bs.log 
    done
done