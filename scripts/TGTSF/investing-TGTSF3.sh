if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/investing_aug" ]; then
    mkdir ./logs/investing_aug
fi
seq_len=60
model_name=TGTSF

root_path_name=./dataset/investing
data_path_name=investing_exchange.csv
news_path_name=News-embedding-openai
des_path_name=des_embeddings_openai.npy
model_id_name=investing_openai
data_name=custom

random_seed=2021
for pred_len in 7 
do
    for bs in 64
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
        --text_dim 1536 \
        --model_id $model_id_name_$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --label_len 0\
        --enc_in 9 \
        --e_layers 3 \
        --n_heads 16 \
        --d_model 256 \
        --d_ff 256 \
        --dropout 0.2\
        --fc_dropout 0.2\
        --head_dropout 0\
        --patch_len 16\
        --stride 8\
        --des 'Exp' \
        --train_epochs 10000\
        --patience 100\
        --target 'CHF' \
        --itr 1 --batch_size $bs --learning_rate 0.0005 --in_batch_augmentation --aug_rate 0.5 --aug_method f_mask   | tee logs/investing_aug/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_bs'$bs'_overhead7.log' 
done
done