export CUDA_VISIBLE_DEVICES=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/TGTSF_Steam_pretrain" ]; then
    mkdir ./logs/TGTSF_Steam_pretrain
fi
seq_len=60
model_name=TGTSF

data_path_name=""
news_path_name=""
des_path_name=""

for gameid in 570 730 230410
do

data_path_name+="data_old/id${gameid}_online_lifelong_clean.csv "
news_path_name+="news_embedding/id${gameid}-embedding-paraphrase-MiniLM-L6-v2 "
des_path_name+="des_embedding/id${gameid}-describe-embedding-paraphrase-MiniLM-L6-v2 "

done

root_path_name=./dataset/steam

model_id_name=toy
data_name=investing


random_seed=2021
for bs in 128
do
    for pred_len in 14
    do
        python -u run_TGTSF_pretrain.py \
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
        --cross_layers 1 \
        --self_layers 7 \
        --mixer_self_layers 5 \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features S \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 1 \
        --e_layers 5 \
        --n_heads 16 \
        --d_model 384 \
        --d_ff 256 \
        --dropout 0.2\
        --fc_dropout 0.2\
        --head_dropout 0\
        --patch_len 14\
        --stride 2\
        --des 'Exp' \
        --train_epochs 100\
        --patience 1\
        --target 'Players' \
        --itr 1 --batch_size $bs --learning_rate 1e-5 --global_norm  | tee logs/TGTSF_Steam_pretrain/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_bs'$bs'TTTTTTTEST'.log 
    done
done

# done