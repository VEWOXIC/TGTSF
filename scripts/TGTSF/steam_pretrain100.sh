export CUDA_VISIBLE_DEVICES=3

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

for gameid in 10 107410 1085660 1091500 1172620 1172470 1222670 105600 1293830 1326470 1361210 1238810 1677740 1454400 1238840 1665460 1811260 1868140 1919590 1948980 214950 1623660 1938090 222880 218620 236390 221100 232050 227300 230410 240 244210 242760 236850 250900 255710 251570 270880 275850 252950 292030 284160 271590 291550 294100 304930 306130 322170 322330 281990 3590 289070 364360 365590 346110 374320 377160 381210 359550 39210 394360 386360 413150 457140 427520 493520 513710 440 529340 548430 489830 552500 526870 552990 550 570 578080 620 646570 648800 582660 582010 761890 814380 730 960090 892970 739630 231430 # 108600 1203220

do

data_path_name+="TS/${gameid}_clean.csv "
news_path_name+="news_embedding/id${gameid}-embedding-paraphrase-MiniLM-L6-v2 "
des_path_name+="des_embedding/id${gameid}-describe-embedding-paraphrase-MiniLM-L6-v2 "

done

root_path_name=./dataset/steam

model_id_name=steam100-NoRIN
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
        --train_epochs 1000\
        --patience 50\
        --target 'Players' \
        --itr 1 --batch_size $bs --learning_rate 5e-6   | tee logs/TGTSF_Steam_pretrain/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_bs'$bs'-100'.log 
    done
done

# done