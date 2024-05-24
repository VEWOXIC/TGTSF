export CUDA_VISIBLE_DEVICES=1

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/NEW_TGTSF_Steam" ]; then
    mkdir ./logs/NEW_TGTSF_Steam
fi
seq_len=60
model_name=TGTSF

# for gameid in 1172470 107410 1517290 289070 292730 730 570 548430 1366540 8500 427520 271590 582010 275850 218620 578080 359550 252950 252490 526870 281990 230410
for gameid in 10 107410 1085660 1091500 1172620 1172470 1222670 105600 1293830 1326470 1361210 1238810 1677740 1454400 1238840 1665460 1811260 1868140 1919590 1948980 214950 1623660 1938090 222880 218620 236390 221100 232050 227300 230410 240 244210 242760 236850 250900 255710 251570 270880 275850 252950 292030 284160 271590 291550 294100 304930 306130 322170 322330 281990 3590 289070 364360 365590 346110 374320 377160 381210 359550 39210 394360 386360 413150 457140 427520 493520 513710 440 529340 548430 489830 552500 526870 552990 550 570 578080 620 646570 648800 582660 582010 761890 814380 730 960090 892970 739630 231430 # 108600 1203220

do

root_path_name=./dataset/steam
data_path_name=TS/${gameid}_clean.csv
news_path_name=news_embedding/id${gameid}-embedding-paraphrase-MiniLM-L6-v2
des_path_name=des_embedding/id${gameid}-describe-embedding-paraphrase-MiniLM-L6-v2
model_id_name=toy
data_name=investing


random_seed=2021
for bs in 128
do
    for pred_len in 14
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
        --d_model 768 \
        --d_ff 256 \
        --dropout 0.2\
        --fc_dropout 0.2\
        --head_dropout 0\
        --patch_len 14\
        --stride 7\
        --des 'Exp' \
        --train_epochs 100\
        --patience 20\
        --target 'Players' \
        --itr 1 --batch_size $bs --learning_rate 0.00005  | tee logs/NEW_TGTSF_Steam/$model_name'_'$gameid'_withdrop_'$model_id_name'_'$seq_len'_'$pred_len'_bs'$bs.log 
    done
done

done