#!/bin/bash

declare -A dataset_config
dataset_config["Agriculture"]="192 48 256"
dataset_config["Climate"]="192 48 256"
dataset_config["Economy"]="192 48 256"
dataset_config["Energy"]="1056 48 256"
dataset_config["Environment"]="528 48 256"
dataset_config["Health"]="96 48 256"
dataset_config["Security"]="220 24 256"
dataset_config["SocialGood"]="192 48 256"
dataset_config["Traffic"]="96 48 256"
dataset_config["Weather"]="1440 48 256"
dataset_config["EWJ"]="1056 48 256"
dataset_config["KR"]="1056 48 256"
dataset_config["MDT"]="528 48 256"

datasets=("Agriculture" "Climate" "Economy" "Energy" "Environment" "Health" "Security" "Traffic" "SocialGood")
model_path="/home/Aurora/aurora"
gpu_id=1

for dataset in "${datasets[@]}"; do
    if [ ! -d "logs" ]; then
    mkdir logs
    fi

    if [ ! -d "logs/Forecasting" ]; then
        mkdir logs/Forecasting
    fi
    if [ ! -d "logs/Forecasting/$dataset" ]; then
        mkdir logs/Forecasting/$dataset
    fi

    IFS=' ' read -r seq_len inference_token_len batch_size <<< "${dataset_config[$dataset]}"

    if [ "$dataset" == "Energy" ] || [ "$dataset" == "Health" ]; then
        horizons=("12" "24" "36" "48")
    elif [ "$dataset" == "Environment" ] || [ "$dataset" == "Weather" ]; then
        horizons=("48" "96" "192" "336")
    else
        horizons=("6" "8" "10" "12")
    fi

    for horizon in "${horizons[@]}"; do
        python run_longExp.py \
            --is_training 0 \
            --features "S" \
            --seq_len ${seq_len} \
            --pred_len ${horizon} \
            --inference_token_len ${inference_token_len} \
            --data ${dataset} \
            --data_path "${dataset}.csv" \
            --batch_size ${batch_size} \
            --model_path ${model_path} \
            --root_path "/home/Aurora/TimeMMD/dataset" \
            --gpu ${gpu_id} \
            >logs/Forecasting/$dataset/'_seq_len'$seq_len'_inference_token_len'$inference_token_len'_pred_len'$horizon.log
    done
done