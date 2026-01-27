#!/bin/bash

declare -A dataset_config
dataset_config["BE"]="528 48 256"
dataset_config["DE"]="528 48 256"
dataset_config["FR"]="528 48 256"
dataset_config["NP"]="528 48 256"
dataset_config["PJM"]="528 48 256"

datasets=("BE" "DE" "FR" "NP" "PJM")

model_path="/home/Aurora/aurora"
gpu_id=7

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

    horizons=("24")
    features='MS'
    data='custom'
    root_path="/home/Aurora/EPF/dataset/EPF"
    end='.csv'


    for horizon in "${horizons[@]}"; do
        python run.py \
            --is_training 0 \
            --task_name 'long_term_forecasting' \
            --features ${features} \
            --seq_len ${seq_len} \
            --pred_len ${horizon} \
            --inference_token_len ${inference_token_len} \
            --data ${data} \
            --data_path "${dataset}${end}" \
            --batch_size ${batch_size} \
            --model_path ${model_path} \
            --root_path ${root_path} \
            --gpu ${gpu_id} \
            >logs/Forecasting/$dataset/'_seq_len'$seq_len'_inference_token_len'$inference_token_len'_pred_len'$horizon.log
    done
done