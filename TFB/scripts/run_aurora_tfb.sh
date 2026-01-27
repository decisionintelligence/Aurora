#!/bin/bash

declare -A dataset_config
dataset_config["ETTh1"]="528 48 256"
dataset_config["ETTh2"]="528 48 256"
dataset_config["ETTm1"]="2112 192 256"
dataset_config["ETTm2"]="2112 192 256"
dataset_config["Weather"]="2880 288 128"
dataset_config["Solar"]="2880 288 32"
dataset_config["PEMS08"]="2880 288 32"
dataset_config["Electricity"]="576 48 8"
dataset_config["Traffic"]="576 48 4"
dataset_config["Wind"]="2112 192 32"
dataset_config["NYSE"]="360 24 64"


datasets=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "Weather" "Solar" "PEMS08" "Electricity" "Traffic" "Wind" "NYSE")
model_path="/home/Aurora/aurora"


for dataset in "${datasets[@]}"; do
    pids=()

    IFS=' ' read -r seq_len inference_token_len batch_size <<< "${dataset_config[$dataset]}"

    if [ "$dataset" == "NYSE" ]; then
        horizons=("24" "36" "48" "60")
    else
        horizons=("96" "192" "336" "720")
    fi

    for horizon in "${horizons[@]}"; do
        gpu_id=""

        case $horizon in
            "96") gpu_id=0 ;;
            "192") gpu_id=5 ;;
            "336") gpu_id=7 ;;
            "720") gpu_id=6 ;;
            "24") gpu_id=1 ;;
            "36") gpu_id=2 ;;
            "48") gpu_id=3 ;;
            "60") gpu_id=4 ;;
        esac

        python ./scripts/run_benchmark.py \
            --config-path rolling_forecast_config.json \
            --data-name-list "${dataset}.csv" \
            --strategy-args "{\"horizon\":${horizon}}" \
            --model-name aurora.Aurora \
            --model-hyper-params "{\"batch_size\":${batch_size},\"horizon\":${horizon},\"seq_len\":${seq_len},\"inference_token_len\":${inference_token_len},\"model_path\":\"${model_path}\"}" \
            --gpus ${gpu_id} \
            --num-workers 1 \
            --timeout 60000 \
            --save-path "${dataset}/Aurora" &

        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        wait $pid
    done
done
