#!/bin/bash

declare -A dataset_config
dataset_config["etth1"]="528 48 256"
dataset_config["etth2"]="528 48 256"
dataset_config["ettm1"]="2112 192 256"
dataset_config["ettm2"]="2112 192 256"
dataset_config["weather_ltsf"]="2880 288 128"
dataset_config["exchange_ltsf"]="96 48 64"
dataset_config["electricity_ltsf"]="576 48 8"
dataset_config["traffic_ltsf"]="576 48 4"
dataset_config["illness_ltsf"]="48 24 64"


datasets=("etth1" "etth2" "ettm1" "ettm2" "weather_ltsf" "illness_ltsf" "exchange_ltsf" "electricity_ltsf" "traffic_ltsf")

model_path="/home/Aurora/aurora"
DATA_DIR=./datasets
LOG_DIR=./results_aurora

for dataset in "${datasets[@]}"; do
    pids=()

    IFS=' ' read -r seq_len inference_token_len batch_size <<< "${dataset_config[$dataset]}"


    if [ "$dataset" == "illness_ltsf" ]; then
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

        CUDA_VISIBLE_DEVICES=${gpu_id} python run.py --config config/tsfm/aurora.yaml --seed_everything 0  \
              --data.data_manager.init_args.path ${DATA_DIR} \
              --trainer.default_root_dir ${LOG_DIR} \
              --data.data_manager.init_args.split_val true \
              --data.data_manager.init_args.dataset ${dataset} \
              --data.data_manager.init_args.context_length ${seq_len} \
              --data.data_manager.init_args.prediction_length ${horizon} \
              --model.forecaster.init_args.model_path ${model_path} \
              --model.forecaster.init_args.inference_token_len ${inference_token_len} \
              --model.num_samples 100 \
              --model.quantiles_num 20 \
              --data.test_batch_size ${batch_size} &

        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        wait $pid
    done
done