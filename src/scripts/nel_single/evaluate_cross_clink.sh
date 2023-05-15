CUDA_VISIBLE_DEVICES=1 python cross/evaluate.py \
    --evaluation_data_path /data/zfw/NEL/data/benchmark/NEL_single \
    --output_path /data/zfw/NEL/data/evaluation/cross_clink_single \
    --model_path /data/zfw/NEL/models/cross_clink_single \
    --score_function add \
    --dataset "NEL" \
    --lambd 0.5 \
    --threshold 0.5