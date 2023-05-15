CUDA_VISIBLE_DEVICES=7 python cross/evaluate.py \
    --evaluation_data_path /data/zfw/NEL/data/benchmark/NEL \
    --output_path /data/zfw/NEL/data/evaluation/cross_clink \
    --model_path /data/zfw/NEL/models/cross_clink \
    --score_function add \
    --dataset "NEL" \
    --lambd 0.5 \
    --threshold 0.5 \
    --grid