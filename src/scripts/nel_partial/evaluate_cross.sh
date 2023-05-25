CUDA_VISIBLE_DEVICES=0 python cross/evaluate.py \
    --evaluation_data_path ../data/benchmark/NEL \
    --output_path ../data/evaluation/partial/cross_blink_$1p \
    --model_path ../models/partial/cross_blink_$1p \
    --score_function add \
    --dataset "NEL" \
    --lambd 1 \
    --threshold 0.5