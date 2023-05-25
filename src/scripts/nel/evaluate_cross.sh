CUDA_VISIBLE_DEVICES=0 python cross/evaluate.py \
    --evaluation_data_path ../data/benchmark/NEL \
    --output_path ../data/evaluation/cross_blink \
    --model_path ../models/cross_blink \
    --score_function add \
    --dataset "NEL" \
    --lambd 1 \
    --threshold 0.5