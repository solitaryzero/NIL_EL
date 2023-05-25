CUDA_VISIBLE_DEVICES=0 python cross/evaluate.py \
    --evaluation_data_path ../data/benchmark/NEL \
    --output_path ../data/evaluation/cross_clink \
    --model_path ../models/cross_clink \
    --score_function add \
    --dataset "NEL" \
    --lambd 0.5 \
    --threshold 0.5 \
    --grid