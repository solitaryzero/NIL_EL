CUDA_VISIBLE_DEVICES=0 python clink/evaluate_cate.py \
    --evaluation_data_path ../data/benchmark/NEL \
    --output_path ../data/cate_evaluation/clink \
    --model_path ../models/clink \
    --score_function add \
    --dataset "NEL" \
    --lambd 0.5 \
    --threshold 0.5 