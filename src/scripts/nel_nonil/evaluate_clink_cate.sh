CUDA_VISIBLE_DEVICES=4 python clink/evaluate_cate.py \
    --evaluation_data_path /data/zfw/NEL/data/benchmark/NEL \
    --output_path /data/zfw/NEL/data/cate_evaluation/clink \
    --model_path /data/zfw/NEL/models/clink \
    --score_function add \
    --dataset "NEL" \
    --lambd 0.5 \
    --threshold 0.5 