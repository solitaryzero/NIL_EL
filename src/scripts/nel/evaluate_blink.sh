# CUDA_VISIBLE_DEVICES=6 python clink/evaluate.py \
#     --evaluation_data_path /data/zfw/NEL/data/benchmark/NEL \
#     --output_path /data/zfw/NEL/data/evaluation/raw_blink \
#     --model_path /data/zfw/NEL/models/raw_blink \
#     --score_function add \
#     --dataset "NEL" \
#     --lambd 1 \
#     --threshold 0.5 \
#     --grid

CUDA_VISIBLE_DEVICES=6 python clink/evaluate.py \
    --evaluation_data_path /data/zfw/NEL/data/benchmark/NEL \
    --output_path /data/zfw/NEL/data/evaluation/raw_blink \
    --model_path /data/zfw/NEL/models/raw_blink \
    --score_function add \
    --dataset "NEL" \
    --lambd 1 \
    --threshold 0.5


# CUDA_VISIBLE_DEVICES=6 python clink/evaluate.py \
#     --evaluation_data_path /data/zfw/NEL/data/benchmark/NEL \
#     --output_path /data/zfw/NEL/data/evaluation/raw_blink \
#     --model_path /data/zfw/NEL/models/raw_blink \
#     --score_function add \
#     --dataset "NEL" \
#     --lambd 1 \
#     --threshold 0