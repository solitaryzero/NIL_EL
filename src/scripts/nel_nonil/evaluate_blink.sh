CUDA_VISIBLE_DEVICES=6 python clink/evaluate.py \
    --evaluation_data_path /data/zfw/NEL/data/benchmark/NEL \
    --output_path /data/zfw/NEL/data/evaluation/blink_nonil \
    --model_path /data/zfw/NEL/models/blink_nonil \
    --score_function add \
    --dataset "NEL" \
    --lambd 1 \
    --threshold 0.5


# CUDA_VISIBLE_DEVICES=6 python clink/evaluate.py \
#     --evaluation_data_path /data/zfw/NEL/data/benchmark/NEL_nonil \
#     --output_path /data/zfw/NEL/data/evaluation/blink_nonil \
#     --model_path /data/zfw/NEL/models/blink_nonil \
#     --score_function add \
#     --dataset "NEL" \
#     --lambd 1 \
#     --threshold 0.5