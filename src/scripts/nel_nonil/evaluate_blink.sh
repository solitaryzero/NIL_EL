CUDA_VISIBLE_DEVICES=0 python clink/evaluate.py \
    --evaluation_data_path ../data/benchmark/NEL \
    --output_path ../data/evaluation/blink_nonil \
    --model_path ../models/blink_nonil \
    --score_function add \
    --dataset "NEL" \
    --lambd 1 \
    --threshold 0.5


# CUDA_VISIBLE_DEVICES=0 python clink/evaluate.py \
#     --evaluation_data_path ../data/benchmark/NEL_nonil \
#     --output_path ../data/evaluation/blink_nonil \
#     --model_path ../models/blink_nonil \
#     --score_function add \
#     --dataset "NEL" \
#     --lambd 1 \
#     --threshold 0.5