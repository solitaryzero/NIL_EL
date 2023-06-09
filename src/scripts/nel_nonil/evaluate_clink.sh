CUDA_VISIBLE_DEVICES=0 python clink/evaluate.py \
    --evaluation_data_path ../data/benchmark/NEL \
    --output_path ../data/evaluation/clink_nonil \
    --model_path ../models/clink_nonil \
    --score_function add \
    --dataset "NEL" \
    --lambd 0.5 \
    --threshold 0.5 \
    --grid

# CUDA_VISIBLE_DEVICES=0 python clink/evaluate.py \
#     --evaluation_data_path ../data/benchmark/NEL_nonil \
#     --output_path ../data/evaluation/clink_nonil \
#     --model_path ../models/clink_nonil \
#     --score_function add \
#     --dataset "NEL" \
#     --lambd 0.5 \
#     --threshold 0.5 \
#     --grid