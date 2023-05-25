CUDA_VISIBLE_DEVICES=0 python cross/evaluate.py \
    --evaluation_data_path ../data/benchmark/NEL \
    --output_path ../data/evaluation/cross_clink_nonil \
    --model_path ../models/cross_clink_nonil \
    --score_function add \
    --dataset "NEL" \
    --lambd 0.5 \
    --threshold 0.5 \
    --grid

# CUDA_VISIBLE_DEVICES=0 python cross/evaluate.py \
#     --evaluation_data_path ../data/benchmark/NEL_nonil \
#     --output_path ../data/evaluation/cross_clink_nonil \
#     --model_path ../models/cross_clink_nonil \
#     --score_function add \
#     --dataset "NEL" \
#     --lambd 0.5 \
#     --threshold 0.5 \
#     --grid