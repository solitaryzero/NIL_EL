# CUDA_VISIBLE_DEVICES=0 python clink/predict.py \
#     --evaluation_data_path ../data/benchmark/NEL \
#     --output_path ../data/prediction/blink \
#     --model_path ../models/raw_blink \
#     --score_function add \
#     --dataset "NEL" \
#     --lambd 1 \
#     --threshold 0.5

CUDA_VISIBLE_DEVICES=0 python clink/predict.py \
    --evaluation_data_path ../data/benchmark/NEL \
    --output_path ../data/prediction/blink_$1p \
    --model_path ../models/partial_nep/blink_$1p \
    --score_function add \
    --dataset "NEL" \
    --lambd 1 \
    --threshold 0.5