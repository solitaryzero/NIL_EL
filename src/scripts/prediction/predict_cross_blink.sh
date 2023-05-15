# CUDA_VISIBLE_DEVICES=7 python cross/predict.py \
#     --evaluation_data_path /data/zfw/NEL/data/benchmark/NEL \
#     --output_path /data/zfw/NEL/data/prediction/cross_blink \
#     --model_path /data/zfw/NEL/models/cross_blink \
#     --score_function add \
#     --dataset "NEL" \
#     --lambd 1 \
#     --threshold 0.5

CUDA_VISIBLE_DEVICES=7 python cross/predict.py \
    --evaluation_data_path /data/zfw/NEL/data/benchmark/NEL \
    --output_path /data/zfw/NEL/data/prediction/cross_blink_$1p \
    --model_path /data/zfw/NEL/models/partial_nep/cross_blink_$1p \
    --score_function add \
    --dataset "NEL" \
    --lambd 1 \
    --threshold 0.5