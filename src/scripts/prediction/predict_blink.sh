# CUDA_VISIBLE_DEVICES=7 python clink/predict.py \
#     --evaluation_data_path /data/zfw/NEL/data/benchmark/NEL \
#     --output_path /data/zfw/NEL/data/prediction/blink \
#     --model_path /data/zfw/NEL/models/raw_blink \
#     --score_function add \
#     --dataset "NEL" \
#     --lambd 1 \
#     --threshold 0.5

CUDA_VISIBLE_DEVICES=7 python clink/predict.py \
    --evaluation_data_path /data/zfw/NEL/data/benchmark/NEL \
    --output_path /data/zfw/NEL/data/prediction/blink_$1p \
    --model_path /data/zfw/NEL/models/partial_nep/blink_$1p \
    --score_function add \
    --dataset "NEL" \
    --lambd 1 \
    --threshold 0.5