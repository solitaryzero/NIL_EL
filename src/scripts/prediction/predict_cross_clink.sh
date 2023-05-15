# CUDA_VISIBLE_DEVICES=7 python cross/predict.py \
#     --evaluation_data_path /data/zfw/NEL/data/benchmark/NEL \
#     --output_path /data/zfw/NEL/data/prediction/cross_clink \
#     --model_path /data/zfw/NEL/models/cross_clink \
#     --score_function add \
#     --dataset "NEL" \
#     --lambd 0.5 \
#     --threshold 0.5 

CUDA_VISIBLE_DEVICES=7 python cross/predict.py \
    --evaluation_data_path /data/zfw/NEL/data/benchmark/NEL \
    --output_path /data/zfw/NEL/data/prediction/cross_clink_$1p \
    --model_path /data/zfw/NEL/models/partial_nep/cross_clink_$1p \
    --score_function add \
    --dataset "NEL" \
    --lambd 0.5 \
    --threshold 0.5