# CUDA_VISIBLE_DEVICES=7 python clink/predict.py \
#     --evaluation_data_path /data/zfw/NEL/data/benchmark/NEL \
#     --output_path /data/zfw/NEL/data/prediction/clink \
#     --model_path /data/zfw/NEL/models/clink \
#     --score_function add \
#     --dataset "NEL" \
#     --lambd 0.5 \
#     --threshold 0.5


CUDA_VISIBLE_DEVICES=7 python clink/predict.py \
    --evaluation_data_path /data/zfw/NEL/data/benchmark/NEL \
    --output_path /data/zfw/NEL/data/prediction/clink_$1p \
    --model_path /data/zfw/NEL/models/partial_nep/clink_$1p \
    --score_function add \
    --dataset "NEL" \
    --lambd 0.5 \
    --threshold 0.5