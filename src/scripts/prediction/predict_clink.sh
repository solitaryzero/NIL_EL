# CUDA_VISIBLE_DEVICES=0 python clink/predict.py \
#     --evaluation_data_path ../data/benchmark/NEL \
#     --output_path ../data/prediction/clink \
#     --model_path ../models/clink \
#     --score_function add \
#     --dataset "NEL" \
#     --lambd 0.5 \
#     --threshold 0.5


CUDA_VISIBLE_DEVICES=0 python clink/predict.py \
    --evaluation_data_path ../data/benchmark/NEL \
    --output_path ../data/prediction/clink_$1p \
    --model_path ../models/partial_nep/clink_$1p \
    --score_function add \
    --dataset "NEL" \
    --lambd 0.5 \
    --threshold 0.5