# CUDA_VISIBLE_DEVICES=0 python cross/predict.py \
#     --evaluation_data_path ../data/benchmark/NEL \
#     --output_path ../data/prediction/cross_clink \
#     --model_path ../models/cross_clink \
#     --score_function add \
#     --dataset "NEL" \
#     --lambd 0.5 \
#     --threshold 0.5 

CUDA_VISIBLE_DEVICES=0 python cross/predict.py \
    --evaluation_data_path ../data/benchmark/NEL \
    --output_path ../data/prediction/cross_clink_$1p \
    --model_path ../models/partial_nep/cross_clink_$1p \
    --score_function add \
    --dataset "NEL" \
    --lambd 0.5 \
    --threshold 0.5