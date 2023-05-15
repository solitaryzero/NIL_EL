CUDA_VISIBLE_DEVICES=7 python cross/evaluate.py \
    --evaluation_data_path /data/zfw/NEL/data/benchmark/NEL \
    --output_path /data/zfw/NEL/data/evaluation/partial_nep/cross_clink_$1p \
    --model_path /data/zfw/NEL/models/partial_nep/cross_clink_$1p \
    --score_function add \
    --dataset "NEL" \
    --lambd 0.5 \
    --threshold 0.5 \
    --grid