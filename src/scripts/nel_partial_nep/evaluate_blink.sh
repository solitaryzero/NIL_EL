CUDA_VISIBLE_DEVICES=6 python clink/evaluate.py \
    --evaluation_data_path /data/zfw/NEL/data/benchmark/NEL \
    --output_path /data/zfw/NEL/data/evaluation/partial_nep/blink_$1p \
    --model_path /data/zfw/NEL/models/partial_nep/blink_$1p \
    --score_function add \
    --dataset "NEL" \
    --lambd 1 \
    --threshold 0.5