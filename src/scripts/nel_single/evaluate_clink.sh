# CUDA_VISIBLE_DEVICES=2 python clink/evaluate.py \
#     --evaluation_data_path /data/zfw/NEL/data/benchmark/NEL \
#     --output_path /data/zfw/NEL/data/evaluation/clink \
#     --model_path /data/zfw/NEL/models/clink \
#     --score_function add \
#     --dataset "NEL" \
#     --lambd 0.5 \
#     --threshold 0.5 \
#     --grid


CUDA_VISIBLE_DEVICES=2 python clink/evaluate.py \
    --evaluation_data_path /data/zfw/NEL/data/benchmark/NEL_single \
    --output_path /data/zfw/NEL/data/evaluation/clink_single \
    --model_path /data/zfw/NEL/models/clink_single \
    --score_function mul \
    --dataset "NEL" \
    --threshold 0.2