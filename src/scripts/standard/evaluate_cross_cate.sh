CUDA_VISIBLE_DEVICES=4 python evaluation/evaluate_cate.py \
    --evaluation_data_path /data/zfw/NEL/data/processed_benchmark \
    --output_path /data/zfw/NEL/data/cate_evaluation/standard_datasets_mask \
    --type_map_path /data/zfw/NEL/data/raw/type_index.jsonl \
    --model_path /data/zfw/NEL/models/cross_standard_mask \
    --score_function add \
    --dataset AIDA-YAGO2_testb \
    --lambd 0.5 \
    --threshold 0.5 


# CUDA_VISIBLE_DEVICES=5 python cross/evaluate_cate.py \
#     --evaluation_data_path /data/zfw/NEL/data/benchmark/NEL \
#     --output_path /data/zfw/NEL/data/cate_evaluation/cross_blink_s2 \
#     --model_path /data/zfw/NEL/models/cross_blink_s2 \
#     --score_function add \
#     --dataset "NEL" \
#     --lambd 0.5 \
#     --threshold 0.5 

# CUDA_VISIBLE_DEVICES=1 python cross/evaluate.py \
#     --evaluation_data_path /data/zfw/NEL/data/benchmark/NEL \
#     --output_path /data/zfw/NEL/data/evaluation/cross_clink_mul \
#     --model_path /data/zfw/NEL/models/cross_clink \
#     --score_function mul \
#     --dataset "NEL" \
#     --lambd 0.5 \
#     --threshold 0.0002 