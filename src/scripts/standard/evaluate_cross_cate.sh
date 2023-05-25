CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_cate.py \
    --evaluation_data_path ../data/processed_benchmark \
    --output_path ../data/cate_evaluation/standard_datasets_mask \
    --type_map_path ../data/raw/type_index.jsonl \
    --model_path ../models/cross_standard_mask \
    --score_function add \
    --dataset AIDA-YAGO2_testb \
    --lambd 0.5 \
    --threshold 0.5 


# CUDA_VISIBLE_DEVICES=0 python cross/evaluate_cate.py \
#     --evaluation_data_path ../data/benchmark/NEL \
#     --output_path ../data/cate_evaluation/cross_blink_s2 \
#     --model_path ../models/cross_blink_s2 \
#     --score_function add \
#     --dataset "NEL" \
#     --lambd 0.5 \
#     --threshold 0.5 

# CUDA_VISIBLE_DEVICES=0 python cross/evaluate.py \
#     --evaluation_data_path ../data/benchmark/NEL \
#     --output_path ../data/evaluation/cross_clink_mul \
#     --model_path ../models/cross_clink \
#     --score_function mul \
#     --dataset "NEL" \
#     --lambd 0.5 \
#     --threshold 0.0002 