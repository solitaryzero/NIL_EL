# CUDA_VISIBLE_DEVICES=4 python evaluation/evaluate_cross.py \
#     --evaluation_data_path /data/zfw/NEL/data/processed_benchmark \
#     --output_path /data/zfw/NEL/data/evaluation/standard_datasets \
#     --model_path /data/zfw/NEL/models/cross_standard_mask \
#     --score_function add \
#     --dataset AIDA-YAGO2_testb \
#     --lambd 0.5 \
#     --threshold 0 \
#     --grid

# CUDA_VISIBLE_DEVICES=4 python evaluation/evaluate_cross.py \
#     --evaluation_data_path /data/zfw/NEL/data/processed_benchmark \
#     --output_path /data/zfw/NEL/data/evaluation/standard_datasets \
#     --model_path /data/zfw/NEL/models/cross_standard_mask \
#     --score_function add \
#     --dataset msnbc_questions \
#     --lambd 0.5 \
#     --threshold 0 \
#     --grid

# CUDA_VISIBLE_DEVICES=4 python evaluation/evaluate_cross.py \
#     --evaluation_data_path /data/zfw/NEL/data/processed_benchmark \
#     --output_path /data/zfw/NEL/data/evaluation/standard_datasets \
#     --model_path /data/zfw/NEL/models/cross_standard_mask \
#     --score_function add \
#     --dataset wnedwiki_questions \
#     --lambd 0.5 \
#     --threshold 0 \
#     --grid

# CUDA_VISIBLE_DEVICES=5 python evaluation/evaluate_cross.py \
#     --evaluation_data_path /data/zfw/NEL/data/processed_benchmark \
#     --output_path /data/zfw/NEL/data/evaluation/standard_datasets_mask \
#     --model_path /data/zfw/NEL/models/cross_standard_mask \
#     --score_function add \
#     --dataset AIDA-YAGO2_testb \
#     --lambd 0.4 \
#     --threshold 0 \
#     --grid

CUDA_VISIBLE_DEVICES=4 python evaluation/evaluate_cross.py \
    --evaluation_data_path /data/zfw/NEL/data/processed_benchmark \
    --output_path /data/zfw/NEL/data/evaluation/standard_datasets_mask \
    --model_path /data/zfw/NEL/models/cross_standard_mask \
    --score_function add \
    --dataset msnbc_questions \
    --lambd 0.4 \
    --threshold 0 \
    --grid

CUDA_VISIBLE_DEVICES=4 python evaluation/evaluate_cross.py \
    --evaluation_data_path /data/zfw/NEL/data/processed_benchmark \
    --output_path /data/zfw/NEL/data/evaluation/standard_datasets_mask \
    --model_path /data/zfw/NEL/models/cross_standard_mask \
    --score_function add \
    --dataset wnedwiki_questions \
    --lambd 0.4 \
    --threshold 0 \
    --grid