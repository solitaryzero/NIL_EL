# CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_cross.py \
#     --evaluation_data_path ../data/processed_benchmark \
#     --output_path ../data/evaluation/standard_datasets \
#     --model_path ../models/cross_standard_mask \
#     --score_function add \
#     --dataset AIDA-YAGO2_testb \
#     --lambd 0.5 \
#     --threshold 0 \
#     --grid

# CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_cross.py \
#     --evaluation_data_path ../data/processed_benchmark \
#     --output_path ../data/evaluation/standard_datasets \
#     --model_path ../models/cross_standard_mask \
#     --score_function add \
#     --dataset msnbc_questions \
#     --lambd 0.5 \
#     --threshold 0 \
#     --grid

# CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_cross.py \
#     --evaluation_data_path ../data/processed_benchmark \
#     --output_path ../data/evaluation/standard_datasets \
#     --model_path ../models/cross_standard_mask \
#     --score_function add \
#     --dataset wnedwiki_questions \
#     --lambd 0.5 \
#     --threshold 0 \
#     --grid

# CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_cross.py \
#     --evaluation_data_path ../data/processed_benchmark \
#     --output_path ../data/evaluation/standard_datasets_mask \
#     --model_path ../models/cross_standard_mask \
#     --score_function add \
#     --dataset AIDA-YAGO2_testb \
#     --lambd 0.4 \
#     --threshold 0 \
#     --grid

CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_cross.py \
    --evaluation_data_path ../data/processed_benchmark \
    --output_path ../data/evaluation/standard_datasets_mask \
    --model_path ../models/cross_standard_mask \
    --score_function add \
    --dataset msnbc_questions \
    --lambd 0.4 \
    --threshold 0 \
    --grid

CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_cross.py \
    --evaluation_data_path ../data/processed_benchmark \
    --output_path ../data/evaluation/standard_datasets_mask \
    --model_path ../models/cross_standard_mask \
    --score_function add \
    --dataset wnedwiki_questions \
    --lambd 0.4 \
    --threshold 0 \
    --grid