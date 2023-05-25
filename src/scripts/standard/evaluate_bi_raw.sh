CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_bi.py \
    --evaluation_data_path ../data/processed_benchmark \
    --output_path ../data/evaluation/standard_datasets_raw \
    --model_path ../models/blink_standard \
    --score_function add \
    --dataset AIDA-YAGO2_testb \
    --lambd 1 \
    --threshold 0 

CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_bi.py \
    --evaluation_data_path ../data/processed_benchmark \
    --output_path ../data/evaluation/standard_datasets_raw \
    --model_path ../models/blink_standard \
    --score_function add \
    --dataset msnbc_questions \
    --lambd 1 \
    --threshold 0 

CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_bi.py \
    --evaluation_data_path ../data/processed_benchmark \
    --output_path ../data/evaluation/standard_datasets_raw \
    --model_path ../models/blink_standard \
    --score_function add \
    --dataset wnedwiki_questions \
    --lambd 1 \
    --threshold 0 