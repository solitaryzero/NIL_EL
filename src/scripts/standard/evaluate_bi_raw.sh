CUDA_VISIBLE_DEVICES=4 python evaluation/evaluate_bi.py \
    --evaluation_data_path /data/zfw/NEL/data/processed_benchmark \
    --output_path /data/zfw/NEL/data/evaluation/standard_datasets_raw \
    --model_path /data/zfw/NEL/models/blink_standard \
    --score_function add \
    --dataset AIDA-YAGO2_testb \
    --lambd 1 \
    --threshold 0 

CUDA_VISIBLE_DEVICES=4 python evaluation/evaluate_bi.py \
    --evaluation_data_path /data/zfw/NEL/data/processed_benchmark \
    --output_path /data/zfw/NEL/data/evaluation/standard_datasets_raw \
    --model_path /data/zfw/NEL/models/blink_standard \
    --score_function add \
    --dataset msnbc_questions \
    --lambd 1 \
    --threshold 0 

CUDA_VISIBLE_DEVICES=4 python evaluation/evaluate_bi.py \
    --evaluation_data_path /data/zfw/NEL/data/processed_benchmark \
    --output_path /data/zfw/NEL/data/evaluation/standard_datasets_raw \
    --model_path /data/zfw/NEL/models/blink_standard \
    --score_function add \
    --dataset wnedwiki_questions \
    --lambd 1 \
    --threshold 0 