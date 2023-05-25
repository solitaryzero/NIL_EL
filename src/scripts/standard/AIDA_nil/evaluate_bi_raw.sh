CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_bi.py \
    --evaluation_data_path ../data/processed_benchmark \
    --output_path ../data/evaluation/standard_datasets/AIDA-YAGO2_testb_nil/blink \
    --model_path ../models/blink_standard_nil \
    --score_function add \
    --dataset AIDA-YAGO2_testb_nil \
    --lambd 1 \
    --threshold 0.5 