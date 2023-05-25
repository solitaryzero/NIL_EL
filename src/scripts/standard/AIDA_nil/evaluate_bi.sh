CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_bi.py \
    --evaluation_data_path ../data/processed_benchmark \
    --output_path ../data/evaluation/standard_datasets/AIDA-YAGO2_testb_nil/clink \
    --model_path ../models/clink_standard_nil \
    --score_function add \
    --dataset AIDA-YAGO2_testb_nil \
    --lambd 0.5 \
    --threshold 0.5 \
    --grid