CUDA_VISIBLE_DEVICES=7 python evaluation/evaluate_cross.py \
    --evaluation_data_path /data/zfw/NEL/data/processed_benchmark \
    --output_path /data/zfw/NEL/data/evaluation/standard_datasets/AIDA-YAGO2_testb_nil/cross \
    --model_path /data/zfw/NEL/models/cross_standard_nil \
    --score_function add \
    --dataset AIDA-YAGO2_testb_nil \
    --lambd 0.5 \
    --threshold 0.5 \
    --grid