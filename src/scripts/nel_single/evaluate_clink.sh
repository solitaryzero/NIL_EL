CUDA_VISIBLE_DEVICES=0 python clink/evaluate.py \
    --evaluation_data_path ../data/benchmark/NEL_single \
    --output_path ../data/evaluation/clink_single \
    --model_path ../models/clink_single \
    --score_function mul \
    --dataset "NEL" \
    --threshold 0.2