CUDA_VISIBLE_DEVICES=0 python genre/evaluate.py \
    --evaluation_data_path ../data/benchmark/NEL \
    --output_path ../data/evaluation/genre \
    --model_path ../models/genre \
    --dataset "NEL" 