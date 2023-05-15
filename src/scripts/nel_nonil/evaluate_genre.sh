# CUDA_VISIBLE_DEVICES=0 python evaluations/evaluate.py \
#     --evaluation_data_path /data/zfw/CLINK/data/benchmarks \
#     --wikipedia_data_path /data/zfw/CLINK/data/precomputed/blink \
#     --wikipedia_info_path /data/zfw/CLINK/data/raw/wikipedia/en_instance_ID.txt \
#     --output_path /data/zfw/CLINK/evaluation/blink \
#     --model_path /data/zfw/CLINK/models/blink \
#     --dataset "AIDA-YAGO2_testb" \
#     --batch_size 16 \
#     --lambd 0.5 \
#     --k 10 \
#     --debug

CUDA_VISIBLE_DEVICES=2 python genre/evaluate.py \
    --evaluation_data_path /data/zfw/NEL/data/benchmark/NEL \
    --output_path /data/zfw/NEL/data/evaluation/genre \
    --model_path /data/zfw/NEL/models/genre \
    --dataset "NEL" 