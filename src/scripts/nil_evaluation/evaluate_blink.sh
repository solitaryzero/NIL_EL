# python evaluation/evaluate_nil_distribution.py \
#     --prediction_path ../data/prediction \
#     --nil_type_path ../data/benchmark/NEL/nil_type.csv \
#     --output_path ../data/nil_evaluation \
#     --model blink

python evaluation/evaluate_nil_distribution.py \
    --prediction_path ../data/prediction \
    --nil_type_path ../data/benchmark/NEL/nil_type.csv \
    --output_path ../data/nil_evaluation \
    --model blink_$1p