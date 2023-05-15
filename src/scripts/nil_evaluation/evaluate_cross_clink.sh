# python evaluation/evaluate_nil_distribution.py \
#     --prediction_path ../data/prediction \
#     --nil_type_path ../data/benchmark/NEL/nil_type.csv \
#     --output_path ../data/nil_evaluation \
#     --model cross_clink

python evaluation/evaluate_nil_distribution.py \
    --prediction_path ../data/prediction \
    --nil_type_path ../data/benchmark/NEL/nil_type.csv \
    --output_path ../data/nil_evaluation \
    --model cross_clink_$1p