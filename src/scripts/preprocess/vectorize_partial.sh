python preprocess/vectorize.py \
    --data_path /data/zfw/NEL/data/processed \
    --nil_type_path /data/zfw/NEL/data/NEL_annotation/NEL_annotated.csv \
    --out_path /data/zfw/NEL/data/partial_vectors/vector_0p \
    --benchmark_out_path /data/zfw/NEL/data/benchmark/NEL_0p \
    --partial_nil \
    --nil_percentage 0

python preprocess/vectorize.py \
    --data_path /data/zfw/NEL/data/processed \
    --nil_type_path /data/zfw/NEL/data/NEL_annotation/NEL_annotated.csv \
    --out_path /data/zfw/NEL/data/partial_vectors/vector_25p \
    --benchmark_out_path /data/zfw/NEL/data/benchmark/NEL_25p \
    --partial_nil \
    --nil_percentage 0.25

python preprocess/vectorize.py \
    --data_path /data/zfw/NEL/data/processed \
    --nil_type_path /data/zfw/NEL/data/NEL_annotation/NEL_annotated.csv \
    --out_path /data/zfw/NEL/data/partial_vectors/vector_50p \
    --benchmark_out_path /data/zfw/NEL/data/benchmark/NEL_50p \
    --partial_nil \
    --nil_percentage 0.5

python preprocess/vectorize.py \
    --data_path /data/zfw/NEL/data/processed \
    --nil_type_path /data/zfw/NEL/data/NEL_annotation/NEL_annotated.csv \
    --out_path /data/zfw/NEL/data/partial_vectors/vector_75p \
    --benchmark_out_path /data/zfw/NEL/data/benchmark/NEL_75p \
    --partial_nil \
    --nil_percentage 0.75