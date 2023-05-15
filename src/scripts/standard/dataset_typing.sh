python evaluation/dataset.py \
    --benchmark_data_path /data/zfw/NEL/data/processed_benchmark \
    --type_index_path /data/zfw/NEL/data/raw/type_index.jsonl \
    --dataset AIDA-YAGO2_train

python evaluation/dataset.py \
    --benchmark_data_path /data/zfw/NEL/data/processed_benchmark \
    --type_index_path /data/zfw/NEL/data/raw/type_index.jsonl \
    --dataset AIDA-YAGO2_testa

python evaluation/dataset.py \
    --benchmark_data_path /data/zfw/NEL/data/processed_benchmark \
    --type_index_path /data/zfw/NEL/data/raw/type_index.jsonl \
    --dataset AIDA-YAGO2_testb

python evaluation/dataset.py \
    --benchmark_data_path /data/zfw/NEL/data/processed_benchmark \
    --type_index_path /data/zfw/NEL/data/raw/type_index.jsonl \
    --dataset AIDA-YAGO2_train_nil

python evaluation/dataset.py \
    --benchmark_data_path /data/zfw/NEL/data/processed_benchmark \
    --type_index_path /data/zfw/NEL/data/raw/type_index.jsonl \
    --dataset AIDA-YAGO2_testa_nil

python evaluation/dataset.py \
    --benchmark_data_path /data/zfw/NEL/data/processed_benchmark \
    --type_index_path /data/zfw/NEL/data/raw/type_index.jsonl \
    --dataset AIDA-YAGO2_testb_nil

python evaluation/dataset.py \
    --benchmark_data_path /data/zfw/NEL/data/processed_benchmark \
    --type_index_path /data/zfw/NEL/data/raw/type_index.jsonl \
    --dataset msnbc_questions

python evaluation/dataset.py \
    --benchmark_data_path /data/zfw/NEL/data/processed_benchmark \
    --type_index_path /data/zfw/NEL/data/raw/type_index.jsonl \
    --dataset wnedwiki_questions