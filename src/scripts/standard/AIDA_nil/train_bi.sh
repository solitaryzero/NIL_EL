CUDA_VISIBLE_DEVICES=0 python evaluation/train_bi.py \
    --data_path ../data/processed_benchmark \
    --output_path ../models/clink_standard_nil \
    --learning_rate 1e-05 \
    --cate_num 14 \
    --num_train_epochs 1 \
    --max_context_length 128 \
    --max_cand_length 128 \
    --train_batch_size 4 \
    --eval_batch_size 32 \
    --cate_ctxt_loss_weight 0.5 \
    --cate_cand_loss_weight 0.5 \
    --cate_score_weight 1 \
    --bert_model bert-large-uncased \
    --type_optimization all \
    --training_objective all \
    --eval_interval 200000 \
    --dataset AIDA-YAGO2_train_nil \
    --shuffle