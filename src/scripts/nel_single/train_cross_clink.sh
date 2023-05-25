CUDA_VISIBLE_DEVICES=0 python cross/train.py \
    --data_path ../data/vector_single \
    --output_path ../models/cross_clink_single \
    --learning_rate 1e-05 \
    --cate_num 212 \
    --num_train_epochs 1 \
    --max_context_length 128 \
    --max_cand_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --cate_ctxt_loss_weight 1 \
    --cate_cand_loss_weight 1 \
    --cate_score_weight 1 \
    --bert_model bert-large-uncased \
    --type_optimization all \
    --training_objective all \
    --eval_interval 200000 \
    --single_type