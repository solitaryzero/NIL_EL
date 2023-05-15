# CUDA_VISIBLE_DEVICES=2,3 python clink/train.py \
#     --data_path /data/zfw/CLINK/data/vector \
#     --output_path /data/zfw/CLINK/models/raw_blink \
#     --learning_rate 1e-05 \
#     --cate_num 398 \
#     --num_train_epochs 4 \
#     --max_context_length 32 \
#     --max_cand_length 32 \
#     --train_batch_size 64 \
#     --eval_batch_size 128 \
#     --cate_ctxt_loss_weight 0 \
#     --cate_cand_loss_weight 0 \
#     --cate_score_weight 0 \
#     --bert_model bert-large-uncased \
#     --type_optimization all \
#     --eval_interval 200000 \
#     --data_parallel 

CUDA_VISIBLE_DEVICES=1 python genre/train.py \
    --data_path /data/zfw/NEL/data/benchmark/NEL \
    --output_path /data/zfw/NEL/models/genre \
    --learning_rate 1e-05 \
    --cate_num 211 \
    --num_train_epochs 1 \
    --max_context_length 128 \
    --max_cand_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --cate_ctxt_loss_weight 0 \
    --cate_cand_loss_weight 0 \
    --cate_score_weight 0 \
    --bert_model bert-large-uncased \
    --type_optimization all \
    --training_objective semantic \
    --eval_interval 200000 