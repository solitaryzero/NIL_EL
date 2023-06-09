CUDA_VISIBLE_DEVICES=0 python cross/train.py \
    --data_path ../data/vector \
    --output_path ../models/cross_blink_s2 \
    --path_to_model ../models/cross_blink/clink_model.ckpt \
    --learning_rate 1e-05 \
    --cate_num 187 \
    --num_train_epochs 4 \
    --max_context_length 128 \
    --max_cand_length 128 \
    --train_batch_size 1 \
    --eval_batch_size 32 \
    --cate_ctxt_loss_weight 1 \
    --cate_cand_loss_weight 1 \
    --cate_score_weight 1 \
    --bert_model bert-large-uncased \
    --type_optimization additional_layers \
    --training_objective category \
    --eval_interval 200000 