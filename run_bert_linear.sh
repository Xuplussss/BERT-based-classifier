#!/bin/bash
for i in {4..4}; do
    CUDA_VISIBLE_DEVICES=5 python bert_linear_train.py --text_file data/tenfold/fold${i}/train_chief.txt --condition_file data/tenfold/fold${i}/train_answer.txt --bert_model bert-base-chinese --output_dir bert_result/chief/fold${i}/ --label_num 2 --max_seq_length 400 --do_train --train_batch_size 16 --learning_rate 3e-5 --num_train_epochs 15
    CUDA_VISIBLE_DEVICES=5 python bert_linear_test.py --text_file data/tenfold/fold${i}/test_chief.txt --bert_model bert-base-chinese --label_num 2 --max_seq_length 400 --eval_batch_size 16 --load_config bert_result/chief/fold${i}/bert_config/16315bert_config.json --load_model bert_result/chief/fold${i}/bert_model/16315pytorch_model.bin --do_eval --object_type usr
done
python evaluation.py --bert_pred_file chief
: <<'END'

END