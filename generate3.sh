#!/usr/bin/env bash

# generate for rnem

CUDA_VISIBLE_DEVICES=0 fairseq-generate /home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.sim.de-en \
--path /home/xuewyang/Xuewen/NLP/checkpoints/rnem_lr_0.0001_lambda_0.5_k_10_de_en_sim/checkpoint45.pt \
--batch-size 1 --beam 1 --remove-bpe \
--task translation_pos --nem 1 -s de -t en --ed 30  #--nbest 10

# RB NNS DT NN IN DT NN IN PRP TO VB RB CC IN DT JJR CC JJR NN IN JJ NN NNS