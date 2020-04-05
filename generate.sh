#!/usr/bin/env bash

# generate for triangular
#CUDA_VISIBLE_DEVICES=0 fairseq-generate /home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.de-en \
#--path /home/xuewyang/Xuewen/NLP/checkpoints/transformer_seqz_pos9/checkpoint7.pt \
#--batch-size 100 --beam 1 --remove-bpe \
#--task translation_pos --tri 1

# generate for vanilla transformer de->en
#CUDA_VISIBLE_DEVICES=1 fairseq-generate /home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.de-en \
#--path /home/xuewyang/Xuewen/NLP/checkpoints/transformer/checkpoint72.pt \
#--batch-size 30 --beam 5 --remove-bpe --nbest 10

## generate for vanilla transformer en->de
#CUDA_VISIBLE_DEVICES=0 fairseq-generate /home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.de-en \
#--path /home/xuewyang/Xuewen/NLP/checkpoints/transformer_en_de/checkpoint35.pt \
#--batch-size 50 --beam 5 --remove-bpe -s en -t de --nbest 10

# generate for vanilla lstm-wiseman de->en
#CUDA_VISIBLE_DEVICES=0 fairseq-generate /home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.de-en \
#--path /home/xuewyang/Xuewen/NLP/checkpoints/lstm/checkpoint60.pt \
#--batch-size 100 --beam 5 --remove-bpe

# generate for vanilla lstm-wiseman en->de
#CUDA_VISIBLE_DEVICES=0 fairseq-generate /home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.de-en \
#--path /home/xuewyang/Xuewen/NLP/checkpoints/lstm_en_de/checkpoint48.pt \
#--batch-size 100 --beam 5 --remove-bpe -s en -t de

# generate for vanilla lstm-luong en->de
#CUDA_VISIBLE_DEVICES=1 fairseq-generate /home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.de-en \
#--path /home/xuewyang/Xuewen/NLP/checkpoints/lstm_en_de_luong/checkpoint59.pt \
#--batch-size 30 --beam 5 --remove-bpe -s en -t de

# generate for vanilla lstm-luong de->en
#CUDA_VISIBLE_DEVICES=0 fairseq-generate /home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.de-en \
#--path /home/xuewyang/Xuewen/NLP/checkpoints/lstm_de_en_luong/checkpoint_best.pt \
#--batch-size 20 --beam 5 --remove-bpe -s de -t en

# generate for pos-tagging translation
#CUDA_VISIBLE_DEVICES=0 fairseq-generate /home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.de-en \
#--path /home/xuewyang/Xuewen/NLP/checkpoints/transformer_seqz_l10/checkpoint11.pt \
#--batch-size 100 --beam 1 --remove-bpe \
#--task translation_pos --tri 1

# generate for upper bound
#CUDA_VISIBLE_DEVICES=0 fairseq-generate /home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.de-en \
#--path /home/xuewyang/Xuewen/NLP/checkpoints/transformer_upper/checkpoint13.pt \
#--batch-size 100 --beam 1 --remove-bpe \
#--task translation_pos --tri 1

# generate for nem
#CUDA_VISIBLE_DEVICES=1 fairseq-generate /home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.de-en \
#--path /home/xuewyang/Xuewen/NLP/checkpoints/transformer_nem_lr_0.0001_lambda_0.5_k_3/checkpoint66.pt \
#--batch-size 4 --beam 1 --remove-bpe \
#--task translation_pos --nem 1

# generate en->de conv
#CUDA_VISIBLE_DEVICES=1 fairseq-generate data-bin/iwslt14.tokenized.de-en \
#--path /home/xuewyang/Xuewen/NLP/checkpoints/fconv_en_de/checkpoint77.pt \
#--batch-size 15 --beam 5 --remove-bpe \
#-s en -t de

# generate de->en conv
#CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/iwslt14.tokenized.de-en \
#--path /home/xuewyang/Xuewen/NLP/checkpoints/fconv_de_en/checkpoint60.pt \
#--batch-size 50 --beam 5 --remove-bpe \
#-s de -t en

# generate for rnem
CUDA_VISIBLE_DEVICES=1 fairseq-generate /home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.sim.de-en \
--path /home/xuewyang/Xuewen/NLP/checkpoints/rnem_lr_0.0001_lambda_0.5_k_10_de_en_sim/checkpoint45.pt \
--batch-size 1 --beam 1 --remove-bpe \
--task translation_pos --nem 1 -s de -t en --ed 20  #--nbest 10

# generate for rnem wmt en-de
#CUDA_VISIBLE_DEVICES=0 fairseq-generate /home/xuewyang/Xuewen/NLP/data-bin/wmt17_en_de \
#--path /home/xuewyang/Xuewen/NLP/checkpoints/wmt_en_de/checkpoint_13_440000.pt \
#--batch-size 1 --beam 1 --remove-bpe \
#--task translation_pos --nem 1 -s en -t de


# use this to get perplexity
# math.pow(2, state['extra_state']['train_meters']['valid_loss'].avg)
