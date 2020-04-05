#!/usr/bin/env bash

# train vanilla transformer en->de
#CUDA_VISIBLE_DEVICES=0 fairseq-train /home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.de-en \
#-a transformer_iwslt_de_en --optimizer adam --lr 0.0001 -s en -t de \
#--label-smoothing 0.1 --dropout 0.3 --max-tokens 600 \
#--min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
#--criterion label_smoothed_cross_entropy --max-update 1000000 \
#--warmup-updates 4000 --warmup-init-lr '1e-07' \
#--adam-betas '(0.9, 0.98)' --save-dir /home/xuewyang/Xuewen/NLP/checkpoints/transformer_en_de


# train vanilla transformer de->en
#CUDA_VISIBLE_DEVICES=0 fairseq-train /home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.de-en \
#-a transformer_iwslt_de_en --optimizer adam --lr 0.0001 -s de -t en \
#--label-smoothing 0.1 --dropout 0.3 --max-tokens 850 \
#--min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
#--criterion label_smoothed_cross_entropy --max-update 1000000 \
#--warmup-updates 4000 --warmup-init-lr '1e-07' \
#--adam-betas '(0.9, 0.98)' --save-dir /home/xuewyang/Xuewen/NLP/checkpoints/transformer


# Train NEM en->de
CUDA_VISIBLE_DEVICES=0 fairseq-train /home/xuewyang/Downloads/iwslt14.tokenized.sim.de-en \
-a transformer_nem_iwslt_de_en --optimizer adam --lr 0.0001 -s en -t de \
--label-smoothing 0.1 --dropout 0.3 --max-tokens 400 \
--min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy_nem --max-update 500000 \
--warmup-updates 4000 --warmup-init-lr '1e-07' \
--adam-betas '(0.9, 0.98)' --save-dir /home/xuewyang/Xuewen/NLP/checkpoints/transformer_nem_lr_0.0001_lambda_0.5_k_1_en_de_mask \
--task translation_pos

# Train NEM de->en
#fairseq-train /home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.de-en \
#-a transformer_nem_iwslt_de_en --optimizer adam --lr 0.0001 -s de -t en \
#--label-smoothing 0.1 --dropout 0.3 --max-tokens 1010 \
#--min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
#--criterion label_smoothed_cross_entropy_nem --max-update 500000 \
#--warmup-updates 4000 --warmup-init-lr '1e-07' \
#--adam-betas '(0.9, 0.98)' --save-dir /home/xuewyang/Xuewen/NLP/checkpoints/transformer_nem_lr_0.0001_lambda_0.5_k_5 \
#--task translation_pos


# Train RNEM de->en no regularization
#CUDA_VISIBLE_DEVICES='0' fairseq-train /home/xuewyang/Downloads/iwslt14.tokenized.sim.de-en \
#-a transformer_rnem_iwslt_de_en --optimizer adam --lr 0.0001 -s en -t de \
#--label-smoothing 0.1 --dropout 0.3 --max-tokens 1030 \
#--min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
#--criterion label_smoothed_cross_entropy_nem_noreg --max-update 500000 \
#--warmup-updates 4000 --warmup-init-lr '1e-07' \
#--adam-betas '(0.9, 0.98)' --save-dir /home/xuewyang/Xuewen/NLP/checkpoints/rnem_lr_0.0001_lambda_0.5_k_1_en_de_sim_noreg \
#--task translation_pos

