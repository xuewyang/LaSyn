#!/usr/bin/env bash

#TEXT=examples/translation/iwslt14.tokenized.de-en/tmp
#fairseq-preprocess --source-lang de --target-lang en --trainpref $TEXT/train --validpref $TEXT/valid \
#--testpref $TEXT/test --destdir data-bin/iwslt14.tokenized.de-en

#TEXT=examples/translation/iwslt14.tokenized.en-fr
#fairseq-preprocess --source-lang en --target-lang fr --trainpref $TEXT/train --validpref $TEXT/valid \
#--testpref $TEXT/test --destdir /home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.en-fr


## Binarize the dataset:
#TEXT=examples/translation/iwslt14.tokenized.de-en
##TEXT=data-bin/iwslt14.tokenized.de-en
#fairseq-preprocess --source-lang de --target-lang en \
#--trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
#--destdir data-bin/iwslt14.tokenized.de-en --srcdict data-bin/iwslt14.tokenized.de-en/dict.ca.de.txt \
#--tgtdict data-bin/iwslt14.tokenized.de-en/dict.ca.en.txt

# Binarize the dataset:
#TEXT=examples/translation/wmt14_en_de
#fairseq-preprocess --source-lang en --target-lang de \
#--trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
#--destdir /home/xuewyang/Xuewen/NLP/data-bin/wmt14_en_de --thresholdtgt 0 --thresholdsrc 0

#TEXT=/home/xuewyang/Xuewen/NLP/POS_EM/fairseq/examples/translation/wmt17_en_de
#fairseq-preprocess --source-lang en --target-lang de \
#--trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
#--destdir /home/xuewyang/Xuewen/NLP/data-bin/wmt17_en_de --thresholdtgt 0 --thresholdsrc 0

#TEXT=/home/xuewyang/Xuewen/NLP/data-bin/wmt17_en_de
#fairseq-preprocess --source-lang en --target-lang de \
#--trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
#--destdir /home/xuewyang/Xuewen/NLP/data-bin/wmt17_en_de --thresholdtgt 0 --thresholdsrc 0

#TEXT=/home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.en-fr
#fairseq-preprocess --source-lang en --target-lang fr \
#--trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
#--destdir /home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.en-fr --thresholdtgt 0 --thresholdsrc 0

# preprocess for IWSLT EN-DE
TEXT=/home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.sim.de-en
fairseq-preprocess --source-lang de --target-lang en \
--trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
--destdir /home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.sim.de-en --thresholdtgt 0 --thresholdsrc 0

# preprocess for IWSLT EN-FR
#TEXT=/home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.sim.en-fr
#fairseq-preprocess --source-lang en --target-lang fr \
#--trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
#--destdir /home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.sim.en-fr --thresholdtgt 0 --thresholdsrc 0