# get mask z for customized softmax

import numpy as np
import pdb

train_file = "/home/xuewyang/Xuewen/NLP/POS_EM/fairseq/examples/translation/wmt14_en_de/train.de"
train_file_c = "/home/xuewyang/Xuewen/NLP/POS_EM/fairseq/examples/translation/wmt14_en_de/train.ca.de"

dict_file = "/home/xuewyang/Xuewen/NLP/fairseq/data-bin/wmt14_en_de/dict.de.txt"
dict_file_c = "/home/xuewyang/Xuewen/NLP/fairseq/data-bin/wmt14_en_de/dict.ca.de.txt"

# make vocab maps
f_d = open(dict_file, 'r')
f_d_c = open(dict_file_c, 'r')
lines_d = f_d.readlines()
lines_d_c = f_d_c.readlines()

voc_idx = {}
voc_idx_c = {}
i = 0
for l_d in lines_d:
    voc, _ = l_d.split()
    voc_idx[voc] = i + 4
    i += 1

i = 0
for l_d in lines_d_c:
    voc, _ = l_d.split()
    voc_idx_c[voc] = i + 4
    i += 1


mask_z = np.zeros((34240, 56))
mask_z[0, 0] = 1
mask_z[1, 1] = 1
mask_z[2, 2] = 1
mask_z[3, 3] = 1

f = open(train_file, 'r')
f_c = open(train_file_c, 'r')

lines = f.readlines()
lines_c = f_c.readlines()

assert len(lines) == len(lines_c)

for i in range(len(lines)):
    # pdb.set_trace()
    # print(0)
    words = lines[i].split()
    words_c = lines_c[i].split()
    assert len(words) == len(words_c)
    for jj in range(len(words)):
        mask_z[voc_idx[words[jj]], voc_idx_c[words_c[jj]]] = 1

pdb.set_trace()
print(mask_z)