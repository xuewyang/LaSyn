import json, codecs, pdb, os

# filesfolder = '/home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.en-fr'
# voc2ca_en_file = "voc2ca_en.json"
# voc2ca_fr_file = "voc2ca_fr.json"
#
# voc2ca_en_file2 = "voc2ca_en2.json"
# voc2ca_fr_file2 = "voc2ca_fr2.json"
#
# with open(os.path.join(filesfolder, voc2ca_en_file), 'r') as ff:
#     voc2ca_en = json.load(ff)
# with open(os.path.join(filesfolder, voc2ca_fr_file), 'r') as ff:
#     voc2ca_de = json.load(ff)
#
# pdb.set_trace()
# voc2ca_en['_'] = 'SYM'
# voc2ca_de['_'] = 'XY'
#
#
# with open(os.path.join(filesfolder, voc2ca_en_file2), 'w') as f:
#     json.dump(voc2ca_en, f)
#
# with open(os.path.join(filesfolder, voc2ca_fr_file2), 'w') as f:
#     json.dump(voc2ca_de, f)

filesfolder = '/home/xuewyang/Xuewen/NLP/data-bin/wmt17_en_de'
voc2ca_de_file = "voc2ca_de.json"

with open(os.path.join(filesfolder, voc2ca_de_file), 'r') as ff:
    voc2ca_de = json.load(ff)

voc2ca_de['_'] = 'XY'

voc2ca_en_file2 = "voc2ca_de2.json"
with open(os.path.join(filesfolder, voc2ca_en_file2), 'w') as f:
    json.dump(voc2ca_de, f)
