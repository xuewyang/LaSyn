import json

voc2ca_en_file = "/home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.sim.de-en/voc2ca_en.json"
string = 'and of course design adds influence on perception too .'

with open(voc2ca_en_file, 'r') as f:
    data  = json.load(f)

for word in string.split():
    print(data[word])

# CC IN VBZ NN NN IN PRP NN NNS , JJ IN NN NN .