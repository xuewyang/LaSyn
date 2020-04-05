import json, pdb

def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

json_1 = "/home/xuewyang/Xuewen/NLP/fairseq/data-bin/wmt17_en_de/voc2ca_de_100000.json"
json_2 = "/home/xuewyang/Xuewen/NLP/fairseq/data-bin/wmt17_en_de/voc2ca_de_200000.json"
json_3 = "/home/xuewyang/Xuewen/NLP/fairseq/data-bin/wmt17_en_de/voc2ca_de_300000.json"
json_4 = "/home/xuewyang/Xuewen/NLP/fairseq/data-bin/wmt17_en_de/voc2ca_de_400000.json"
json_5 = "/home/xuewyang/Xuewen/NLP/fairseq/data-bin/wmt17_en_de/voc2ca_de_416516.json"

json_file = "/home/xuewyang/Xuewen/NLP/data-bin/wmt17_en_de/voc2ca_de.json"


with open(json_1, "r") as f1:
    data1 = json.load(f1)
with open(json_2, "r") as f2:
    data2 = json.load(f2)
with open(json_3, "r") as f3:
    data3 = json.load(f3)
with open(json_4, "r") as f4:
    data4 = json.load(f4)
with open(json_5, "r") as f5:
    data5 = json.load(f5)

voc2ca = Merge(data1, data2)
voc2ca = Merge(data3, voc2ca)
voc2ca = Merge(data4, voc2ca)
voc2ca = Merge(data5, voc2ca)
with open(json_file, 'w') as f:
    json.dump(voc2ca, f)