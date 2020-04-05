from nltk.tag import StanfordPOSTagger
import codecs, time
import pdb, json


st = StanfordPOSTagger('english-bidirectional-distsim.tagger')

categories = []
vocab = [line.split()[0] for line in codecs.open('/home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.en-fr/worddict.en.txt', 'r', 'utf-8').read().splitlines()]
voc2ca = {}


t0 = time.time()
i = 0
for i, voc in enumerate(vocab):
    if i != 0 and i % 1000 == 0:
        print(i, categories, time.time() - t0)
    tag = st.tag([voc])
    if tag[0][1] not in categories:
        categories.append(tag[0][1])
    voc2ca[tag[0][0]] = tag[0][1]

file = "/home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.en-fr/voc2ca_en.json"
with open(file, 'w') as f:
    json.dump(voc2ca, f)

print("categories: ", categories)

# categories:  [',', '.', 'DT', 'CC', 'TO', 'IN', 'LS', 'PRP', 'VBZ', 'NNS',
# 'RB', 'NN', 'VBD', ':', 'VBP', 'VB', 'WP', 'MD', 'CD', 'PRP$', 'WRB', 'VBG',
# 'WDT', 'JJR', 'JJ', 'VBN', 'JJS', 'UH', 'SYM', 'WP$', '$', 'FW', 'NNP', 'RBR',
#  'NNPS', '#']

