import json, codecs, pdb, os

filesfolder = '/home/xuewyang/Xuewen/NLP/data-bin/wmt17_en_de'
voc2ca_de_file = "voc2ca_de.json"

count = {}

categories =  ['$,', '$.', 'ART', 'KON', 'APPR', 'VAFIN', 'KOUS', 'PTKNEG', 'PPER', 'APPRART', 'ADV', 'PRF', 'XY',
              '$[', 'KOKOM', 'NN', 'PDAT', 'VMFIN', 'ADJA', 'PPOSAT', 'PWS', 'NE', 'PIDAT', 'PIS', 'PDS', 'VVFIN',
              'PIAT', 'PROAV', 'VVINF', 'ADJD', 'CARD', 'PWAV', 'VAPP', 'VVPP', 'PWAT', 'PTKVZ', 'KOUI', 'VVIZU',
              'TRUNC', 'APPO', 'VVIMP', 'PTKA', 'PRELS', 'APZR', 'FM', 'PTKANT', 'VMPP', 'ITJ', 'VAIMP', 'PPOSS',
              'VAINF', 'PRELAT']

categories2 = {'APPR', 'ADV', 'XY', 'NN', 'ADJA', 'NE', 'PIS', 'VVFIN', 'PROAV', 'VVINF', 'ADJD', 'CARD': 5144, 'PWAV': 30, 'VAPP': 4, 'VVPP': 8173, 'PWAT': 10, 'PTKVZ': 66, 'KOUI': 2, 'VVIZU': 384, 'TRUNC': 279, 'APPO': 3, 'VVIMP': 23, 'PTKA': 2, 'PRELS': 2, 'APZR': 2, 'FM': 191, 'PTKANT': 2, 'VMPP': 2, 'ITJ': 8, 'VAIMP': 1, 'PPOSS': 2, 'VAINF': 1, 'PRELAT': 1}


for cc in categories:
    count[cc] = 0
with open(os.path.join(filesfolder, voc2ca_de_file), 'r') as ff:
    voc2ca_de = json.load(ff)
    # pdb.set_trace()
    for dd in voc2ca_de.values():
        # pdb.set_trace()
        if dd in categories:
            count[dd] += 1
        else:
            print("error")
            pdb.set_trace()

pdb.set_trace()
print(count)

