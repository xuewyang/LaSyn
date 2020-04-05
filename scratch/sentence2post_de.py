import json, pdb, os

filesfolder = '/home/xuewyang/Xuewen/NLP/data-bin/wmt17_en_de'
train_file_de = "/home/xuewyang/Xuewen/temp/examples/translation/wmt17_en_de/train.de"
test_file_de = "/home/xuewyang/Xuewen/temp/examples/translation/wmt17_en_de/test.de"
valid_file_de = "/home/xuewyang/Xuewen/temp/examples/translation/wmt17_en_de/valid.de"
voc2ca_de_file = "voc2ca_de.json"
train_ca_de = "train.ca.de"
test_ca_de = "test.ca.de"
valid_ca_de = "valid.ca.de"

# categories =  ['$,', '$.', 'ART', 'KON', 'APPR', 'VAFIN', 'KOUS', 'PTKNEG', 'PPER', 'APPRART', 'ADV', 'PRF', 'XY',
#               '$[', 'KOKOM', 'NN', 'PDAT', 'VMFIN', 'ADJA', 'PPOSAT', 'PWS', 'NE', 'PIDAT', 'PIS', 'PDS', 'VVFIN',
#               'PIAT', 'PROAV', 'VVINF', 'ADJD', 'CARD', 'PWAV', 'VAPP', 'VVPP', 'PWAT', 'PTKVZ', 'KOUI', 'VVIZU',
#               'TRUNC', 'APPO', 'VVIMP', 'PTKA', 'PRELS', 'APZR', 'FM', 'PTKANT', 'VMPP', 'ITJ', 'VAIMP', 'PPOSS',
#               'VAINF', 'PRELAT']



group1 = ['ADJA', 'ADJD', ]             # ADJECTIVE
group2 = ['ADV', 'PROAV', 'PWAV', ]                        # ADVERB
group3 = ['APPR', 'APPRART', 'APZR', 'KOUI', 'APPO']    # PREPOSITION, POSTPOSITION
group4 = ['ART']                        # ARTICLE
group5 = ['$,', '$.', '$[']
group6 = ['KON', 'KOKOM', 'KOUS', 'ITJ']                      # conjection, interjection
group7 = ['NE', 'NN', 'TRUNC', 'CARD', 'FM']          # NOUN, cardinal number, FOREIGN WORDS
group8 = ['PDAT', 'PIAT', 'PPOSAT', 'PIDAT', 'PWAT', ]                # demonstrative determiner
group9 = ['PDS', 'PIS', 'PPER', 'PRF', 'PPOSS', 'PRELS', 'PRELAT', 'PWS', ]         # pronoun
group10 = ['PTKA', 'PTKANT', 'PTKNEG', 'PTKVZ']      # particle, prefix
group11 = ['VAFIN', 'VMFIN', 'VVFIN', 'VVINF', 'VAPP', 'VVPP', 'VAIMP', 'VVIMP', 'VMPP', 'VVIZU', 'VAINF', ]      # verb
group12 = ['XY']

new_dict = {}
for dd in group1:
    new_dict[dd] = 'ADJA'
for dd in group2:
    new_dict[dd] = 'ADV'
for dd in group3:
    new_dict[dd] = 'APPR'
for dd in group4:
    new_dict[dd] = 'ART'
for dd in group5:
    new_dict[dd] = '$'
for dd in group6:
    new_dict[dd] = 'KON'
for dd in group7:
    new_dict[dd] = 'NN'
for dd in group8:
    new_dict[dd] = 'PDAT'
for dd in group9:
    new_dict[dd] = 'PDS'
for dd in group10:
    new_dict[dd] = 'PTKA'
for dd in group11:
    new_dict[dd] = 'VAFIN'
for dd in group12:
    new_dict[dd] = 'XY'


f_de_test = open(os.path.join(filesfolder, test_ca_de), "w")
f_de_valid = open(os.path.join(filesfolder, valid_ca_de), "w")

def sen2post(fpath, voc2ca, f_towrite):
    f = open(fpath, 'r')
    wrong = 0
    for ii, line in enumerate(f):
        words = line.rstrip().split(' ')
        at = False
        cas = ''
        whole_word = ''
        count = 0
        for jj, word in enumerate(words):
            if '@@' in word:
                at = True
                count += 1
                whole_word += word[:-2]
            else:
                if at == True:
                    count += 1
                    at = False
                whole_word += word
            if at == False:
                if count == 0:
                    try:
                        cas += (new_dict[voc2ca[whole_word]] + ' ')
                    except KeyError:
                        wrong += 1
                        cas += ('XY' + ' ')
                else:
                    for i in range(count):
                        try:
                            cas += (new_dict[voc2ca[whole_word]] + ' ')
                        except KeyError:
                            wrong += 1
                            cas += ('XY' + ' ')
                whole_word = ''
                count = 0
        f_towrite.write(cas[:-1] + '\n')

    f.close()
    f_towrite.close()
    print(wrong)

with open(os.path.join(filesfolder, voc2ca_de_file), 'r') as ff:
    voc2ca_de = json.load(ff)

sen2post(test_file_de, voc2ca_de, f_de_test)

sen2post(valid_file_de, voc2ca_de, f_de_valid)


f_de = open(os.path.join(filesfolder, train_ca_de), "w")

def sen2post(fpath, voc2ca, f_towrite):
    f = open(fpath, 'r')
    wrong = 0
    for ii, line in enumerate(f):
        words = line.rstrip().split(' ')
        at = False
        cas = ''
        whole_word = ''
        count = 0
        for jj, word in enumerate(words):
            if '@@' in word:
                at = True
                count += 1
                whole_word += word[:-2]
            else:
                if at == True:
                    count += 1
                    at = False
                whole_word += word
            if at == False:
                if count == 0:
                    try:
                        cas += (new_dict[voc2ca[whole_word]] + ' ')
                    except KeyError:
                        wrong += 1
                        cas += ('XY' + ' ')
                else:
                    for i in range(count):
                        try:
                            cas += (new_dict[voc2ca[whole_word]] + ' ')
                        except KeyError:
                            wrong += 1
                            cas += ('XY' + ' ')
                whole_word = ''
                count = 0
        f_towrite.write(cas[:-1] + '\n')

    f.close()
    f_towrite.close()
    print(wrong)

with open(os.path.join(filesfolder, voc2ca_de_file), 'r') as ff:
    voc2ca_de = json.load(ff)

sen2post(train_file_de, voc2ca_de, f_de)