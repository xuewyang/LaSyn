import json, pdb, os

filesfolder = '/home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.sim.de-en'
train_file_en = "/home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.sim.de-en/train.de-en.en"
test_file_en = "/home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.sim.de-en/test.de-en.en"
valid_file_en = "/home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.sim.de-en/valid.de-en.en"

# filesfolder = '/home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.sim.en-fr'
# train_file_en = "/home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.sim.en-fr/train.en"
# test_file_en = "/home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.sim.en-fr/test.en"
# valid_file_en = "/home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.sim.en-fr/valid.en"

voc2ca_en_file = "voc2ca_en.json"
train_ca_en = "train.ca.en"
test_ca_en = "test.ca.en"
valid_ca_en = "valid.ca.en"

# categories =  ['NN', 'IN', 'NNS', 'DT', 'RB', 'VB', 'PRP', 'JJ', ',', '.', 'CC', 'VBN', 'VBG', 'TO',
#                'VBZ', 'VBD', 'LS', 'CD', 'PRP$', 'MD', ':', 'WRB', 'WP', 'VBP', 'JJR', 'JJS', 'WDT',
#                'NNP', 'UH', 'FW', 'SYM', 'WP$', '$', 'RBR', 'NNPS', '#']


group1 = ['NN', 'FW', 'SYM', '$', '#']
group2 = ['IN']
group3 = ['NNS']
group4 = ['DT']
group5 = ['RB', 'RBR']
group6 = ['VB']
group7 = ['PRP']
group8 = ['JJ']
group9 = [',']
group10 = ['.']
group11 = ['CC', 'UH']
group12 = ['VBN']
group13 = ['VBG']
group14 = ['TO']
group15 = ['VBZ']
group16 = ['VBD']
group17 = ['LS']
group18 = ['CD']
group19 = ['PRP$', 'WP$']
group20 = ['MD']
group21 = [':']
group22 = ['WRB']
group23 = ['WP']
group24 = ['VBP']
group25 = ['JJR']
group26 = ['JJS']
group27 = ['WDT']
group28 = ['NNP', 'NNPS']

new_dict = {}
for dd in group1:
    new_dict[dd] = 'NN'
for dd in group2:
    new_dict[dd] = 'IN'
for dd in group3:
    new_dict[dd] = 'NNS'
for dd in group4:
    new_dict[dd] = 'DT'
for dd in group5:
    new_dict[dd] = 'RB'
for dd in group6:
    new_dict[dd] = 'VB'
for dd in group7:
    new_dict[dd] = 'PRP'
for dd in group8:
    new_dict[dd] = 'JJ'
for dd in group9:
    new_dict[dd] = ','
for dd in group10:
    new_dict[dd] = '.'
for dd in group11:
    new_dict[dd] = 'CC'
for dd in group12:
    new_dict[dd] = 'VBN'
for dd in group13:
    new_dict[dd] = 'VBG'
for dd in group14:
    new_dict[dd] = 'TO'
for dd in group15:
    new_dict[dd] = 'VBZ'
for dd in group16:
    new_dict[dd] = 'VBD'
for dd in group17:
    new_dict[dd] = 'LS'
for dd in group18:
    new_dict[dd] = 'CD'
for dd in group19:
    new_dict[dd] = 'PRP$'
for dd in group20:
    new_dict[dd] = 'MD'
for dd in group21:
    new_dict[dd] = ':'
for dd in group22:
    new_dict[dd] = 'WRB'
for dd in group23:
    new_dict[dd] = 'WP'
for dd in group24:
    new_dict[dd] = 'VBP'
for dd in group25:
    new_dict[dd] = 'JJR'
for dd in group26:
    new_dict[dd] = 'JJS'
for dd in group27:
    new_dict[dd] = 'WDT'
for dd in group28:
    new_dict[dd] = 'NNP'


f_en_test = open(os.path.join(filesfolder, test_ca_en), "w")
f_en_valid = open(os.path.join(filesfolder, valid_ca_en), "w")

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
                        cas += ('NN' + ' ')
                else:
                    for i in range(count):
                        try:
                            cas += (new_dict[voc2ca[whole_word]] + ' ')
                        except KeyError:
                            wrong += 1
                            cas += ('NN' + ' ')
                whole_word = ''
                count = 0
        f_towrite.write(cas[:-1] + '\n')

    f.close()
    f_towrite.close()
    print(wrong)

with open(os.path.join(filesfolder, voc2ca_en_file), 'r') as ff:
    voc2ca_en = json.load(ff)

sen2post(test_file_en, voc2ca_en, f_en_test)

sen2post(valid_file_en, voc2ca_en, f_en_valid)


f_en = open(os.path.join(filesfolder, train_ca_en), "w")

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
                        cas += ('NN' + ' ')
                else:
                    for i in range(count):
                        try:
                            cas += (new_dict[voc2ca[whole_word]] + ' ')
                        except KeyError:
                            wrong += 1
                            cas += ('NN' + ' ')
                whole_word = ''
                count = 0
        f_towrite.write(cas[:-1] + '\n')

    f.close()
    f_towrite.close()
    print(wrong)

with open(os.path.join(filesfolder, voc2ca_en_file), 'r') as ff:
    voc2ca_en = json.load(ff)

sen2post(train_file_en, voc2ca_en, f_en)