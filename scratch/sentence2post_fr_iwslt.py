import json, pdb, os

filesfolder = '/home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.sim.en-fr'
train_file_fr = "/home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.sim.en-fr/train.fr"
test_file_fr = "/home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.sim.en-fr/test.fr"
valid_file_fr = "/home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.sim.en-fr/valid.fr"
voc2ca_fr_file = "voc2ca_fr.json"
train_ca_fr = "train.ca.fr"
test_ca_fr = "test.ca.fr"
valid_ca_fr = "valid.ca.fr"

group1 = ['NC']
group2 = ['P']
group3 = ['PUNC']
group4 = ['DET']
group5 = ['V']
group6 = ['ADJ', 'ADJWH']
group7 = ['ADV']
group8 = ['CLS']
group9 = ['VPP']
group10 = ['VINF']
group11 = ['CC']
group12 = ['N']
group13 = ['PRO']
group14 = ['CS']
group15 = ['VIMP']
group16 = ['PROREL']
group17 = ['ET']
group18 = ['VPR']
group19 = ['CLO']
group20 = ['PREF']
group21 = ['CLR']
group22 = ['ADVWH']
group23 = ['I']
group24 = ['C']
group25 = ['VS']
group26 = ['NPP']
group27 = ['PROWH']
group28 = ['DETWH']

new_dict = {}
for dd in group1:
    new_dict[dd] = 'NC'
for dd in group2:
    new_dict[dd] = 'P'
for dd in group3:
    new_dict[dd] = 'PUNC'
for dd in group4:
    new_dict[dd] = 'DET'
for dd in group5:
    new_dict[dd] = 'V'
for dd in group6:
    new_dict[dd] = 'ADJ'
for dd in group7:
    new_dict[dd] = 'ADV'
for dd in group8:
    new_dict[dd] = 'CLS'
for dd in group9:
    new_dict[dd] = 'VPP'
for dd in group10:
    new_dict[dd] = 'VINF'
for dd in group11:
    new_dict[dd] = 'CC'
for dd in group12:
    new_dict[dd] = 'N'
for dd in group13:
    new_dict[dd] = 'PRO'
for dd in group14:
    new_dict[dd] = 'CS'
for dd in group15:
    new_dict[dd] = 'VIMP'
for dd in group16:
    new_dict[dd] = 'PROREL'
for dd in group17:
    new_dict[dd] = 'ET'
for dd in group18:
    new_dict[dd] = 'VPR'
for dd in group19:
    new_dict[dd] = 'CLO'
for dd in group20:
    new_dict[dd] = 'PREF'
for dd in group21:
    new_dict[dd] = 'CLR'
for dd in group22:
    new_dict[dd] = 'ADVWH'
for dd in group23:
    new_dict[dd] = 'I'
for dd in group24:
    new_dict[dd] = 'C'
for dd in group25:
    new_dict[dd] = 'VS'
for dd in group26:
    new_dict[dd] = 'NPP'
for dd in group27:
    new_dict[dd] = 'PROWH'
for dd in group28:
    new_dict[dd] = 'DETWH'



f_fr_test = open(os.path.join(filesfolder, test_ca_fr), "w")
f_fr_valid = open(os.path.join(filesfolder, valid_ca_fr), "w")

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
                        cas += ('N' + ' ')
                else:
                    for i in range(count):
                        try:
                            cas += (new_dict[voc2ca[whole_word]] + ' ')
                        except KeyError:
                            wrong += 1
                            cas += ('N' + ' ')
                whole_word = ''
                count = 0
        f_towrite.write(cas[:-1] + '\n')

    f.close()
    f_towrite.close()
    print(wrong)

with open(os.path.join(filesfolder, voc2ca_fr_file), 'r') as ff:
    voc2ca_fr = json.load(ff)

sen2post(test_file_fr, voc2ca_fr, f_fr_test)

sen2post(valid_file_fr, voc2ca_fr, f_fr_valid)


f_fr = open(os.path.join(filesfolder, train_ca_fr), "w")

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
                        cas += ('N' + ' ')
                else:
                    for i in range(count):
                        try:
                            cas += (new_dict[voc2ca[whole_word]] + ' ')
                        except KeyError:
                            wrong += 1
                            cas += ('N' + ' ')
                whole_word = ''
                count = 0
        f_towrite.write(cas[:-1] + '\n')

    f.close()
    f_towrite.close()
    print(wrong)

with open(os.path.join(filesfolder, voc2ca_fr_file), 'r') as ff:
    voc2ca_fr = json.load(ff)

sen2post(train_file_fr, voc2ca_fr, f_fr)