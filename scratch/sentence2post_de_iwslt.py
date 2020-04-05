import json, pdb, os

filesfolder = '/home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.sim.de-en'
train_file_de = "/home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.sim.de-en/train.de-en.de"
test_file_de = "/home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.sim.de-en/test.de-en.de"
valid_file_de = "/home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.sim.de-en/valid.de-en.de"
voc2ca_de_file = "voc2ca_de.json"
train_ca_de = "train.ca.de"
test_ca_de = "test.ca.de"
valid_ca_de = "valid.ca.de"


group1 = ['XY', 'FM', 'TRUNC']
group2 = ['ADJA']
group3 = ['ART']
group4 = ['ADJD']
group5 = ['APPR', 'APZR', 'APPO']
group6 = ['$,']
group7 = ['ADV']
group8 = ['$.']
group9 = ['PPER']
group10 = ['VVFIN']
group11 = ['VVINF', 'VVIMP', 'VAINF']
group12 = ['VAFIN']
group13 = ['VVPP', 'VAPP', 'VMPP']
group14 = ['KON']
group15 = ['NN']
group16 = ['KOUS']
group17 = ['PPOSAT']
group18 = ['VMFIN']
group19 = ['CARD']
group20 = ['PRF']
group21 = ['PDAT']
group22 = ['APPRART']
group23 = ['NE']
group24 = ['PIS']
group25 = ['PIDAT']
group26 = ['PTKNEG', 'PTKANT', 'PTKA']
group27 = ['PROAV']
group28 = ['KOKOM', 'ITJ']
group29 = ['PWS']
group30 = ['PTKVZ']
group31 = ['PIAT', 'PPOSS']
group32 = ['PWAV']
group33 = ['VVIZU']
group34 = ['PDS', 'PRELS']
group35 = ['$[']
group36 = ['PWAT']

new_dict = {}
for dd in group1:
    new_dict[dd] = 'XY'
for dd in group2:
    new_dict[dd] = 'ADJA'
for dd in group3:
    new_dict[dd] = 'ART'
for dd in group4:
    new_dict[dd] = 'ADJD'
for dd in group5:
    new_dict[dd] = 'APPR'
for dd in group6:
    new_dict[dd] = '$,'
for dd in group7:
    new_dict[dd] = 'ADV'
for dd in group8:
    new_dict[dd] = '$.'
for dd in group9:
    new_dict[dd] = 'PPER'
for dd in group10:
    new_dict[dd] = 'VVFIN'
for dd in group11:
    new_dict[dd] = 'VVINF'
for dd in group12:
    new_dict[dd] = 'VAFIN'
for dd in group13:
    new_dict[dd] = 'VVPP'
for dd in group14:
    new_dict[dd] = 'KON'
for dd in group15:
    new_dict[dd] = 'NN'
for dd in group16:
    new_dict[dd] = 'KOUS'
for dd in group17:
    new_dict[dd] = 'PPOSAT'
for dd in group18:
    new_dict[dd] = 'VMFIN'
for dd in group19:
    new_dict[dd] = 'CARD'
for dd in group20:
    new_dict[dd] = 'PRF'
for dd in group21:
    new_dict[dd] = 'PDAT'
for dd in group22:
    new_dict[dd] = 'APPRART'
for dd in group23:
    new_dict[dd] = 'NE'
for dd in group24:
    new_dict[dd] = 'PIS'
for dd in group25:
    new_dict[dd] = 'PIDAT'
for dd in group26:
    new_dict[dd] = 'PTKNEG'
for dd in group27:
    new_dict[dd] = 'PROAV'
for dd in group28:
    new_dict[dd] = 'KOKOM'
for dd in group29:
    new_dict[dd] = 'PWS'
for dd in group30:
    new_dict[dd] = 'PTKVZ'
for dd in group31:
    new_dict[dd] = 'PIAT'
for dd in group32:
    new_dict[dd] = 'PWAV'
for dd in group33:
    new_dict[dd] = 'VVIZU'
for dd in group34:
    new_dict[dd] = 'PDS'
for dd in group35:
    new_dict[dd] = '$['
for dd in group36:
    new_dict[dd] = 'PWAT'


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