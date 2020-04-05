import json, pdb, os

filesfolder = '/home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.en-fr'
worddict_fr = "worddict.fr.txt"
worddict_en = "worddict.en.txt"
train_file_fr = "/home/xuewyang/Xuewen/NLP/POS_EM/fairseq/examples/translation/iwslt14.tokenized.en-fr/train.fr"
train_file_en = "/home/xuewyang/Xuewen/NLP/POS_EM/fairseq/examples/translation/iwslt14.tokenized.en-fr/train.en"
test_file_fr = "/home/xuewyang/Xuewen/NLP/POS_EM/fairseq/examples/translation/iwslt14.tokenized.en-fr/test.fr"
test_file_en = "/home/xuewyang/Xuewen/NLP/POS_EM/fairseq/examples/translation/iwslt14.tokenized.en-fr/test.en"
valid_file_fr = "/home/xuewyang/Xuewen/NLP/POS_EM/fairseq/examples/translation/iwslt14.tokenized.en-fr/valid.fr"
valid_file_en = "/home/xuewyang/Xuewen/NLP/POS_EM/fairseq/examples/translation/iwslt14.tokenized.en-fr/valid.en"
voc2ca_en_file = "voc2ca_en.json"
voc2ca_fr_file = "voc2ca_fr.json"
train_ca_fr = "train.ca.en-fr.fr"
train_ca_en = "train.ca.en-fr.en"
test_ca_fr = "test.ca.en-fr.fr"
test_ca_en = "test.ca.en-fr.en"
valid_ca_fr = "valid.ca.en-fr.fr"
valid_ca_en = "valid.ca.en-fr.en"


f_en_test = open(os.path.join(filesfolder, test_ca_en), "w")
f_fr_test = open(os.path.join(filesfolder, test_ca_fr), "w")
f_en_valid = open(os.path.join(filesfolder, valid_ca_en), "w")
f_fr_valid = open(os.path.join(filesfolder, valid_ca_fr), "w")

def sen2post(fpath, voc2ca, f_towrite, lan="en"):
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
                    wrong += 1
                    try:
                        cas += (voc2ca[whole_word] + ' ')
                    except KeyError:
                        if lan == "en":
                            cas += ('NN' + ' ')
                        else:
                            cas += ('N' + ' ')
                else:
                    wrong += 1
                    for i in range(count):
                        try:
                            cas += (voc2ca[whole_word] + ' ')
                        except KeyError:
                            if lan == "en":
                                cas += ('NN' + ' ')
                            else:
                                cas += ('N' + ' ')
                whole_word = ''
                count = 0
        f_towrite.write(cas[:-1] + '\n')

    f.close()
    f_towrite.close()
    print(wrong)

with open(os.path.join(filesfolder, voc2ca_en_file), 'r') as ff:
    voc2ca_en = json.load(ff)
with open(os.path.join(filesfolder, voc2ca_fr_file), 'r') as ff:
    voc2ca_fr = json.load(ff)

sen2post(test_file_en, voc2ca_en, f_en_test, lan="en")
sen2post(test_file_fr, voc2ca_fr, f_fr_test, lan="fr")

sen2post(valid_file_en, voc2ca_en, f_en_valid, lan="en")
sen2post(valid_file_fr, voc2ca_fr, f_fr_valid, lan="fr")


# f_en = open(os.path.join(filesfolder, train_ca_en), "w")
# f_fr = open(os.path.join(filesfolder, train_ca_fr), "w")
#
# def sen2post(fpath, voc2ca, f_towrite, lan="en"):
#     f = open(fpath, 'r')
#     for ii, line in enumerate(f):
#         words = line.rstrip().split(' ')
#         at = False
#         cas = ''
#         whole_word = ''
#         count = 0
#         for jj, word in enumerate(words):
#             if '@@' in word:
#                 at = True
#                 count += 1
#                 whole_word += word[:-2]
#             else:
#                 if at == True:
#                     count += 1
#                     at = False
#                 whole_word += word
#             if at == False:
#                 if count == 0:
#                     cas += (voc2ca[whole_word] + ' ')
#                 else:
#                     for i in range(count):
#                         cas += (voc2ca[whole_word] + ' ')
#                 whole_word = ''
#                 count = 0
#         f_towrite.write(cas[:-1] + '\n')
#
#     f.close()
#     f_towrite.close()
#
# with open(os.path.join(filesfolder, voc2ca_en_file), 'r') as ff:
#     voc2ca_en = json.load(ff)
# with open(os.path.join(filesfolder, voc2ca_fr_file), 'r') as ff:
#     voc2ca_fr = json.load(ff)
#
# sen2post(train_file_en, voc2ca_en, f_en, lan="en")
# sen2post(train_file_fr, voc2ca_fr, f_fr, lan="fr")