import editdistance, torch, pdb

file = "/home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.sim.de-en/train.ca.en"
string = 'IN NNS NNS VBN TO DT JJ IN PRP VBD DT JJ NN ,'
dict_file = "/home/xuewyang/Xuewen/NLP/data-bin/iwslt14.tokenized.sim.de-en/dict.ca.en.txt"

def get_dictionaries(dict_file):
    voc2chr = {}
    chr2voc = {}
    chr2tok = {}
    with open(dict_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            voc, _ = line.split()
            voc2chr[voc] = chr(i + 97)
            chr2voc[chr(i + 97)] = voc
            chr2tok[chr(i + 97)] = i + 4
    return voc2chr, chr2voc, chr2tok


def string_shorten(string, voc2chr, chr2tok):
    tokens = []
    new_string = ''
    for word in string.split():
        token = chr2tok[voc2chr[word]]
        tokens.append(token)
        new_string += voc2chr[word]
    return new_string, tokens


def search_edit(string, distance1, distance2, file):
    # string: the base string
    # distance: edit distance
    # file: the POS tags sequence file - training
    voc2chr, chr2voc, chr2tok = get_dictionaries(dict_file)
    new1, tk1 = string_shorten(string, voc2chr, chr2tok)
    tokens = []
    strings = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            new2, tk2 = string_shorten(line, voc2chr, chr2tok)
            if editdistance.eval(new1, new2) > distance1 and editdistance.eval(new1, new2) <= distance2:
                tokens.append(tk2)
                strings.append(line)
    return tokens, strings


tokens, strings = search_edit(string, 0, 6, file)
pdb.set_trace()