import pdb

file = "/home/xuewyang/Xuewen/NLP/temp/edit30.txt"
# 0.242
def distinct1(result_file):
    scores = []
    with open(result_file, "r") as f:
        lines = f.readlines()
        count = 0
        num_count = 0
        for line in lines:
            if line[0] != 'H':
                continue
            else:
                if count % 5 == 0:
                    if count == 5:
                        score = float(len(word_set)) / float(num_count)
                        scores.append(score)
                    count = 0
                    num_count = 0
                    word_set = set()
                words = line.split()
                words = words[2:]
                word_set.update(words)
                num_count += len(words)
                count += 1
    return scores

scores = distinct1(file)
print(sum(scores) / len(scores))