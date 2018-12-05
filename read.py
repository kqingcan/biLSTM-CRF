import numpy as np


word_map = {}


def read_train_file(filename):
    words = []
    with open(filename, 'rt', encoding='utf8') as file:
        for line in file.readlines():
            if len(line.split(' ')) == 2:
                word, tag = line.split(' ')
                words.append(word)
    return words


def read_pre_train_file(filename):
    with open(filename, 'r', encoding='utf8') as file:
        for line in file.readlines():

            # print(line)
            line_array = line.split(" ")
            word = line_array[0]
            word_vec = []
            line_array.pop(0)
            for num in line_array:
                num = float(num)
                word_vec.append(num)
            # print(word_vec)
            word_map[word] = word_vec


read_pre_train_file("data/wiki.zh.text.simplified.character.txt")
print("finished")
all_words = read_train_file("data/train.utf8")
train_word_map = {}
for word in all_words:
    if word in word_map:
        vec = word_map[word]
        train_word_map[word] = word_map[word]
with open('train.txt', 'w', encoding='utf8') as file:
    for word in train_word_map:
        vec = train_word_map[word]
        temp = ""
        temp = temp + word
        for num in vec:
            temp = temp + " " + str(num)
        file.write(temp+"\n")





