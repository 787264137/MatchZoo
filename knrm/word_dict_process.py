with open('word_dict.txt', 'r', encoding='utf-8') as f:
    with open('word_dict_userdict.txt', 'w', encoding='utf-8') as fw:
        while True:
            line = f.readline()
            if not line: break
            word, word_dig = line.split(':')
            print(word)
            fw.write(word)
            fw.write('\n')
