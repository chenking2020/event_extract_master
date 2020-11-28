import jieba


def get_seg_features(text):
    word_sentence = []
    seg_sentence = []
    for word in jieba.cut(text):
        word = word.lower()
        for ch in word:
            word_sentence.append(ch)
            seg_sentence.append(word)

    return word_sentence, seg_sentence


def joint_output_str(sentence, i, j):
    return sentence[i:j + 1]
