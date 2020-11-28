import nltk


def get_seg_features(text):
    word_sentence = []
    seg_sentence = []
    for word in nltk.word_tokenize(text):
        word = word.lower()
        word_sentence.append(word)
        seg_sentence.append(word)

    return word_sentence, seg_sentence


def replace_first_substr(sentence, word):
    start_pos = sentence.index(word)
    start_str = sentence[:start_pos]
    end_str = sentence[start_pos + len(word):]
    return start_str + "-" * len(word) + end_str


def joint_output_str(sentence, i, j):
    word_sentence = []
    for word in nltk.word_tokenize(sentence):
        word_sentence.append(word)
    return " ".join(word_sentence[i:j + 1])
