from __future__ import print_function

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import numpy as np
import importlib
import random
import math


def load_sentences(path, lang, seq_len):
    global processor_module
    processor_module = importlib.import_module('event_extractor.dataprocess.process_{}'.format(lang))
    sentences = []
    with open(path, "r") as f:
        for line in f:
            line_dict = json.loads(line.strip())
            word_sentence, seg_sentence = processor_module.get_seg_features(line_dict["text"])
            if len(word_sentence) > seq_len:
                continue
            line_dict["words"] = word_sentence
            line_dict["segs"] = seg_sentence
            sentences.append(line_dict)
    return sentences


def create_dico(item_list):
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def word_mapping(sentences):
    words = [[x.lower() for x in s["words"]] for s in sentences]
    word_dico = create_dico(words)
    word_dico["<PAD>"] = 100000001
    word_dico['<UNK>'] = 100000000
    word_to_id, id_to_word = create_mapping(word_dico)
    print("Found %i unique words (%i in total)" % (
        len(word_dico), sum(len(x) for x in words)
    ))

    return word_dico, word_to_id, id_to_word


def seg_mapping(sentences):
    segs = [[x.lower() for x in s["segs"]] for s in sentences]
    seg_dico = create_dico(segs)
    seg_dico["<PAD>"] = 100000001
    seg_dico['<UNK>'] = 100000000
    seg_to_id, id_to_seg = create_mapping(seg_dico)
    print("Found %i unique segs (%i in total)" % (
        len(seg_dico), sum(len(x) for x in segs)
    ))
    return seg_dico, seg_to_id, id_to_seg


def load_schema(schema_file):
    id2eventtype, eventtype2id, id2role, role2id = {}, {}, {}, {}
    with open(schema_file, "r") as f:
        for line in f:
            line_dict = json.loads(line)
            event_type = line_dict["event_type"]
            if event_type not in eventtype2id:
                event_type_id = len(eventtype2id)
                id2eventtype[event_type_id] = event_type
                eventtype2id[event_type] = event_type_id

            for role_dict in line_dict["role_list"]:
                role_name = role_dict["role"]
                if role_name not in role2id:
                    role_id = len(role2id)
                    role2id[role_name] = role_id
                    id2role[role_id] = role_name
    return id2eventtype, eventtype2id, id2role, role2id


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def prepare_dataset(sentences, eventtype2id, role2id, word2id, seg2id):
    features = []
    for idx, sentence_info in enumerate(sentences):
        text_words = sentence_info['words']
        text_segs = sentence_info["segs"]
        text = sentence_info["text"]
        s1, s2 = np.zeros((len(text_words), len(eventtype2id))), np.zeros((len(text_words), len(eventtype2id)))
        for event_info in sentence_info["event_list"]:
            trigger_word = event_info["trigger"]
            trigger_word_split, _ = processor_module.get_seg_features(trigger_word)
            event_type = event_info["event_type"]
            trigger_start_index = event_info["trigger_start_index"]
            s1[trigger_start_index][eventtype2id[event_type]] = 1
            s2[trigger_start_index + len(trigger_word_split) - 1][eventtype2id[event_type]] = 1

        if s1.max() == 0:
            continue

        for event_info in sentence_info["event_list"]:
            trigger_word = event_info["trigger"]
            trigger_word_split, _ = processor_module.get_seg_features(trigger_word)
            trigger_start_index = event_info["trigger_start_index"]
            k1 = trigger_start_index
            k2 = trigger_start_index + len(trigger_word_split) - 1

            o1, o2 = np.zeros((len(text_words), len(role2id))), np.zeros((len(text_words), len(role2id)))
            for event_argument in event_info["arguments"]:
                argument_word = event_argument["argument"]
                argument_word_split, _ = processor_module.get_seg_features(argument_word)
                argument_role = event_argument["role"]

                argument_start_index = event_argument["argument_start_index"]

                o1[argument_start_index][role2id[argument_role]] = 1
                o2[argument_start_index + len(argument_word_split) - 1][role2id[argument_role]] = 1
            features.append(
                [text, [word2id.get(word, 1) for word in text_words], [seg2id.get(seg, 1) for seg in text_segs], s1,
                 s2, [k1], [k2], o1, o2])
    return features


class BatchManager(object):
    def __init__(self, data, batch_size, num_events, num_roles, is_sorted=True):
        self.batch_data = self.sort_and_pad(data, batch_size, num_events, num_roles, is_sorted)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size, num_events, num_roles, is_sorted):
        num_batch = int(math.ceil(len(data) / batch_size))
        if is_sorted:
            sorted_data = sorted(data, key=lambda x: len(x[1]))
        else:
            sorted_data = data
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i * batch_size: (i + 1) * batch_size], num_events, num_roles))
        return batch_data

    @staticmethod
    def pad_data(data, num_events, num_roles):
        TEXTS, T1, T2, S1, S2, K1, K2, O1, O2 = [], [], [], [], [], [], [], [], []
        for line in data:
            text, t1, t2, s1, s2, k1, k2, o1, o2 = line
            TEXTS.append(text)
            T1.append(t1)
            T2.append(t2)
            S1.append(s1)
            S2.append(s2)
            K1.append(k1)
            K2.append(k2)
            O1.append(o1)
            O2.append(o2)

        T1 = seq_padding(T1)
        T2 = seq_padding(T2)
        S1 = seq_padding(S1, np.zeros(num_events))
        S2 = seq_padding(S2, np.zeros(num_events))
        O1 = seq_padding(O1, np.zeros(num_roles))
        O2 = seq_padding(O2, np.zeros(num_roles))
        K1, K2 = np.array(K1), np.array(K2)
        return [TEXTS, T1, T2, S1, S2, K1, K2, O1, O2]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]
