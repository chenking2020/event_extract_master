from __future__ import print_function

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import importlib
import numpy as np
import re


class EventExtractor(object):
    def __init__(self, lang):
        self.lang = lang
        self.max_word_count = 500

        self.processor_module = importlib.import_module('event_extractor.dataprocess.process_{}'.format(lang))

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        arg_path = os.path.join(BASE_DIR, "checkpoint", "event_params.json")
        map_path = os.path.join(BASE_DIR, "checkpoint", "event_map.json")
        model_path = os.path.join(BASE_DIR, "checkpoint", "event.model")

        with open(arg_path, 'r') as f:
            jd = json.load(f)
        self.params = jd['params']
        with open(map_path, "r") as f:
            self.event_map = json.load(f)

        # self.params["batch_size"] = 1

        event_module = importlib.import_module(
            "event_extractor.model.{}".format(self.params["model_name"]))
        self.event_model = event_module.EventModule(self.params, len(self.event_map["word_to_id"]),
                                                    len(self.event_map["seg_to_id"]),
                                                    len(self.event_map["eventtype2id"]), len(self.event_map["role2id"]))

        self.event_model.load_checkpoint_file(model_path)
        self.event_model.eval()

    def get_event_result(self, input_json):
        sentences = re.split("。|？|！|；|;|,|.|\\?", input_json["text"])
        event_result = {"id": input_json["id"], "event_list": []}

        for sentence in sentences:
            word_sentence, seg_sentence = self.processor_module.get_seg_features(sentence)
            t1 = [[self.event_map["word_to_id"].get(word, 1) for word in word_sentence]]
            t2 = [[self.event_map["seg_to_id"].get(seg, 1) for seg in seg_sentence]]
            ps1_out, ps2_out, pn1_out, pn2_out, t_dgout, mask = self.event_model.trigger_model_forward(t1, t2)
            ps1_out_array = ps1_out.detach().numpy()
            ps2_out_array = ps2_out.detach().numpy()
            k1s = np.where(ps1_out_array > 0.4)[1]
            k2s = np.where(ps2_out_array > 0.3)[1]
            for i in k1s:
                j = k2s[k2s >= i]
                if len(j) > 0:
                    j = j[0]
                    event_info = {}
                    event_info["trigger"] = self.processor_module.joint_output_str(sentence, i, j)
                    k1s_t = ps1_out_array[0][i]
                    k2s_t = ps2_out_array[0][j]
                    k1s_t_max_index = np.argmax(k1s_t)
                    k2s_t_max_index = np.argmax(k2s_t)
                    if k1s_t_max_index == k2s_t_max_index:
                        event_info["event_type"] = self.event_map["id2eventtype"][str(k1s_t_max_index)]
                        event_info["arguments"] = []
                        his_argument = []
                        po1_out, po2_out = self.event_model.argument_model_forward([i], [j], pn1_out, pn2_out,
                                                                                   t_dgout, mask)

                        po1_out_array = po1_out.detach().numpy()
                        po2_out_array = po2_out.detach().numpy()
                        k1o = np.where(po1_out_array > 0.3)[1]
                        k2o = np.where(po2_out_array > 0.2)[1]
                        for m in k1o:
                            n = k2o[k2o >= m]
                            if len(n) > 0:
                                n = n[0]
                                k1o_a = po1_out_array[0][m]
                                k2o_a = po2_out_array[0][n]
                                k1o_a_max_index = np.argmax(k1o_a)
                                k2o_a_max_index = np.argmax(k2o_a)
                                argument_key = "{}_{}".format(m, n)
                                if k1o_a_max_index == k2o_a_max_index and argument_key not in his_argument:
                                    his_argument.append(argument_key)
                                    event_info["arguments"].append(
                                        {"role": self.event_map["id2role"][str(k1o_a_max_index)],
                                         "argument": self.processor_module.joint_output_str(sentence, m, n)})
                        event_result["event_list"].append(event_info)

        return event_result


if __name__ == '__main__':
    model = EventExtractor(lang="zh")
    model = EventExtractor(lang="en")
    # lang = input("input the language:")
    # text = input("input the text:")
    print(model.get_event_result(
        {
            "text": "朱隽表奏了孙坚、刘备的功勋。刘备因朝中无人说情，被封为中山府安喜县县尉。不久，督邮来到安喜。刘备因没有向督邮送钱被督邮陷害。刘备几次到馆驿求见督邮，都被看门人拦住，拒于门外。",
            "id": "0aaab8985832ef9d062eab12ec82f6cf"}))

    print(model.get_event_result(
        {
            "text": "About 500 militants attacked the village of Masteri in the West Darfur state on Saturday afternoon, More than 60 people have been killed and another 60 injured in violence in Sudan's West Darfur region.",
            "id": "0aaab8985832ef9d062eab12ec82f6cf"}))
