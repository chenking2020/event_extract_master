from __future__ import print_function

import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from event_extractor.dataprocess import data_loader
from event_extractor.train.eval import evaluate
import importlib
import mlflow
import time


class TrainProcess(object):
    def __init__(self, params):
        self.params = params
        mlflow.set_tracking_uri(
            "file://{}/runmodels".format(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
        mlflow.set_experiment("event_task")

    def load_data(self):
        # ToDo 暂时从本地读取文件，以后改成从库中读取，暂时按照本地文件分训练、验证和测试，以后改成自动切分
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data",
                                 self.params["lang"])
        self.all_train_sentences = data_loader.load_sentences(os.path.join(data_path, "train.json"),
                                                              self.params["lang"], self.params["seq_len"])
        self.all_dev_sentences = data_loader.load_sentences(os.path.join(data_path, "dev.json"), self.params["lang"],
                                                            self.params["seq_len"])

        _w, self.word_to_id, self.id_to_word = data_loader.word_mapping(self.all_train_sentences)
        _s, self.seg_to_id, self.id_to_seg = data_loader.seg_mapping(self.all_train_sentences)

        self.id2eventtype, self.eventtype2id, self.id2role, self.role2id = data_loader.load_schema(
            os.path.join(data_path, "event_schema.json"))

        train_data = data_loader.prepare_dataset(self.all_train_sentences, self.eventtype2id, self.role2id,
                                                 self.word_to_id, self.seg_to_id)
        dev_data = data_loader.prepare_dataset(self.all_dev_sentences, self.eventtype2id, self.role2id,
                                               self.word_to_id, self.seg_to_id)

        self.train_manager = data_loader.BatchManager(train_data, self.params["batch_size"], len(self.eventtype2id),
                                                      len(self.role2id), is_sorted=True)
        self.dev_manager = data_loader.BatchManager(dev_data, self.params["batch_size"], len(self.eventtype2id),
                                                    len(self.role2id), is_sorted=True)

    def train(self):
        with mlflow.start_run(run_name=self.params["task_name"]):
            mlflow.log_params(self.params)
            model_path = mlflow.get_artifact_uri()
            model_path = model_path[model_path.index("/"):]

            event_module = importlib.import_module(
                "event_extractor.model.{}".format(self.params["model_name"]))
            event_model = event_module.EventModule(self.params, len(self.word_to_id), len(self.seg_to_id),
                                                   len(self.eventtype2id), len(self.role2id))
            event_model.rand_init_word_embedding()
            event_model.rand_init_seg_embedding()
            event_model.rand_init_s1_position_embedding()
            event_model.rand_init_k1_position_embedding()
            event_model.rand_init_k2_position_embedding()

            optimizer = event_model.set_optimizer()
            # tot_length = len(self.all_train_sentences)
            print("has train data: {}".format(len(self.all_train_sentences)))
            print("has dev data: {}".format(len(self.all_dev_sentences)))

            best_f1 = float('-inf')
            best_f1_epoch = 0
            start_time = time.time()
            patience_count = 0

            for epoch_idx in range(self.params["epoch"]):
                event_model.train()
                print("-------------------------------------------------------------------------------------")
                epoch_loss = 0
                iter_step = 0
                for batch in self.train_manager.iter_batch(shuffle=True):
                    text, t1, t2, s1, s2, k1, k2, o1, o2 = batch
                    iter_step += 1
                    step_start_time = time.time()
                    event_model.zero_grad()
                    loss = event_model(t1, t2, s1, s2, k1, k2, o1, o2)
                    epoch_loss += event_model.to_scalar(loss)
                    loss.backward()
                    # event_model.clip_grad_norm()
                    optimizer.step()
                    print("epoch: %s, current step: %s, current loss: %.4f time use: %s" % (
                        epoch_idx, iter_step, loss / len(t1),
                        time.time() - step_start_time))
                epoch_loss /= iter_step
                mlflow.log_metric("loss", epoch_loss)
                # update lr
                event_model.adjust_learning_rate(optimizer)
                for param_group in optimizer.param_groups:
                    mlflow.log_metric("learning_rate", param_group['lr'])

                f1, p, r = evaluate(event_model, self.dev_manager)
                mlflow.log_metric("f1", f1)
                print("dev: f1: {}, p: {}, r: {}".format(f1, p, r))

                if f1 >= best_f1:
                    best_f1 = f1
                    best_f1_epoch = epoch_idx
                    patience_count = 0
                    print('best average f1: %.4f in epoch_idx: %d , saving...' % (best_f1, best_f1_epoch))

                    try:
                        event_model.save_checkpoint({
                            'epoch': epoch_idx,
                            'state_dict': event_model.state_dict(),
                            'optimizer': optimizer.state_dict()}, {
                            'word_to_id': self.word_to_id,
                            'id_to_word': self.id_to_word,
                            'seg_to_id': self.seg_to_id,
                            'id_to_seg': self.id_to_seg,
                            "id2eventtype": self.id2eventtype,
                            "eventtype2id": self.eventtype2id,
                            "id2role": self.id2role,
                            "role2id": self.role2id
                        }, {'params': self.params},
                            os.path.join(model_path, 'event'))
                    except Exception as inst:
                        print(inst)
                else:
                    patience_count += 1
                    print(
                        'poor current average f1: %.4f, best average f1: %.4f in epoch_idx: %d' % (
                            f1, best_f1, best_f1_epoch))

                print('epoch: ' + str(epoch_idx) + '\t in ' + str(self.params["epoch"]) + ' take: ' + str(
                    time.time() - start_time) + ' s')

                if patience_count >= self.params["patience"] and epoch_idx >= self.params["least_iters"]:
                    mlflow.end_run()
                    break


if __name__ == '__main__':
    # train_d = TrainProcess(
    #     {"task_name": "event_test", "lang": "zh", "model_name": "dgcnn", "batch_size": 32, "epochs": 200,
    #      "seq_len": 500, "emb_dim": 128, "drop_out": 0.25,
    #      "update": "adam", "lr": 0.0001, "lr_decay": 0.05, "clip_grad": 5, "epoch": 500, "patience": 15,
    #      "least_iters": 50, "gpu": -1})
    # train_d.load_data()
    # train_d.train()

    train_d = TrainProcess(
        {"task_name": "event_test", "lang": "en", "model_name": "dgcnn", "batch_size": 32, "epochs": 200,
         "seq_len": 500, "emb_dim": 128, "drop_out": 0.25,
         "update": "adam", "lr": 0.0001, "lr_decay": 0.05, "clip_grad": 5, "epoch": 500, "patience": 15,
         "least_iters": 50, "gpu": -1})
    train_d.load_data()
    train_d.train()
