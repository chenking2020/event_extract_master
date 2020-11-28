import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import json


class EventModule(nn.Module):
    def __init__(self, params, word_size, seg_size, num_events, num_roles):
        super(EventModule, self).__init__()
        self.params = params
        self.word_embeds = nn.Embedding(word_size, self.params["emb_dim"])
        self.seg_embeds = nn.Embedding(seg_size, self.params["emb_dim"])
        self.s1_position_embeds = nn.Embedding(self.params["seq_len"], self.params["emb_dim"])
        self.k1_position_embeds = nn.Embedding(self.params["seq_len"], self.params["emb_dim"])
        self.k2_position_embeds = nn.Embedding(self.params["seq_len"], self.params["emb_dim"])
        self.dropout = nn.Dropout(p=self.params["drop_out"])

        self.dgc_m1 = DilatedGatedConv1d(self.params["emb_dim"], 1)
        self.dgc_m2 = DilatedGatedConv1d(self.params["emb_dim"], 2)
        self.dgc_m3 = DilatedGatedConv1d(self.params["emb_dim"], 5)
        self.dgc_m4 = DilatedGatedConv1d(self.params["emb_dim"], 1)
        self.dgc_m5 = DilatedGatedConv1d(self.params["emb_dim"], 2)
        self.dgc_m6 = DilatedGatedConv1d(self.params["emb_dim"], 5)
        self.dgc_m7 = DilatedGatedConv1d(self.params["emb_dim"], 1)
        self.dgc_m8 = DilatedGatedConv1d(self.params["emb_dim"], 2)
        self.dgc_m9 = DilatedGatedConv1d(self.params["emb_dim"], 5)
        self.dgc_m10 = DilatedGatedConv1d(self.params["emb_dim"], 1)
        self.dgc_m11 = DilatedGatedConv1d(self.params["emb_dim"], 1)
        self.dgc_m12 = DilatedGatedConv1d(self.params["emb_dim"], 1)

        self.pn1_dense = Dense(self.params["emb_dim"], word_size)
        self.pn1_out_dense = Dense(word_size, 1, "sigmoid")

        self.pn2_dense = Dense(self.params["emb_dim"], word_size)
        self.pn2_out_dense = Dense(word_size, 1, "sigmoid")

        self.trigger_attention = SelfAttention(8, 16, self.params["emb_dim"])

        self.trigger_h_conv = nn.Conv1d(in_channels=8 * 16 + self.params["emb_dim"], out_channels=word_size,
                                        kernel_size=3, padding=1)

        self.ps_dense = Dense(word_size, 1, "sigmoid")
        self.ps1_dense = Dense(word_size, num_events, "sigmoid")
        self.ps2_dense = Dense(word_size, num_events, "sigmoid")

        self.pc_dense = Dense(self.params["emb_dim"], word_size)
        self.pc_out_dense = Dense(word_size, num_roles, "sigmoid")

        self.k_gru_model = nn.GRU(input_size=self.params["emb_dim"], hidden_size=self.params["emb_dim"],
                                  bidirectional=True,
                                  batch_first=True)

        self.argument_attention = SelfAttention(8, 16, self.params["emb_dim"])

        self.argument_h_conv = nn.Conv1d(in_channels=8 * 16 + self.params["emb_dim"] * 3, out_channels=word_size,
                                         kernel_size=3, padding=1)

        self.po_dense = Dense(word_size, 1, "sigmoid")
        self.po1_dense = Dense(word_size, num_roles, "sigmoid")
        self.po2_dense = Dense(word_size, num_roles, "sigmoid")

        self.s1_loss_model = nn.BCELoss(reduction="none")
        self.s2_loss_model = nn.BCELoss(reduction="none")

        self.o1_loss_model = nn.BCELoss(reduction="none")
        self.o2_loss_model = nn.BCELoss(reduction="none")

        self.word_size = word_size
        self.seg_size = seg_size
        self.num_events = num_events
        self.num_roles = num_roles

        self.batch_size = 1
        self.word_seq_length = 1

    def set_batch_seq_size(self, sentence):
        tmp = sentence.size()
        self.batch_size = tmp[0]
        self.word_seq_length = tmp[1]

    def rand_init_seg_embedding(self):
        nn.init.uniform_(self.seg_embeds.weight, -0.25, 0.25)

    def rand_init_word_embedding(self):
        nn.init.uniform_(self.word_embeds.weight, -0.25, 0.25)

    def rand_init_s1_position_embedding(self):
        nn.init.uniform_(self.s1_position_embeds.weight, -0.25, 0.25)

    def rand_init_k1_position_embedding(self):
        nn.init.uniform_(self.k1_position_embeds.weight, -0.25, 0.25)

    def rand_init_k2_position_embedding(self):
        nn.init.uniform_(self.k2_position_embeds.weight, -0.25, 0.25)

    def set_optimizer(self):
        if self.params["update"] == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.params["lr"], momentum=self.params["momentum"])
        elif self.params["update"] == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.params["lr"])
        return optimizer

    def to_scalar(self, var):
        return var.view(-1).data.tolist()[0]

    def clip_grad_norm(self):
        nn.utils.clip_grad_norm_(self.parameters(), self.params["clip_grad"])

    def adjust_learning_rate(self, optimizer):
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr

    def save_checkpoint(self, state, data_map, params, filename):
        with open(filename + '_map.json', 'w') as f:
            f.write(json.dumps(data_map, ensure_ascii=False))
        with open(filename + '_params.json', 'w') as f:
            f.write(json.dumps(params, ensure_ascii=False))
        torch.save(state, filename + '.model')

    def load_checkpoint_file(self, model_path):
        checkpoint_file = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(checkpoint_file['state_dict'])

    def seq_maxpool(self, x):
        """seq是[None, seq_len, s_size]的格式，
        mask是[None, seq_len, 1]的格式，先除去mask部分，
        然后再做maxpooling。
        """
        seq, mask = x
        seq_out = seq - (1 - mask) * 1e10
        return torch.max(seq_out, 1, keepdims=True)

    def seq_gather(self, x):
        """seq是[None, seq_len, s_size]的格式，
        idxs是[None, 1]的格式，在seq的第i个序列中选出第idxs[i]个向量，
        最终输出[None, s_size]的向量。
        """
        seq, idxs = x
        # batch_idxs = [idxs for _ in range(seq.shape[0])]
        idxs = torch.unsqueeze(idxs.long(), -1)
        batch_idxs = idxs.expand(idxs.shape[0], idxs.shape[1], seq.shape[-1])
        seq_index = torch.gather(seq, 1, batch_idxs)
        seq_index_out = seq_index.view(seq_index.shape[0], seq_index.shape[-1])
        return seq_index_out

    def position_id(self, x):
        if isinstance(x, list) and len(x) == 2:
            x, r = x
        else:
            r = 0

        pid = torch.unsqueeze(torch.arange(x.shape[1]), 0).repeat([x.shape[0], 1])
        return torch.abs(pid - r)

    def trigger_model_forward(self, t1, t2):
        t1 = torch.LongTensor(t1)
        t2 = torch.LongTensor(t2)

        self.set_batch_seq_size(t1)

        mask_temp = torch.unsqueeze(t1, 2)
        mask_temp0 = torch.zeros(mask_temp.size())
        mask_temp1 = torch.ones(mask_temp.size())
        mask = torch.where(mask_temp > 0, mask_temp1, mask_temp0).float()

        pid = self.position_id(t1)
        pv = self.s1_position_embeds(pid)

        t1_embed = self.word_embeds(t1)  # 0: padding, 1: unk
        t2_embed = self.seg_embeds(t2)
        t_embed = t1_embed + t2_embed + pv
        t_dropout = self.dropout(t_embed)

        # t = [t, mask]
        t_mask = t_dropout * mask
        t_dg1 = self.dgc_m1(t_mask, mask)
        t_dg2 = self.dgc_m2(t_dg1, mask)
        t_dg3 = self.dgc_m3(t_dg2, mask)
        t_dg4 = self.dgc_m4(t_dg3, mask)
        t_dg5 = self.dgc_m5(t_dg4, mask)
        t_dg6 = self.dgc_m6(t_dg5, mask)
        t_dg7 = self.dgc_m7(t_dg6, mask)
        t_dg8 = self.dgc_m8(t_dg7, mask)
        t_dg9 = self.dgc_m9(t_dg8, mask)
        t_dg10 = self.dgc_m10(t_dg9, mask)
        t_dg11 = self.dgc_m11(t_dg10, mask)
        t_dgout = self.dgc_m12(t_dg11, mask)

        pn1 = self.pn1_dense(t_dgout)
        pn1_out = self.pn1_out_dense(pn1)

        pn2 = self.pn2_dense(t_dgout)
        pn2_out = self.pn2_out_dense(pn2)

        h = self.trigger_attention([t_dgout, t_dgout, t_dgout, mask])
        h_cat = torch.cat((t_dgout, h), dim=-1)

        h_conv = self.trigger_h_conv(h_cat.permute(0, 2, 1)).permute(0, 2, 1)

        ps = self.ps_dense(h_conv)
        ps1 = self.ps1_dense(h_conv)
        ps2 = self.ps2_dense(h_conv)

        ps1_out = ps * ps1 * pn1_out
        ps2_out = ps * ps2 * pn2_out
        return ps1_out, ps2_out, pn1_out, pn2_out, t_dgout, mask

    def argument_model_forward(self, k1, k2, pn1_out, pn2_out, t_dgout, mask):
        k1 = torch.LongTensor(k1)
        k2 = torch.LongTensor(k2)

        t_max, _ = self.seq_maxpool([t_dgout, mask])
        pc = self.pc_dense(t_max)
        pc_out = self.pc_out_dense(pc)

        def get_k_inter(x, n=6):
            seq, k1, k2 = x
            k_inter = [torch.round(k2 - (k2 - k1) * (a / (n - 1.))) for a in np.arange(n)]
            k_inter_gather = [self.seq_gather([seq, k]) for k in k_inter]
            k_inter_unsqueeze = [torch.unsqueeze(k, 1) for k in k_inter_gather]
            k_inter_out = torch.cat(tuple(k_inter_unsqueeze), 1)
            return k_inter_out

        k = get_k_inter([t_dgout, k1, k2])
        k_gru, _ = self.k_gru_model(k)
        k_last = k_gru[:, -1, :]
        k1v = self.k1_position_embeds(self.position_id([t_dgout, k1]))
        k2v = self.k2_position_embeds(self.position_id([t_dgout, k2]))
        kv = torch.cat((k1v, k2v), dim=-1)
        k_out = torch.unsqueeze(k_last, 1) + kv

        h_o = self.argument_attention([t_dgout, t_dgout, t_dgout, mask])
        h_o_cat = torch.cat((t_dgout, h_o, k_out), dim=-1)
        h_o_conv = self.argument_h_conv(h_o_cat.permute(0, 2, 1)).permute(0, 2, 1)
        po = self.po_dense(h_o_conv)
        po1 = self.po1_dense(h_o_conv)
        po2 = self.po2_dense(h_o_conv)
        po1_out = po * po1 * pc_out * pn1_out
        po2_out = po * po2 * pc_out * pn2_out
        return po1_out, po2_out

    def forward(self, t1, t2, s1, s2, k1, k2, o1, o2):

        s1 = torch.LongTensor(s1)
        s2 = torch.LongTensor(s2)

        o1 = torch.LongTensor(o1)
        o2 = torch.LongTensor(o2)

        ps1_out, ps2_out, pn1_out, pn2_out, t_dgout, mask = self.trigger_model_forward(t1, t2)

        po1_out, po2_out = self.argument_model_forward(k1, k2, pn1_out, pn2_out, t_dgout, mask)

        # s1_expand = torch.unsqueeze(s1, 2)
        # s2_expand = torch.unsqueeze(s2, 2)

        s1_loss = torch.sum(self.s1_loss_model(ps1_out, s1.float()), 2, keepdims=True)
        s1_loss_sum = torch.sum(s1_loss * mask) / torch.sum(mask)
        s2_loss = torch.sum(self.s2_loss_model(ps2_out, s2.float()), 2, keepdims=True)
        s2_loss_sum = torch.sum(s2_loss * mask) / torch.sum(mask)

        o1_loss = torch.sum(self.o1_loss_model(po1_out, o1.float()), 2, keepdims=True)
        o1_loss_sum = torch.sum(o1_loss * mask) / torch.sum(mask)
        o2_loss = torch.sum(self.o2_loss_model(po2_out, o2.float()), 2, keepdims=True)
        o2_loss_sum = torch.sum(o2_loss * mask) / torch.sum(mask)

        loss = (s1_loss_sum + s2_loss_sum) + (o1_loss_sum + o2_loss_sum)

        return loss


class Dense(nn.Module):
    '''
        dense层
    '''

    def __init__(self, input_size, out_size, activation="relu"):
        super(Dense, self).__init__()
        self.linear_layer = nn.Linear(input_size, out_size)
        if activation == "sigmoid":
            self.active_layer = nn.Sigmoid()
        else:
            self.active_layer = nn.ReLU()

    def forward(self, input_tensor):
        linear_result = self.linear_layer(input_tensor)
        return self.active_layer(linear_result)


class DilatedGatedConv1d(nn.Module):
    '''
    膨胀门卷积
    '''

    def __init__(self, dim, dilation_rate):
        super(DilatedGatedConv1d, self).__init__()
        self.dim = dim
        self.conv1d = nn.Conv1d(in_channels=dim, out_channels=dim * 2,
                                kernel_size=3, dilation=dilation_rate, padding=dilation_rate)
        self.dropout_dgc = nn.Dropout(p=0.1)

    def _gate(self, x):
        s, h1 = x
        g1, h_inner = h1[:, :, :self.dim], h1[:, :, self.dim:]
        g2 = self.dropout_dgc(g1)
        g = torch.sigmoid(g2)
        return g * s + (1 - g) * h_inner

    def forward(self, seq, mask):
        convert_seq = seq.permute(0, 2, 1)
        h = self.conv1d(convert_seq).permute(0, 2, 1)
        seq_gate = self._gate([seq, h])
        seq_out = seq_gate * mask
        return seq_out


class SelfAttention(nn.Module):
    """多头注意力机制
    """

    def __init__(self, nb_head, size_per_head, input_size):
        super(SelfAttention, self).__init__()
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.out_dim = nb_head * size_per_head
        q_in_dim = input_size
        k_in_dim = input_size
        v_in_dim = input_size
        self.q_linear = nn.Linear(q_in_dim, self.out_dim)
        self.k_linear = nn.Linear(k_in_dim, self.out_dim)
        self.v_linear = nn.Linear(v_in_dim, self.out_dim)

    def mask(self, x, mask, mode='mul'):
        if mask is None:
            return x
        else:
            for _ in range(x.dim() - mask.dim()):
                mask = torch.unsqueeze(mask, mask.dim())
            if mode == 'mul':
                return x * mask
            else:
                return x - (1 - mask) * 1e10

    def forward(self, inputs):
        q, k, v = inputs[:3]
        v_mask, q_mask = None, None
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变换
        qw1 = self.q_linear(q)
        kw1 = self.k_linear(k)
        vw1 = self.v_linear(v)
        # 形状变换
        qw2 = qw1.reshape(-1, qw1.shape[1], self.nb_head, self.size_per_head)
        kw2 = kw1.reshape(-1, kw1.shape[1], self.nb_head, self.size_per_head)
        vw2 = vw1.reshape(-1, vw1.shape[1], self.nb_head, self.size_per_head)
        # 维度置换
        qw = qw2.permute(0, 2, 1, 3)
        kw = kw2.permute(0, 2, 1, 3)
        vw = vw2.permute(0, 2, 1, 3)
        # Attention
        a1 = torch.matmul(qw, kw.permute(0, 1, 3, 2)) / self.size_per_head ** 0.5
        a2 = a1.permute(0, 3, 2, 1)
        a3 = self.mask(a2, v_mask, 'add')
        a4 = a3.permute(0, 3, 2, 1)
        a = torch.softmax(a4, dim=-1)
        # 完成输出
        o1 = torch.matmul(a, vw)
        o2 = o1.permute(0, 2, 1, 3)
        o3 = o2.reshape(-1, o2.shape[1], self.out_dim)
        o = self.mask(o3, q_mask, 'mul')
        return o
