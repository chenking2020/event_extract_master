import numpy as np


def evaluate(event_model, dev_manager):
    event_model.eval()

    A, B, C = 1e-10, 1e-10, 1e-10

    for batch in dev_manager.iter_batch(shuffle=True):
        text, t1, t2, s1, s2, k1, k2, o1, o2 = batch
        ps1_out, ps2_out, pn1_out, pn2_out, t_dgout, mask = event_model.trigger_model_forward(t1, t2)
        po1_out, po2_out = event_model.argument_model_forward(k1, k2, pn1_out, pn2_out, t_dgout, mask)

        s1_pre = ps1_out.detach().numpy()
        s1_pre = np.where(s1_pre > 0.4, 1, 0)
        s1 = s1.astype(np.int)
        A += np.sum(s1 & s1_pre)
        B += np.sum(s1_pre)
        C += np.sum(s1)

        s2_pre = ps2_out.detach().numpy()
        s2_pre = np.where(s2_pre > 0.3, 1, 0)
        s2 = s2.astype(np.int)
        A += np.sum(s2 & s2_pre)
        B += np.sum(s2_pre)
        C += np.sum(s2)

        o1_pre = po1_out.detach().numpy()
        o1_pre = np.where(o1_pre > 0.3, 1, 0)
        o1 = o1.astype(np.int)
        A += np.sum(o1 & o1_pre)
        B += np.sum(o1_pre)
        C += np.sum(o1)

        o2_pre = po2_out.detach().numpy()
        o2_pre = np.where(o2_pre > 0.2, 1, 0)
        o2 = o2.astype(np.int)
        A += np.sum(o2 & o2_pre)
        B += np.sum(o2_pre)
        C += np.sum(o2)

    return 2 * A / (B + C), A / B, A / C
