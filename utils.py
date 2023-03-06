import numpy as np
import torch
import os

def save_data(logdir, batch_id, indata, outdata, target, category):
    _, out_predict = torch.max(outdata.data, 1)
    np.save(os.path.join(logdir, 'batch' + str(batch_id) + '_' + category + 'in.npy'),
            np.transpose(indata.data.cpu().numpy()[:, 0, :, :], (1, 2, 0)))
    np.save(os.path.join(logdir, 'batch' + str(batch_id) + '_' + category + 'pred.npy'),
            out_predict.data.cpu().numpy())
    np.save(os.path.join(logdir, 'batch' + str(batch_id) + '_' + category + 'prob.npy'),
            outdata.data.cpu().numpy())
    np.save(os.path.join(logdir, 'batch' + str(batch_id) + '_target.npy'),
            target.data.cpu().numpy())