import numpy as np
import torch
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def save_data(logdir, batch_id, indata, outdata, target, category):
    _, out_predict = torch.max(outdata.data, 1)
    np.save(os.path.join(logdir, 'batch' + str(batch_id) + '_' + category + 'in.npy'),
            np.transpose(indata.data.cpu().numpy()[:, 0, :, :]))
    np.save(os.path.join(logdir, 'batch' + str(batch_id) + '_' + category + 'pred.npy'),
            out_predict.data.cpu().numpy())
    np.save(os.path.join(logdir, 'batch' + str(batch_id) + '_' + category + 'prob.npy'),
            outdata.data.cpu().numpy())
    np.save(os.path.join(logdir, 'batch' + str(batch_id) + '_target.npy'),
            target.data.cpu().numpy())


def mse(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)

def mae(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Mean Squared Error (MSE)"""
    return np.mean(np.abs(gt - pred))

def is_class_correct(target: np.ndarray, pred: np.ndarray):
    return int(target == np.argmax(pred))

def is_class_different(arr1: np.ndarray, arr2: np.ndarray):
    return int(np.argmax(arr2) == np.argmax(arr1))

def cross_entropy(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    ce = torch.nn.CrossEntropyLoss(reduction='none')
    return ce(torch.FloatTensor(np.expand_dims(arr2, 0)), torch.LongTensor([arr1])).data.numpy()[0]

def psnr(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    maxval = gt.max()

    return peak_signal_noise_ratio(gt, pred, data_range=maxval)

def ssim(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""

    maxval = gt.max()
    ssim = 0

    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    return ssim / gt.shape[0]