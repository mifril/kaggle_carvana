import numpy as np
from keras import backend as K

def dice(im1, im2, empty_score=1.0):
    im1 = im1.astype(np.bool)
    im2 = im2.astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / im_sum

def get_score(train_masks, preds, thr=0.5):
    d = 0.0

    for i in range(train_masks.shape[0]):
        d += dice(train_masks[i], preds[i] > thr)
    return d / train_masks.shape[0]

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-12) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1e-12)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
