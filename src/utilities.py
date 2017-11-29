import numpy as np
import pandas as pd
import glob
import os
import cv2
from tqdm import tqdm

from io_utils import *
from metric import *
from models import *

from keras.optimizers import *
from keras.layers import *
from keras.callbacks import *
from keras.models import Model
from keras import backend as K
import tensorflow as tf
import h5py
import gc

from sklearn.model_selection import train_test_split

def search_best_threshold(model, model_name, img_h, img_w, load_best_weights, batch_size=32, start_thr=0.3, end_thr=0.71, delta=0.01):
    df_train = pd.read_csv('../input/train_masks.csv')
    ids_train = df_train['img'].map(lambda s: s.split('.')[0])
    ids_train_split, ids_val_split = train_test_split(ids_train, test_size=0.2, random_state=RS)
    print (ids_train_split.shape, ids_val_split.shape)

    load_best_weights(model, model_name)
   
    y_pred = []
    y_val = []
    cur_batch = 0
    for val_batch in val_generator(ids_val_split, batch_size, img_h, img_w):
        X_val_batch, y_val_batch = val_batch
        y_val.append(y_val_batch)
        y_pred.append(model.predict_on_batch(X_val_batch))
        cur_batch += 1
        if cur_batch > ids_val_split.shape[0]:
            break

    y_val, y_pred = np.concatenate(y_val), np.concatenate(y_pred)

    best_dice = -1
    best_thr = -1

    for cur_thr in tqdm(np.arange(start_thr, end_thr, delta)):
        cur_dice = get_score(y_val, y_pred, cur_thr)
        if cur_dice > best_dice:
            print('thr: {}, val dice: {:.5}'.format(cur_thr, cur_dice))
            best_dice = cur_dice
            best_thr = cur_thr
    return best_thr
            
def search_alpha_itr(preds_1, preds_2, y_val):
    best_alpha = -1
    best_score = -1
    for alpha in tqdm(np.arange(0, 1.01, 0.01)):
        y_pred = np.zeros(preds_1.shape)
        for i in range(len(y_pred)):
            y_pred[i] = alpha * preds_1[i] + (1 - alpha) * preds_2[i]
        cur_score = get_score(y_val, y_pred, 0.5)
        if cur_score > best_score:
            print('alpha: {}, val dice: {:.5}'.format(alpha, cur_score))
            best_score = cur_score
            best_alpha = alpha
    print('BEST: alpha: {}, val dice: {:.5}'.format(alpha, cur_score))
    return best_alpha

def search_alphas(models, model_names, img_h, img_w, batch_size=32):
    df_train = pd.read_csv('../input/train_masks.csv')
    ids_train = df_train['img'].map(lambda s: s.split('.')[0])
    ids_train_split, ids_val_split = train_test_split(ids_train, test_size=0.2, random_state=RS)
    print (ids_train_split.shape, ids_val_split.shape)
   
    models_preds = []

    for i in tqdm(range(len(models))):
        if i == 0:
            y_val = []
        y_pred = []
        cur_batch = 0
        for val_batch in val_generator(ids_val_split, batch_size, img_h[i], img_w[i]):
            X_val_batch, y_val_batch = val_batch
            if i == 0:
                y_val.append(y_val_batch)
            y_pred.append(models[i].predict_on_batch(X_val_batch))
            cur_batch += 1
            if cur_batch > ids_val_split.shape[0] / 2:
                break
        if i == 0:
            y_val = np.concatenate(y_val)
        y_pred = np.concatenate(y_pred)
        models_preds.append(y_pred)
    
    alphas = np.ones(len(models))
    preds_1 = models_preds[0]
    for i in range(1, len(models)):
        preds_2 = models_preds[i]
        alpha = search_alpha_itr(preds_1, preds_2, y_val)
        for j in range(i):
            alphas[j] *= alpha
        alphas[i] *= (1 - alpha)

        preds_1 = alpha * preds_1 + (1 - alpha) * preds_2

    preds = np.average(models_preds, weights=alphas, axis=0)
    print('val dice {}; alphas: {}'.format(get_score(preds, y_val, 0.5), alphas))

    return alphas
