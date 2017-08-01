import numpy as np
import pandas as pd
import glob
import os
import cv2
import h5py

from tqdm import tqdm
import tensorflow as tf
from keras.callbacks import *
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import *

import logging
from scipy.misc import imread, imresize
from dask import delayed, threaded, compute

from io_utils import *

class LRSheduler(Callback):
    def __init__(self, patience=0, reduce_rate=0.5, reduce_rounds=3, verbose=1):
        super(Callback, self).__init__()
        self.patience = patience
        self.wait = 0
        self.best_score = -1.
        self.reduce_rate = reduce_rate
        self.current_reduce_round = 0
        self.reduce_rounds = reduce_rounds
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs):
        current_score = logs.get('val_loss')
        if current_score > self.best_score:
            self.best_score = current_score
            self.wait = 0
            if self.verbose > 0:
                print('---current best val accuracy: %.3f' % current_score)
        else:
            if self.wait >= self.patience:
                self.current_reduce_round += 1
                if self.current_reduce_round <= reduce_rounds:
                    lr = self.model.optimizer.lr.get_value()
                    self.model.optimizer.lr.set_value(lr*self.reduce_rate)
                else:
                    if self.verbose > 0:
                        print("Epoch %d: early stopping" % (epoch))
                    self.model.stop_training = True
            self.wait += 1

def preprocess_batch_patched(X, y=None):
    mid_h = X.shape[1] // 2
    mid_w = X.shape[2] // 2

    X_patches = np.array([
        X[:, :mid_h, :mid_w, :],
        X[:, mid_h:, :mid_w, :],
        X[:, :mid_h, mid_w:, :],
        X[:, mid_h:, mid_w:, :]
    ])

    if y is None:
        return X_patches.astype(np.float32) * 1./255
    else:
        y_patches = np.array([
            y[:, :mid_h, :mid_w, :],
            y[:, mid_h:, :mid_w, :],
            y[:, :mid_h, mid_w:, :],
            y[:, mid_h:, mid_w:, :]
        ])
        return X_patches.astype(np.float32) * 1./255, y_patches.astype(np.float32) * 1./255

def preprocess_batch(X, y=None):
    if y is None:
        return X.astype(np.float32) * 1./255
    else:
        return X.astype(np.float32) * 1./255, y.astype(np.float32) * 1./255
