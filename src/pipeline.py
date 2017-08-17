import numpy as np
import pandas as pd
import glob
import os
import cv2
from tqdm import tqdm

from utilities import *
from io_utils import *
from metric import *


from keras.optimizers import *
from keras.layers import *
from keras.callbacks import *
from keras.models import Model
from keras import backend as K
import tensorflow as tf
import h5py
import gc

from sklearn.model_selection import train_test_split

RS = 17

def get_unet(img_h, img_w):
    inputs = Input((img_h, img_w, 3))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = Concatenate(axis=3)([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    return model

def load_best_weights_min(model, model_name):
    wdir = 'weights_' + str(model_name) + '/'
    if not os.path.exists(wdir):
        os.mkdir(wdir)
    elif len(os.listdir(wdir)) > 0:
        wf = sorted(glob.glob(wdir + '*.h5'))[0]
        model.load_weights(wf)
        print('loaded weights file: ', wf)

def load_best_weights_max(model, model_name):
    wdir = 'weights_' + str(model_name) + '/'
    if not os.path.exists(wdir):
        os.mkdir(wdir)
    elif len(os.listdir(wdir)) > 0:
        wf = sorted(glob.glob(wdir + '*.h5'))[-1]
        model.load_weights(wf)
        print('loaded weights file: ', wf)

def train(model, model_name, img_h, img_w, load_best_weights, n_epochs=100, batch_size=32, patience=5, reduce_rate=0.5):
    df_train = pd.read_csv('../input/train_masks.csv')
    ids_train = df_train['img'].map(lambda s: s.split('.')[0])
    ids_train_split, ids_val_split = train_test_split(ids_train, test_size=0.2, random_state=RS)

    load_best_weights(model, model_name)
    
    callbacks = [
        EarlyStopping(monitor='val_dice_loss',
            patience=patience * 2,
            verbose=1,
            min_delta=1e-4,
            mode='max'),
        ReduceLROnPlateau(monitor='val_dice_loss',
            factor=reduce_rate,
            patience=patience,
            verbose=1,
            epsilon=1e-4,
            mode='max'),
        ModelCheckpoint(monitor='val_dice_loss',
            filepath='weights_' + str(model_name) + '/{val_dice_loss:.6f}-{epoch:03d}.h5',
            save_best_only=True,
            save_weights_only=True,
            mode='max'),
        TensorBoard(log_dir='logs')]

    model.fit_generator(generator=train_generator(ids_train_split, batch_size, img_h, img_w),
        steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
        epochs=n_epochs,
        verbose=1,
        callbacks=callbacks,
        validation_data=val_generator(ids_val_split, batch_size, img_h, img_w),
        validation_steps=np.ceil(float(len(ids_val_split)) / float(batch_size)))

def search_best_threshold(model, model_name, img_h, img_w, load_best_weights, batch_size=32, start_thr=0.1, end_thr=0.51, delta=0.01):
    df_train = pd.read_csv('../input/train_masks.csv')
    ids_train = df_train['img'].map(lambda s: s.split('.')[0])
    ids_train_split, ids_val_split = train_test_split(ids_train, test_size=0.2, random_state=RS)

    load_best_weights(model, model_name)

    best_dice = -1
    best_thr = -1
    for cur_thr in tqdm(np.arange(start_thr, end_thr, delta)):
        y_pred = []
        y_val = []
        for val_batch in data.batch_iterator_val(n_val, batch_size):
            _, X_val_batch, y_val_batch, _ = val_batch
            X_val_batch, y_val_batch = preprocess_batch(X_val_batch, y_val_batch)
            y_val.append(y_val_batch)
            y_pred.append(model.predict_on_batch(X_val_batch))
            
        cur_dice = get_score(np.concatenate(y_val), np.concatenate(y_pred), cur_thr)

        if cur_dice > best_dice:
            print('thr: {}, val dice: {:.5}'.format(cur_thr, cur_dice))
            best_dice = cur_dice
            best_thr = cur_thr
    return best_thr
            
def predict(model, model_name, img_h, img_w, load_best_weights, batch_size=32):
    load_best_weights(model, model_name)
    batch_iterator = test_generator(batch_size, img_h, img_w)

    preds = []
    names = []
    for batch in tqdm(batch_iterator):
        i, X_batch, names_batch = batch
        cur_pred = model.predict_on_batch(X_batch)
        preds.append(cur_pred)
        names.append(names_batch)
        if i % 1000 == 0:
            gc.collect()

    preds = np.concatenate(preds)
    names = np.concatenate(names)

    return preds, names

def make_submission(preds, names, out_file, threshold=0.5):
    rles = []
    for mask in tqdm(preds):
        rles.append(rle(mask > threshold))

    df = pd.DataFrame({'img' : names, 'rle_mask' : rles})
    df.to_csv(out_file, index=False, compression='gzip')

def predict_and_make_submission(model, model_name, img_h, img_w, load_best_weights, out_file, batch_size=32, threshold=0.5):
    load_best_weights(model, model_name)
    batch_iterator = test_generator(batch_size, img_h, img_w)

    rles = []
    names = []
    for batch in tqdm(batch_iterator):
        i, X_batch, names_batch = batch
        cur_pred = model.predict_on_batch(X_batch)
        for mask in cur_pred:
            rles.append(rle(mask > threshold))
        names.append(names_batch)
        gc.collect()

    names = np.concatenate(names)

    df = pd.DataFrame({'img' : names, 'rle_mask' : rles})
    df.to_csv(out_file, index=False, compression='gzip')

def pipeline(model, model_name, img_h, img_w, batch_size=32, patience=10, low_memory=False):
    model.compile(optimizer=Adam(1e-4), loss=bce_dice_loss, metrics=[dice_loss])
    load_weights_func = load_best_weights_max
    # train(model, model_name, img_h, img_w, load_best_weights=load_weights_func, n_epochs=100, batch_size=batch_size, patience=patience, reduce_rate=0.1)
        
    # best_thr = search_best_threshold(model, model_name, img_h, img_w, load_best_weights=load_weights_func,
        # batch_size=batch_size, start_thr=0.1, end_thr=0.5, delta=0.01)
    # preds, names = predict(model, model_name, img_h, img_w, load_best_weights=load_weights_func, batch_size=batch_size)
    # best_thr = 0.5
    # make_submission(preds, names, out_file='../output/' + model_name + '.csv.gz', threshold=best_thr)

    predict_and_make_submission(model, model_name, img_h, img_w, load_best_weights=load_weights_func,
        out_file='../output/' + model_name + '.csv.gz', batch_size=batch_size, threshold=0.5)

    gc.collect()

def model_1():
    img_h, img_w = 128, 128
    model = get_unet(img_h, img_w)
    model_name = 'unet_128'
    pipeline(model, model_name, img_h, img_w, batch_size=32, patience=10)

def model_2():
    img_h, img_w = 480, 720
    model = get_unet(img_h, img_w)
    model_name = 'unet_480_720_2'
    pipeline(model, model_name, img_h, img_w, batch_size=2, patience=3, low_memory=True)

if __name__ == '__main__':
    # model_1()
    model_2()