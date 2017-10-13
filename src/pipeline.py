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
            min_delta=1e-6,
            mode='max'),
        ReduceLROnPlateau(monitor='val_dice_loss',
            factor=reduce_rate,
            patience=patience,
            verbose=1,
            epsilon=1e-6,
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
            
def predict(model, model_name, img_h, img_w, load_best_weights, batch_size=32):
    load_best_weights(model, model_name)
    batch_iterator = test_generator(batch_size, img_h, img_w)

    preds = []
    names = []
    for batch in tqdm(batch_iterator):
        i, X_batch, names_batch = batch
        cur_pred = model.predict_on_batch(X_batch)
        if tta:
            X_batch_flip = np.array([cv2.flip(image, 1) for image in X_batch])
            # print (X_batch.shape, X_batch_flip.shape)
            cur_pred_flip = model.predict_on_batch(X_batch_flip)
            # print (models_preds.shape, models_preds_flip.shape)
            cur_pred_flip = np.array([cv2.flip(image, 1).reshape(image.shape) for image in cur_pred_flip])
            # print (models_preds.shape, models_preds_flip.shape)
            cur_pred = 0.5 * cur_pred + 0.5 * cur_pred_flip
        
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

def predict_and_make_submission(model, model_name, img_h, img_w, load_best_weights, out_file, batch_size=32, threshold=0.5, tta=True):
    load_best_weights(model, model_name)
    batch_iterator = test_generator(batch_size, img_h, img_w)

    rles = []
    names = []
    for batch in tqdm(batch_iterator):
        i, X_batch, names_batch = batch
        cur_pred = model.predict_on_batch(X_batch)
        if tta:
            X_batch_flip = np.array([cv2.flip(image, 1) for image in X_batch])
            # print (X_batch.shape, X_batch_flip.shape)
            cur_pred_flip = model.predict_on_batch(X_batch_flip)
            # print (cur_pred.shape, cur_pred_flip.shape)
            cur_pred_flip = np.array([cv2.flip(image, 1).reshape(image.shape) for image in cur_pred_flip])
            # print (cur_pred.shape, cur_pred_flip.shape)
            cur_pred = 0.5 * cur_pred + 0.5 * cur_pred_flip
        # print (cur_pred.shape)
        for mask in cur_pred:
            rles.append(rle(mask > threshold))
        names.append(names_batch)
        if i % 1000 == 0:
            gc.collect()

    names = np.concatenate(names)

    df = pd.DataFrame({'img' : names, 'rle_mask' : rles})
    df.to_csv(out_file, index=False, compression='gzip')

def pipeline(model, model_name, img_h, img_w, batch_size=32, patience=10, low_memory=False):
    load_weights_func = load_best_weights_max
    # train(model, model_name, img_h, img_w, load_best_weights=load_weights_func, n_epochs=100, batch_size=batch_size, patience=patience, reduce_rate=0.1)
        
    # best_thr = search_best_threshold(model, model_name, img_h, img_w, load_best_weights=load_weights_func,
    #     batch_size=batch_size, start_thr=0.3, end_thr=0.81, delta=0.01)
    # print (best_thr)
    # preds, names = predict(model, model_name, img_h, img_w, load_best_weights=load_weights_func, batch_size=batch_size)
    # best_thr = 0.5
    # make_submission(preds, names, out_file='../output/' + model_name + '.csv.gz', threshold=best_thr)

    # predict_and_make_submission(model, model_name, img_h, img_w, load_best_weights=load_weights_func,
    #     out_file='../output/' + model_name + '_{}.csv.gz'.format(best_thr), batch_size=batch_size, threshold=best_thr)
    predict_and_make_submission(model, model_name, img_h, img_w, load_best_weights=load_weights_func,
        out_file='../output/' + model_name + '_05.csv.gz', batch_size=batch_size, threshold=0.5)

    gc.collect()

def model_1():
    img_h, img_w = 128, 128
    model = get_unet(img_h, img_w)
    model_name = 'unet_128'
    model.compile(optimizer=Adam(1e-4), loss=bce_dice_loss, metrics=[dice_loss])
    pipeline(model, model_name, img_h, img_w, batch_size=32, patience=10)

def model_2():
    img_h, img_w = 480, 720
    model = get_unet(img_h, img_w)
    model_name = 'unet_480_720'
    model.compile(optimizer=Adam(1e-4), loss=bce_dice_loss, metrics=[dice_loss])
    pipeline(model, model_name, img_h, img_w, batch_size=2, patience=3, low_memory=True)

def model_3():
    img_h, img_w = 800, 1280
    model = get_unet(img_h, img_w)
    model_name = 'unet_1280_800'
    model.compile(optimizer=Adam(1e-4), loss=bce_dice_loss, metrics=[dice_loss])
    best_thr = 0.4
    pipeline(model, model_name, img_h, img_w, batch_size=1, patience=3, low_memory=True)

def model_4():
    img_h, img_w = 1024, 1024
    model = get_unet(img_h, img_w)
    model_name = 'unet_1024_1024'
    model.compile(optimizer=Adam(1e-4), loss=bce_dice_loss, metrics=[dice_loss])
    pipeline(model, model_name, img_h, img_w, batch_size=1, patience=3, low_memory=True)

def model_5():
    img_h, img_w = 832, 1280
    model = get_unet(img_h, img_w)
    model_name = 'unet_1280_832'
    model.compile(optimizer=Adam(1e-5), loss=bce_dice_loss, metrics=[dice_loss])
    best_thr = 0.5
    pipeline(model, model_name, img_h, img_w, batch_size=1, patience=3, low_memory=True)

def model_6():
    img_h, img_w = 800, 1280
    model = get_unet(img_h, img_w)
    model_name = 'unet_1280_800_w'
    model.compile(optimizer=Adam(1e-5), loss=weighted_bce_dice_loss, metrics=[dice_loss])
    best_thr = 0.73
    pipeline(model, model_name, img_h, img_w, batch_size=1, patience=3, low_memory=True)

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

def wmean_models(out_file, batch_size=1, tta=True):
    img_h = []
    img_w = []
    models = []
    model_names = []
    
    img_h.append(800)
    img_w.append(1280)
    model_name = 'unet_1280_800'
    model = get_unet(800, 1280)
    load_best_weights_max(model, model_name)
    model.compile(optimizer=Adam(1e-5), loss=bce_dice_loss, metrics=[dice_loss])
    model_names.append(model_name)
    models.append(model)

    img_h.append(800)
    img_w.append(1280)
    model_name = 'unet_1280_800_w'
    model = get_unet(800, 1280)
    load_best_weights_max(model, model_name)
    model.compile(optimizer=Adam(1e-5), loss=weighted_bce_dice_loss, metrics=[dice_loss])
    model_names.append(model_name)
    models.append(model)

    # alphas = search_alphas(models, model_names, img_h, img_w, batch_size=1)
    alphas = [0.49, 0.51]

    # print(alphas)

    # import pandas as pd
    # pd.DataFrame(alphas).to_csv('alphas.csv')



    batch_iterator = test_generator(batch_size=1, img_h=800, img_w=1280)
    rles = []
    names = []
    best_thr = 0.5

    for batch in tqdm(batch_iterator):
        i, X_batch, names_batch = batch
        models_preds = np.concatenate([model.predict_on_batch(X_batch) for model in models])
        if tta:
            X_batch_flip = np.array([cv2.flip(image, 1) for image in X_batch])
            # print (X_batch.shape, X_batch_flip.shape)
            models_preds_flip = np.concatenate([model.predict_on_batch(X_batch_flip) for model in models])
            # print (models_preds.shape, models_preds_flip.shape)
            models_preds_flip = np.array([cv2.flip(image, 1).reshape(image.shape) for image in models_preds_flip])
            # print (models_preds.shape, models_preds_flip.shape)
            models_preds = 0.5 * models_preds + 0.5 * models_preds_flip
        # print (models_preds.shape)
        cur_pred = np.average(models_preds, weights=alphas, axis=0)
        for mask in cur_pred:
            rles.append(rle(mask > best_thr))
        names.append(names_batch)
        if i % 1000 == 0:
            gc.collect()

    names = np.concatenate(names)

    df = pd.DataFrame({'img' : names, 'rle_mask' : rles})
    df.to_csv(out_file, index=False, compression='gzip')

if __name__ == '__main__':
    # model_1()
    # model_2()
    # model_3()
    # model_4()
    # model_5()
    model_6()
    # wmean_models('../output/wmean[356]_05.csv.gz')