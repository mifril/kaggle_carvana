import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

from utilities import *
from io_utils import *
from metric import *
from models import *

from keras.optimizers import *
from keras.layers import *
import gc

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

def predict_model(model, model_name, img_h, img_w, load_weights_func, batch_size=32):
    best_thr = search_best_threshold(model, model_name, img_h, img_w, load_best_weights=load_weights_func,
        batch_size=batch_size, start_thr=0.3, end_thr=0.81, delta=0.01)
    print (best_thr)

    preds, names = predict(model, model_name, img_h, img_w, load_best_weights=load_weights_func, batch_size=batch_size)
    best_thr = 0.5
    make_submission(preds, names, out_file='../output/' + model_name + '.csv.gz', threshold=best_thr)

    predict_and_make_submission(model, model_name, img_h, img_w, load_best_weights=load_weights_func,
        out_file='../output/' + model_name + '_{}.csv.gz'.format(best_thr), batch_size=batch_size, threshold=best_thr)

def predict_wmean_models(out_file, batch_size=1, tta=True):
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

    alphas = [0.49, 0.51]

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
    predict_wmean_models('../output/wmean[356]_05.csv.gz')
