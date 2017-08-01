from utilities import *
from io_utils import *
from metric import *

from keras.optimizers import *
from keras.layers import *
from keras.models import Model
from keras import backend as K
import tensorflow as tf
import h5py
import gc

def get_unet_zp(img_h, img_w):
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

    up6 = Concatenate(axis=3)([ZeroPadding2D(padding=((0,0),(0,1)))(UpSampling2D(size=(2, 2))(conv5)), conv4])
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = Concatenate(axis=3)([ZeroPadding2D(padding=((0,0),(0,1)))(UpSampling2D(size=(2, 2))(conv6)), conv3])
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = Concatenate(axis=3)([ZeroPadding2D(padding=((0,0),(0,1)))(UpSampling2D(size=(2, 2))(conv7)), conv2])
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = Concatenate(axis=3)([ZeroPadding2D(padding=((0,0),(0,1)))(UpSampling2D(size=(2, 2))(conv8)), conv1])
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    return model

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

def load_best_weights(model, model_name):
    wdir = 'weights_' + str(model_name) + '/'
    if not os.path.exists(wdir):
        os.mkdir(wdir)
    elif len(os.listdir(wdir)) > 0:
        wf = sorted(glob.glob(wdir + '*.h5'))[-1]
        model.load_weights(wf)
        print('loaded weights file: ', wf)

def train(model, model_name, img_h, img_w, h5_file, n_epochs=100, n_val=1024, batch_size=32,
                patience=5, reduce_rate=0.5, reduce_rounds=6):
    data = Dataset(h5_file, MAIN_DIR, img_h, img_w)
    best_loss = 10
    current_reduce_round = 0
    load_best_weights(model, model_name)
    
    for e in range(n_epochs):
        print('Epoch: ', e)
        batch_count = 0
        for batch in data.batch_iterator_train(n_val, batch_size, shuffle=True):
            i, X_batch, y_batch, _ = batch
            X_batch, y_batch = preprocess_batch(X_batch, y_batch)
            model.train_on_batch(X_batch, y_batch)

            batch_count = i

        # Validation
        y_pred = []
        y_val = []
        val_loss = []
        for val_batch in data.batch_iterator_val(n_val, batch_size):
            j, X_val_batch, y_val_batch, _ = val_batch
            X_val_batch, y_val_batch = preprocess_batch(X_val_batch, y_val_batch)
            y_val.append(y_val_batch)
            y_pred.append(model.predict_on_batch(X_val_batch))
            val_loss.append(model.test_on_batch(X_val_batch, y_val_batch))

        y_pred = np.concatenate(y_pred)
        y_val = np.concatenate(y_val)
        # print (y_pred[y_pred > 0.5].size, y_val[y_val > 0.5].size)
        cur_dice = get_score(y_val, y_pred)
        cur_loss = np.mean(val_loss)
        print('val loss {:.6}; val dice: {:.5}; epoch: {}; batch: {}'.format(cur_loss, cur_dice, e, batch_count))
        
        if cur_loss < best_loss:
            best_loss = cur_loss
            wait = 0
            model.save_weights('weights_' + str(model_name) + '/{:.6}_{}_{}.h5'.format(best_loss, e, batch_count))
        else:
            if wait >= patience:
                current_reduce_round += 1
                if current_reduce_round <= reduce_rounds:
                    lr = K.get_value(model.optimizer.lr)
                    K.set_value(model.optimizer.lr, lr * reduce_rate)
                    wait = 0
                    print('New lr: {:.5}'.format(lr * reduce_rate))
                    load_best_weights(model, model_name)
                else:
                    print('Epoch {}: early stopping'.format(e))
                    model.stop_training = True
                    return
            wait += 1

def search_best_threshold(model, model_name, img_h, img_w, h5_file, n_val=1024, batch_size=32,
        start_thr=0.1, end_thr=0.51, delta=0.01):
    wdir = 'weights_' + str(model_name) + '/'
    if not os.path.exists(wdir):
        os.mkdir(wdir)
    elif len(os.listdir(wdir)) > 0:
        wf = sorted(glob.glob(wdir + '*.h5'))[0]
        model.load_weights(wf)
        print('loaded weights file: ', wf)

    data = Dataset(h5_file, MAIN_DIR, img_h, img_w)

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
            
def predict(model, model_name, img_h, img_w, h5_file=None, batch_size=32):
    wdir = 'weights_' + str(model_name) + '/'
    if not os.path.exists(wdir):
        os.mkdir(wdir)
    elif len(os.listdir(wdir)) > 0:
        wf = sorted(glob.glob(wdir + '*.h5'))[0]
        model.load_weights(wf)
        print('loaded weights file: ', wf)

    data = DatasetTest(h5_file, MAIN_DIR, img_h, img_w)

    if h5_file is None:
        batch_iterator = data.batch_iterator_dir(batch_size)
    else:
        batch_iterator = data.batch_iterator(batch_size)

    preds = []
    names = []
    for batch in tqdm(batch_iterator):
        i, X_batch, names_batch = batch
        X_batch = preprocess_batch(X_batch)

        cur_pred = model.predict_on_batch(X_batch)
        preds.append(cur_pred)
        names.append(names_batch)

    preds = np.concatenate(preds)
    names = np.concatenate(names)

    return preds, names

def make_submission(preds, names, out_file, threshold=0.5):
    rles = []
    for mask in tqdm(preds):
        rles.append(rle(mask > threshold))

    df = pd.DataFrame({'img' : names, 'rle_mask' : rles})
    df.to_csv(out_file, index=False, compression='gzip')

def predict_and_make_submission(model, model_name, img_h, img_w, out_file, h5_file=None, batch_size=32, threshold=0.5):
    wdir = 'weights_' + str(model_name) + '/'
    if not os.path.exists(wdir):
        os.mkdir(wdir)
    elif len(os.listdir(wdir)) > 0:
        wf = sorted(glob.glob(wdir + '*.h5'))[0]
        model.load_weights(wf)
        print('loaded weights file: ', wf)

    data = DatasetTest(h5_file, MAIN_DIR, img_h, img_w)

    if h5_file is None:
        batch_iterator = data.batch_iterator_dir(batch_size)
    else:
        batch_iterator = data.batch_iterator(batch_size)

    rles = []
    names = []
    for batch in tqdm(batch_iterator):
        i, X_batch, names_batch = batch
        X_batch = preprocess_batch(X_batch)

        cur_pred = model.predict_on_batch(X_batch)
        for mask in cur_pred:
            rles.append(rle(mask > threshold))
        names.append(names_batch)
        gc.collect()

    names = np.concatenate(names)

    df = pd.DataFrame({'img' : names, 'rle_mask' : rles})
    df.to_csv(out_file, index=False, compression='gzip')

def pipeline(model, model_name, img_h, img_w, train_h5, test_h5, batch_size=32, patience=10, low_memory=False):
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy')
    # model.summary()
    train(model, model_name, img_h, img_w, h5_file=train_h5,
            n_epochs=2000, n_val=768, batch_size=batch_size,
            patience=patience, reduce_rate=0.1, reduce_rounds=2)
    
    if low_memory:
        predict_and_make_submission(model, model_name, img_h, img_w, out_file='../output/' + model_name + '_05.csv.gz', h5_file=test_h5, batch_size=batch_size, threshold=0.5)
    else:
        best_thr = search_best_threshold(model, model_name, img_h, img_w, h5_file=train_h5,
            n_val=1024, batch_size=batch_size, start_thr=0.1, end_thr=0.5, delta=0.01)
        preds, names = predict(model, model_name, img_h, img_w, h5_file=test_h5, batch_size=batch_size)
        best_thr = 0.1
        make_submission(preds, names, out_file='../output/' + model_name + '_002.csv.gz', threshold=best_thr)
        best_thr = 0.25
        make_submission(preds, names, out_file='../output/' + model_name + '_025.csv.gz', threshold=best_thr)
        best_thr = 0.5
        make_submission(preds, names, out_file='../output/' + model_name + '_05.csv.gz', threshold=best_thr)
    gc.collect()

def model_1():
    img_h, img_w = 128, 128
    model = get_unet(img_h, img_w)
    model_name = 'unet_128'
    pipeline(model, model_name, img_h, img_w, TRAIN_H5_FILE_128, TEST_H5_FILE_128, batch_size=32, patience=10)

def model_2():
    img_h, img_w = 480, 720
    model = get_unet(img_h, img_w)
    model_name = 'unet_480_720'
    pipeline(model, model_name, img_h, img_w, TRAIN_H5_FILE_480_720, None, batch_size=2, patience=5, low_memory=True)

if __name__ == '__main__':
    # model_1()
    model_2()