import numpy as np

from utilities import *
from io_utils import *
from metric import *
from models import *

from keras.optimizers import *
from keras.callbacks import *
import argparse
import gc

RS = 17

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

def train_model_1():
    img_h, img_w = 128, 128
    model = get_unet(img_h, img_w)
    model_name = 'unet_128'
    model.compile(optimizer=Adam(1e-4), loss=bce_dice_loss, metrics=[dice_loss])
    train(model, model_name, img_h, img_w, load_best_weights=load_weights_func_max, n_epochs=100, batch_size=32, patience=10, reduce_rate=0.1)

def train_model_2():
    img_h, img_w = 480, 720
    model = get_unet(img_h, img_w)
    model_name = 'unet_480_720'
    model.compile(optimizer=Adam(1e-4), loss=bce_dice_loss, metrics=[dice_loss])
    pipeline(model, model_name, img_h, img_w, batch_size=2, patience=3, low_memory=True)
    train(model, model_name, img_h, img_w, load_best_weights=load_weights_func_max, n_epochs=100, batch_size=2, patience=3, reduce_rate=0.1)

def train_model_3():
    img_h, img_w = 800, 1280
    model = get_unet(img_h, img_w)
    model_name = 'unet_1280_800'
    model.compile(optimizer=Adam(1e-4), loss=bce_dice_loss, metrics=[dice_loss])
    train(model, model_name, img_h, img_w, load_best_weights=load_weights_func_max, n_epochs=100, batch_size=1, patience=3, reduce_rate=0.1)

def train_model_4():
    img_h, img_w = 1024, 1024
    model = get_unet(img_h, img_w)
    model_name = 'unet_1024_1024'
    model.compile(optimizer=Adam(1e-4), loss=bce_dice_loss, metrics=[dice_loss])
    train(model, model_name, img_h, img_w, load_best_weights=load_weights_func_max, n_epochs=100, batch_size=1, patience=3, reduce_rate=0.1)

def train_model_5():
    img_h, img_w = 832, 1280
    model = get_unet(img_h, img_w)
    model_name = 'unet_1280_832'
    model.compile(optimizer=Adam(1e-5), loss=bce_dice_loss, metrics=[dice_loss])
    train(model, model_name, img_h, img_w, load_best_weights=load_weights_func_max, n_epochs=100, batch_size=1, patience=3, reduce_rate=0.1)

def train_model_6():
    img_h, img_w = 800, 1280
    model = get_unet(img_h, img_w)
    model_name = 'unet_1280_800_w'
    model.compile(optimizer=Adam(1e-5), loss=weighted_bce_dice_loss, metrics=[dice_loss])
    train(model, model_name, img_h, img_w, load_best_weights=load_weights_func_max, n_epochs=100, batch_size=1, patience=3, reduce_rate=0.1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument("--model", type=int, default=6, help="model to train")
    args = parser.parse_args()

    train_functions = [None, train_model_1, train_model_2, train_model_3, train_model_4, train_model_5, train_model_6]
    model_f = train_functions[args.model]
    model_f()
    gc.collect()
