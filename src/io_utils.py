import numpy as np
import glob
import os
import cv2
import scipy.misc as scm
from dask import delayed, threaded, compute

MAIN_DIR = 'C://data//carvana//'
TRAIN_DIR = MAIN_DIR + 'train'
TEST_DIR = MAIN_DIR + 'test'
MASK_DIR = MAIN_DIR + 'train_masks'

# TRAIN_H5_FILE = 'C://data//carvana_train.h5'
# TRAIN_H5_FILE_128 = 'C://data//carvana_train_128.h5'
# TEST_H5_FILE_128 = 'C://data//carvana_test_128.h5'
# TRAIN_H5_FILE_256 = 'C://data//carvana_train_256.h5'
# TRAIN_H5_FILE_480_720 = 'C://data//carvana_train_480_720.h5'
# TEST_H5_FILE_480_720 = '../features/carvana_test_480_720.h5'

CARVANA_H = 1280
CARVANA_W = 1918

def rle(img):
    img = cv2.resize(img.astype(np.uint8).reshape(img.shape[0], img.shape[1]), (CARVANA_W, CARVANA_H))
    flat_img = img.flatten()
    flat_img = np.where(flat_img > 0.5, 1, 0).astype(np.uint8)

    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix

    return ' '.join([str(starts_ix[i]) + ' ' + str(lengths[i]) for i in range(len(lengths))]) #starts_ix, lengths


def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def train_generator(ids_train_split, batch_size, img_h, img_w):
    while True:
        for start in range(0, len(ids_train_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]
            for id in ids_train_batch.values:
                img = cv2.imread(os.path.join(TRAIN_DIR, '{}.jpg'.format(id)))
                img = cv2.resize(img, (img_w, img_h))
                mask = scm.imread(os.path.join(MASK_DIR, '{}_mask.gif'.format(id)), mode='L')
                mask = cv2.resize(mask, (img_w, img_h))
                img, mask = randomShiftScaleRotate(img, mask,
                                                   shift_limit=(-0.0625, 0.0625),
                                                   scale_limit=(-0.1, 0.1),
                                                   rotate_limit=(-0, 0))
                img, mask = randomHorizontalFlip(img, mask)
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            
            yield x_batch, y_batch

def val_generator(ids_val_split, batch_size, img_h, img_w):
    while True:
        for start in range(0, len(ids_val_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_val_split))
            ids_valid_batch = ids_val_split[start:end]
            for id in ids_valid_batch.values:
                img = cv2.imread(os.path.join(TRAIN_DIR, '{}.jpg'.format(id)))
                img = cv2.resize(img, (img_w, img_h))
                mask = scm.imread(os.path.join(MASK_DIR, '{}_mask.gif'.format(id)), mode='L')
                mask = cv2.resize(mask, (img_w, img_h))
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch

def test_generator(batch_size, img_h, img_w):
    test_files = os.listdir(TEST_DIR)
    x_batch = np.zeros((batch_size, img_h, img_w, 3))
    batch_names = []
    batch_n = 0

    for i, fn in enumerate(test_files):
        img = cv2.imread(os.path.join(TEST_DIR, fn))
        x_batch[i % batch_size] = cv2.resize(img, (img_w, img_h))
        batch_names.append(fn)
        if i != 0 and (i - 1) % batch_size == 0:
            yield batch_n, np.array(x_batch, np.float32) / 255, batch_names
            batch_names = []
            batch_n += 1
