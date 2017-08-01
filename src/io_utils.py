import numpy as np
import glob
import os
import h5py
import logging
from scipy.misc import imread, imresize
from dask import delayed, threaded, compute

logging.getLogger('dataset loader').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S', )
logger = logging.getLogger(__name__)

MAIN_DIR = '../input/'
TRAIN_DIR = MAIN_DIR + 'train/'
TEST_DIR = MAIN_DIR + 'test/'
MASK_DIR = MAIN_DIR + 'train_masks/'

TRAIN_H5_FILE = 'C://data//carvana_train.h5'
TRAIN_H5_FILE_128 = 'C://data//carvana_train_128.h5'
TEST_H5_FILE_128 = 'C://data//carvana_test_128.h5'
TRAIN_H5_FILE_256 = 'C://data//carvana_train_256.h5'
TRAIN_H5_FILE_480_720 = 'C://data//carvana_train_480_720.h5'
TEST_H5_FILE_480_720 = '../features/carvana_test_480_720.h5'

CARVANA_H = 1280
CARVANA_W = 1918

def rle (img):
    img = imresize(img.reshape(img.shape[0], img.shape[1]), (CARVANA_H, CARVANA_W))
    flat_img = img.flatten()
    flat_img = np.where(flat_img > 0.5, 1, 0).astype(np.uint8)

    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix


    return ' '.join([str(starts_ix[i]) + ' ' + str(lengths[i]) for i in range(len(lengths))]) #starts_ix, lengths

class Dataset:
    def __init__(self, filename: str=None, dir_to_cache: str=None, img_h: int=1280, img_w: int=1918):
        self.file = filename
        self.img_h = img_h
        self.img_w = img_w
        logger.info('Cache file: {}'.format(self.file))
        if not os.path.exists(self.file):
            self.h5_file = h5py.File(self.file, 'w')
            self.cache(dir_to_cache)
        else:
            self.h5_file = h5py.File(self.file, 'r')
            logger.info("Loaded h5py dataset with {} examples.".format(len(self.h5_file['names'])))

    @staticmethod
    def read_img(filename):
        return np.clip(imread(filename), 0, 255).astype(np.uint8)

    def cache(self, dir_to_cache):
        logger.info('Caching files from {}'.format(dir_to_cache))
        train_files = os.listdir(os.path.join(dir_to_cache, "train"))
        x_data = self.h5_file.create_dataset('x_data', shape=(len(train_files), self.img_h, self.img_w, 3), dtype=np.uint8)
        y_data = self.h5_file.create_dataset('y_data', shape=(len(train_files), self.img_h, self.img_w, 1), dtype=np.uint8)
        names = self.h5_file.create_dataset('names', shape=(len(train_files),), dtype=h5py.special_dtype(vlen=str))

        logger.info('There are {} files in train'.format(len(train_files)))
        for i, fn in enumerate(train_files):
            img = self.read_img(os.path.join(dir_to_cache, "train", fn))
            names[i] = fn
            x_data[i, :, :, :] = imresize(img, (self.img_h, self.img_w, 3))
            mask = imread(os.path.join(os.path.join(dir_to_cache, "train_masks"),
                                                     fn.replace('.jpg', '_mask.gif')))[:,:,0]
            y_data[i, :, :, :] = imresize(mask, (self.img_h, self.img_w)).reshape((self.img_h, self.img_w, 1))
            
            if i % 100 == 0:
                logger.info("Processed {} files.".format(i))

    def batch_iterator_train(self, n_val: int=1000, batch_size: int=32, shuffle=False):
        """Generates a batch iterator for a dataset."""
        names = self.h5_file['names']
        data_size = len(names[:-n_val])
        x_dat = self.h5_file['x_data']
        y_dat = self.h5_file['y_data']
        num_batches_per_epoch = int((data_size-1) / batch_size) + 1

        # Shuffle the data at each epoch
        shuffle_indices = np.arange(data_size)
        if shuffle:
            shuffle_indices = np.random.permutation(shuffle_indices)
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            batch_indices = sorted(list(shuffle_indices[start_index:end_index]))

            yield batch_num, \
                  np.concatenate(np.array(compute([delayed(x_dat.__getitem__)(i) for i in batch_indices], get=threaded.get))), \
                  np.concatenate(np.array(compute([delayed(y_dat.__getitem__)(i) for i in batch_indices], get=threaded.get))), \
                  np.concatenate(np.array(compute([delayed(names.__getitem__)(i) for i in batch_indices], get=threaded.get)))

    def batch_iterator_val(self, n_val: int=None, batch_size: int=32):
        """Generates a batch iterator for a dataset."""
        names = self.h5_file['names']

        x_dat = self.h5_file['x_data']
        y_dat = self.h5_file['y_data']
        num_batches_per_epoch = int((n_val - 1) / batch_size) + 1

        start_idx = len(names) - n_val
        end_idx = start_idx + n_val
        indices = np.arange(start_idx, end_idx)

        for batch_num in range(num_batches_per_epoch):
            start_batch_index = batch_num * batch_size
            end_batch_index = min((batch_num + 1) * batch_size, end_idx)
            batch_indices = sorted(list(indices[start_batch_index : end_batch_index]))
            yield batch_num, \
                  np.concatenate(np.array(compute([delayed(x_dat.__getitem__)(i) for i in batch_indices], get=threaded.get))), \
                  np.concatenate(np.array(compute([delayed(y_dat.__getitem__)(i) for i in batch_indices], get=threaded.get))), \
                  np.concatenate(np.array(compute([delayed(names.__getitem__)(i) for i in batch_indices], get=threaded.get)))

class DatasetTest:
    def __init__(self, filename: str=None, dir_to_cache: str=None, img_h: int=1280, img_w: int=1918):
        self.file = filename
        self.img_h = img_h
        self.img_w = img_w
        logger.info('Cache file: {}'.format(self.file))
        if self.file is not None:
            if not os.path.exists(self.file):
                self.h5_file = h5py.File(self.file, 'w')
                self.cache(dir_to_cache)
            else :
                self.h5_file = h5py.File(self.file, 'r')
                logger.info("Loaded h5py dataset with {} examples.".format(len(self.h5_file['names'])))

    @staticmethod
    def read_img(filename):
        return np.clip(imread(filename), 0, 255).astype(np.uint8)

    def cache(self, dir_to_cache):
        logger.info('Caching files from {}'.format(dir_to_cache))
        test_files = os.listdir(os.path.join(dir_to_cache, "test"))
        x_data = self.h5_file.create_dataset('x_data', shape=(len(test_files), self.img_h, self.img_w, 3), dtype=np.uint8)
        names = self.h5_file.create_dataset('names', shape=(len(test_files),), dtype=h5py.special_dtype(vlen=str))

        logger.info('There are {} files in test'.format(len(test_files)))
        for i, fn in enumerate(test_files):
            img = self.read_img(os.path.join(dir_to_cache, "test", fn))
            x_data[i, :, :, :] = imresize(img, (self.img_h, self.img_w, 3))
            names[i] = fn
            if i % 100 == 0:
                logger.info("Processed {} files.".format(i))

    def batch_iterator(self, batch_size: int=32):
        """Generates a batch iterator for a dataset."""
        names = self.h5_file['names']
        data_size = len(names)

        x_dat = self.h5_file['x_data']
        num_batches_per_epoch = int((data_size-1) / batch_size) + 1

        indices = np.arange(data_size)
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            batch_indices = sorted(list(indices[start_index:end_index]))
            yield batch_num, \
                  np.concatenate(np.array(compute([delayed(x_dat.__getitem__)(i) for i in batch_indices], get=threaded.get))), \
                  np.concatenate(np.array(compute([delayed(names.__getitem__)(i) for i in batch_indices], get=threaded.get)))

    def batch_iterator_dir(self, batch_size: int=32):
        test_files = os.listdir(TEST_DIR)
        batch_x = np.zeros((batch_size, self.img_h, self.img_w, 3))
        batch_names = []
        batch_n = 0

        for i, fn in enumerate(test_files):
            img = self.read_img(os.path.join(TEST_DIR, fn))
            batch_x[i % batch_size] = imresize(img, (self.img_h, self.img_w, 3))
            batch_names.append(fn)
            if i != 0 and (i - 1) % batch_size == 0:
                yield batch_n, batch_x, batch_names
                batch_names = []
                batch_n += 1
