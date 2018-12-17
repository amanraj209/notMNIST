import pickle

from tqdm import tqdm

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from utils.reconstructImages import getReconstructedImages
from utils.trainTestVal import getTrainTestVal


class notMNISTDataLoaderNumpy:
    def __init__(self, config):
        self.config = config

        reconstructed_images = getReconstructedImages(self.config.data_numpy_pkl)

        temp_train , temp_test = train_test_split(reconstructed_images, test_size=0.2)
        temp_train , temp_val = train_test_split(temp_train, test_size=0.2)

        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.x_val = []
        self.y_val = []

        self.x_train, self.y_train, self.x_test, self.y_test, self.x_val, self.y_val = getTrainTestVal(temp_train, temp_test,temp_val)

        print('x_train: ', self.x_train.shape, self.x_train.dtype)
        print('y_train: ', self.y_train.shape, self.y_train.dtype)
        print('x_test: ', self.x_test.shape, self.x_test.dtype)
        print('y_test: ', self.y_test.shape, self.y_test.dtype)
        print('x_val: ', self.x_val.shape, self.x_val.dtype)
        print('y_val: ', self.y_val.shape, self.y_val.dtype)

        self.train_len = self.x_train.shape[0]
        self.test_len = self.x_test.shape[0]
        self.valid_len = self.x_val.shape[0]

        self.num_iterations_train = (self.train_len + self.config.batch_size - 1) // self.config.batch_size
        self.num_iterations_test = (self.test_len + self.config.batch_size - 1) // self.config.batch_size
        self.num_iterations_val = (self.valid_len + self.config.batch_size - 1) // self.config.batch_size

        print("Data loaded successfully..")

        self.features_placeholder = None
        self.labels_placeholder = None

        self.dataset = None
        self.iterator = None
        self.init_iterator_op = None
        self.next_batch = None

        self.build_dataset_api()

    def build_dataset_api(self):
        with tf.device('/cpu:0'):
            self.features_placeholder = tf.placeholder(tf.float32, [None] + list(self.x_train.shape[1:]))
            self.labels_placeholder = tf.placeholder(tf.int64, [None, ])

            self.dataset = tf.data.Dataset.from_tensor_slices((self.features_placeholder, self.labels_placeholder))
            self.dataset = self.dataset.batch(self.config.batch_size)

            self.iterator = tf.data.Iterator.from_structure(self.dataset.output_types,
                                                            self.dataset.output_shapes)

            self.init_iterator_op = self.iterator.make_initializer(self.dataset)

            self.next_batch = self.iterator.get_next()

            print("X_batch shape dtype: ", self.next_batch[0].shape)
            print("Y_batch shape dtype: ", self.next_batch[1].shape)

    def initialize(self, sess, is_train, is_val):
        if is_train:
            idx = np.random.choice(self.train_len, self.train_len, replace=False)
            self.x_train = self.x_train[idx]
            self.y_train = self.y_train[idx]
            sess.run(self.init_iterator_op, feed_dict={self.features_placeholder: self.x_train,
                                                       self.labels_placeholder: self.y_train})
        else:
            if is_val:
                sess.run(self.init_iterator_op, feed_dict={self.features_placeholder: self.x_val,
                                                           self.labels_placeholder: self.y_val})
            else:
                sess.run(self.init_iterator_op, feed_dict={self.features_placeholder: self.x_test,
                                                       self.labels_placeholder: self.y_test})

    def get_input(self):
        return self.next_batch


def main():
    class Config:
        data_numpy_pkl = '/Users/anandzutshi/Desktop/notMNIST/data/data.tfrecord'

        image_height = 28
        image_width = 28
        batch_size = 8

    tf.reset_default_graph()

    sess = tf.Session()

    data_loader = notMNISTDataLoaderNumpy(Config)

    x, y = data_loader.next_batch

    data_loader.initialize(sess, is_train=True)

    out_x, out_y = sess.run([x, y])

    print(out_x.shape, out_x.dtype)
    print(out_y.shape, out_y.dtype)

    data_loader.initialize(sess, is_train=False)

    out_x, out_y = sess.run([x, y])

    print(out_x.shape, out_x.dtype)
    print(out_y.shape, out_y.dtype)


if __name__ == '__main__':
    main()
