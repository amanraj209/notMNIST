import os

import numpy as np
from PIL import Image
import imageio
import pickle
from tqdm import tqdm

import tensorflow as tf

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def read_data(directory):
    dataset = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            files = []
            for filez in os.walk(os.path.join(directory, dir)):
                for file in filez[2]:
                    img = Image.open(os.path.join(os.path.join(directory, dir),file))
                    files.append(np.array(img))
            for cur_imag in files:
                temp = []
                temp.append(cur_imag)
                temp.append(dir)
                dataset.append(tuple(temp))
    return dataset

def save_tfrecord_to_disk(path, arr_x, arr_y):
    with tf.python_io.TFRecordWriter(path) as writer:
        for i in tqdm(range(arr_x.shape[0])):
            image_raw = arr_x[i].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[arr_y[i]])),
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[arr_x[i].shape[0]])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[arr_x[i].shape[1]]))
            }))
            writer.write(example.SerializeToString())


def main():

    dataset = read_data('/Users/anandzutshi/Desktop/notMNIST/data_notMNIST/notMNIST_small')

    labels = {}
    X = []
    y = []
    ct = 0

    for x in dataset:
        if x[1] not in labels.keys():
            labels[x[1]] = ct
            ct = ct + 1

    dataset = np.array(dataset)

    for x in dataset:
        X.append(x[0])
        y.append(labels[x[1]])

    X = np.array(X)
    y = np.array(y)

    print('saving tfrecord..')

    save_tfrecord_to_disk('data.tfrecord', X, y)

    print('tfrecord saved successfully..')

if __name__ == '__main__':
    main()
