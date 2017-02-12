from os import listdir, makedirs
from os.path import isfile, join, exists
import os
import argparse
import numpy as np
import h5py
from scipy import misc


def remove_if_exist(file_name):
    if exists(file_name):
        os.remove(file_name)


def preprocess_dataset(option, **kwargs):

    input_dir = option.input_dir
    output_file = option.output_file

    scale = kwargs.pop('scale', 3)
    input_size = kwargs.pop('input_size', 33)
    label_size = kwargs.pop('label_size', 21)
    channels = kwargs.pop('channels', 1)
    stride = kwargs.pop('stride', 14)
    chunks = kwargs.pop('chunks', 1024)

    pad = (input_size - label_size) // 2

    input_nums = 1024
    remove_if_exist(output_file)
    with h5py.File(output_file, 'w') as f:
        f.create_dataset("input", (input_nums, input_size, input_size, channels),
                           maxshape=(None, input_size, input_size, channels),
                           chunks=(128, input_size, input_size, channels),
                           dtype='float32')
        f.create_dataset("label", (input_nums, label_size, label_size, channels),
                           maxshape=(None, label_size, label_size, channels),
                           chunks=(128, label_size, label_size, channels),
                           dtype='float32')
        f.create_dataset("count", data=(0,))

    count = 0
    for f in listdir(input_dir):
        f = join(input_dir, f)
        if not isfile(f):
            continue
        print(f)

        image = misc.imread(f, flatten=False, mode='YCbCr')

        w, h, c = image.shape
        w -= int(w % scale)
        h -= int(h % scale)
        image = image[0:w, 0:h, 0]

        scaled = misc.imresize(image, 1.0 / scale, 'bicubic')
        scaled = misc.imresize(scaled, scale / 1.0, 'bicubic')

        h5f = h5py.File(output_file, 'a')
        if count + chunks > h5f['input'].shape[0]:
            input_nums = count + chunks
            h5f['input'].resize((input_nums, input_size, input_size, channels))
            h5f['label'].resize((input_nums, label_size, label_size, channels))

        for i in range(0, h - input_size + 1, stride):
            for j in range(0, w - input_size + 1, stride):

                sub_img = scaled[j: j + input_size, i: i + input_size]
                sub_img = sub_img.reshape([1, input_size, input_size, 1])
                sub_img = sub_img / 255

                sub_img_label = image[j + pad: j + pad + label_size, i + pad: i + pad + label_size]
                sub_img_label = sub_img_label.reshape([1, label_size, label_size, 1])
                sub_img_label = sub_img_label / 255

                h5f['input'][count] = sub_img
                h5f['label'][count] = sub_img_label
                count += 1

    h5f = h5py.File(output_file, 'a')
    h5f['input'].resize((count, input_size, input_size, channels))
    h5f['label'].resize((count, label_size, label_size, channels))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-I', '--input-dir',
                        default='./dataset/Train',
                        dest='input_dir',
                        type=str,
                        nargs=1,
                        help="Data input directory")
    parser.add_argument('-O', '--output-file',
                        default='./data.h5',
                        dest='output_file',
                        type=str,
                        nargs=1,
                        help="Data output file with hdf5 format")
    option = parser.parse_args()

    preprocess_dataset(option=option,
                       scale=3,
                       input_size=33,
                       label_size=21,
                       stride=14,
                       channels=1,
                       chunks=1024)
