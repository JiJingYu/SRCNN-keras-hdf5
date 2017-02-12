from os import listdir
from os.path import isfile, join
import argparse
import h5py

import numpy as np
from scipy import misc
import network

input_size = 33
label_size = 21
pad = (33 - 21) // 2

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    return rgb.dot(xform.T)


def predict():
    model = network.srcnn((None, None, 1))
    f = h5py.File(option.model, mode='r')
    model.load_weights_from_hdf5_group(f['model_weights'])

    X = misc.imread(option.input, mode='YCbCr')

    w, h, c = X.shape
    w -= int(w % option.scale)
    h -= int(h % option.scale)
    X = X[0:w, 0:h, :]
    X[:,:,1] = X[:,:,0]
    X[:,:,2] = X[:,:,0]

    scaled = misc.imresize(X, 1.0/option.scale, 'bicubic')
    scaled = misc.imresize(scaled, option.scale/1.0, 'bicubic')
    newimg = np.zeros(scaled.shape)

    if option.baseline:
        misc.imsave(option.baseline, scaled[pad : w - w % input_size, pad: h - h % input_size, :])

    newimg[pad:-pad, pad:-pad, 0, None] = model.predict(scaled[None, :, :, 0, None] / 255)
    newimg[pad:-pad, pad:-pad, 1, None] = model.predict(scaled[None, :, :, 1, None] / 255)
    newimg[pad:-pad, pad:-pad, 2, None] = model.predict(scaled[None, :, :, 2, None] / 255)
    misc.imsave(option.output, newimg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-M', '--model',
                        default='./save/model_205.h5',
                        dest='model',
                        type=str,
                        nargs=1,
                        help="The model to be used for prediction")
    parser.add_argument('-I', '--input-file',
                        default='./dataset/Test/Set5/baby_GT.bmp',
                        dest='input',
                        type=str,
                        nargs=1,
                        help="Input image file path")
    parser.add_argument('-O', '--output-file',
                        default='./dataset/Test/Set5/baby_SRCNN.bmp',
                        dest='output',
                        type=str,
                        nargs=1,
                        help="Output image file path")
    parser.add_argument('-B', '--baseline',
                        default='./dataset/Test/Set5/baby_bicubic.bmp',
                        dest='baseline',
                        type=str,
                        nargs=1,
                        help="Baseline bicubic interpolated image file path")
    parser.add_argument('-S', '--scale-factor',
                        default=3.0,
                        dest='scale',
                        type=float,
                        nargs=1,
                        help="Scale factor")
    option = parser.parse_args()
    predict()
