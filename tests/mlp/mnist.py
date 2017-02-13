#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import struct
import numpy as np
import path_magic
from neural_network import NeuralNetwork as Network
from conv_net import ConvolutionalNet as ConvNet
from test_utils import train, validate


"""
This file is forked from https://gist.github.com/akesling/5358964
which is GPL licensed.
"""


def read(dataset="training", path="."):
    """
    Python function for importing the MNIST data set. It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.unit8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'tesing' or 'training'")

    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    return lbl, img


def one_hot_encoding(target):
    # generate a diagonal matrix generate one-hot encoidng
    expansion_matrix = np.eye(10)

    return expansion_matrix[target]


def main():
    if len(sys.argv) < 1:
        print 'Too few arguments!'
        sys.exit(-1)

    labels, images = read(path=sys.argv[1])
    length, x, y = images.shape
    target = one_hot_encoding(labels)
    inputs = images.reshape(1, 1, length, x * y)
    #network = Network(
    #    [784, 400, 10],
    #    'SM', 'C',
    #    learning_rate=0.0001,
    #    momentum=0.7)
    network = ConvNet(3, 32, None, padding=2, stride=2)

    train(network, inputs, target, batch_size=1, epoch=int(sys.argv[2]))

    labels, images = read(dataset='testing', path=sys.argv[1])
    length, x, y = images.shape
    target = one_hot_encoding(labels)
    inputs = images.reshape(1, 1, length, x * y)

    validate(network, inputs, target)

if __name__ == '__main__':
    main()
