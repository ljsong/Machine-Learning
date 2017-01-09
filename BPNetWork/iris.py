#! /usr/bin/env python
# -*- coding:utf-8 -*-

import numpy
import os
import sys
import matplotlib.pyplot as plt
from NeuronNetwork import NeuronNetwork as NN
import Utils

CLASS_MAP = {'Iris-setosa' : 0,
             'Iris-versicolor' : 1,
             'Iris-virginica' : 2
            }

input_neurons = 4
hidden_neurons = 6
output_neurons = 3

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def load_data(data_path):
    if (not os.path.exists(data_path)):
        print 'Can not find data file: %s' % data_path
        sys.exit(-1)

    total_data = numpy.loadtxt(data_path, delimiter = ',', converters = {4:lambda s: CLASS_MAP[s]}) 
    numpy.random.shuffle(total_data)
    total_label = numpy.rint(total_data[:, 4]).astype(int)
    total_data = total_data[:, range(0, 4)]

    return total_data, one_hot_encoding(total_label, 2)

def one_hot_encoding(output, maxVal):
    out_trans = output.T
    length = out_trans.shape[0]

    one_hot = numpy.zeros((length, maxVal + 1))
    one_hot[numpy.arange(length), output] = 1

    return one_hot

def split_data(total_data, total_label):
    row, column = total_data.shape
    # the `test_cnt of data is used to train the network
    test_cnt = int(0.7 * row)

    # used to test
    xt = total_data[range(0, test_cnt), :]
    # used to validate
    xv = total_data[range(test_cnt, row), :]

    yt = total_label[range(0, test_cnt)]
    yv = total_label[range(test_cnt, row)]

    return xt, xv, yt, yv

def main():
    if len(sys.argv) <= 1:
        print 'Too few arguments!'
        sys.exit(-1)

    total_data, total_label = load_data(sys.argv[1])
    xt, xv, yt, yv = split_data(total_data, total_label)
    active_function = Utils.Sigmoid()
    #eval_function = Utils.CrossEntropyCost()
    eval_function = Utils.QuadraticCost()
    neural_network = NN([4, 6, 3], activator = active_function, evaluator = eval_function, learning_rate = 0.5)
    neural_network.train(xt, yt, batch_size = 1, epoch = int(sys.argv[2]))
    neural_network.validate(xv, yv)

if __name__ == '__main__':
    main()
