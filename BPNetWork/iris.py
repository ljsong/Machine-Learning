#! /usr/bin/env python
# -*- coding:utf-8 -*-

import numpy
import os
import sys
import matplotlib.pyplot as plt
from NeuronNetwork import NeuronNetwork as NN

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

    total_data = numpy.loadtxt(data_path, delimiter = ',', usecols = (0, 1, 2, 3))
    total_label = numpy.genfromtxt(data_path, delimiter = ',', usecols = 4, dtype = int, converters = {4:lambda s:CLASS_MAP[s]})

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

def predict(features, target, weight_h, weight_o, input_bias, hid_bias):
    nrow, ncol = features.shape
    
    ix = 0
    correct = 0
    while ix < nrow:
        each_input = features[ix, :].reshape(ncol, 1)
        each_target = target[ix, :].reshape(output_neurons, 1)

        hidden_input = weight_h.T.dot(each_input) + input_bias
        hidden_output = sigmoid(hidden_input)

        each_output = sigmoid(weight_o.T.dot(hidden_output) + hid_bias)
        each_output = numpy.rint(each_output)
        

        diff = each_output - each_target
        error = 1.0 / 2 * diff.T.dot(diff)

        if not error:
            correct += 1

        ix += 1

    print "Correct percentage: %3.3f%%" % (correct * 1.0 / nrow * 100)

def learn(x, y, epoch):
    nrow, ncol = x.shape
    input_bias = numpy.ones((hidden_neurons, 1))
    hid_bias = numpy.ones((output_neurons, 1))

    weight_h = numpy.random.random((ncol, hidden_neurons))   # 4 hidden neurons
    weight_o = numpy.random.random((hidden_neurons, output_neurons))   # 3 output neurons

    total_error = 0
    learning_rate = 0.2
    times = 1

    each_input = numpy.ones((ncol, 1))
    hid_to_output = numpy.ones((hidden_neurons, 1))
    while times <= epoch:
        output_diff = 0
        total_error = 0
        ix = 0

        while ix < nrow:
            each_input = x[ix, :].reshape(ncol, 1)
            target = y[ix].reshape(output_neurons, 1)
            
            hidden_input = weight_h.T.dot(each_input) + input_bias
            hidden_output = sigmoid(hidden_input)

            each_output = weight_o.T.dot(hidden_output) + hid_bias
            each_output = sigmoid(each_output)

            error = target - each_output
            error_deriv = error * each_output * (1 - each_output)

            hid_to_output_weight_gradient = hidden_output.dot(error_deriv.T)
            input_hid_error_deriv = weight_o.dot(error_deriv) * hidden_output * (1 - hidden_output)
            input_to_hid_weight_gradient = each_input.dot(input_hid_error_deriv.T)

            weight_o += learning_rate * hid_to_output_weight_gradient
            weight_h += learning_rate * input_to_hid_weight_gradient

            hid_bias = learning_rate * error
            input_bias = learning_rate * weight_o.dot(error_deriv)

            total_error += 1.0 / 2 * error.T.dot(error)
            ix += 1
        
        if times % 100 == 0:
            print 'Total error: %3.3f' % total_error

        if numpy.fabs(total_error) < 1e-5:
            break
        times += 1

    return weight_h, weight_o, input_bias, hid_bias

def main():
    if len(sys.argv) <= 1:
        print 'Too few arguments!'
        sys.exit(-1)

    total_data, total_label = load_data(sys.argv[1])
    xt, xv, yt, yv = split_data(total_data, total_label)
    neural_network = NN([4, 6, 3], learning_rate = 0.05)
    neural_network.train(xt, yt)
    neural_network.validate(xv, yv)
    #weight_h, weight_o, input_bias, hid_bias = learn(xt, yt, int(sys.argv[2]))
    #predict(xv, yv, weight_h, weight_o, input_bias, hid_bias)

if __name__ == '__main__':
    main()
