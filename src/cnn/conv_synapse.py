#! /usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import maximum
from numpy import minimum
from numpy import ones
from numpy import argmax
from numpy import sum
from numpy import zeros_like
from numpy.random import normal
from numpy import zeros
from im2col import im2col
from im2col import col2im
# import matplotlib.pyplot as plt
# import matplotlib
# import numpy


class Synapse(object):
    """
    Interface of kinds of synapses
    """

    def __init__(self):
        pass

    def active(self):
        pass

    def derivative(self):
        pass

    def feed_forward(self):
        pass

    def back_propagated(self, error):
        pass


class ConvSynapse(Synapse):
    def __init__(self, kernel_size=3, kernel_cnt=1, padding=1, stride=1,
                 learning_rate=0.1, momentum=0.7):
        """
        This class define a convolutional layer which has a filter matrix,
        input vector and output vector
        :param kernel_size: this should be a number which indicate the size of filter matrix
        :param kernel_cnt: the number of the filters
        :param padding: how many zeros should be padded around the input value
        :param stride: the step which kernel go through the input value
        :param learning_rate: the speed that network adjust itself from errors
        :param momentum:  
        """
        self.kernel_size = kernel_size
        self.kernel_cnt = kernel_cnt
        self.padding = padding
        self.stride = stride
        self.learning_rate = learning_rate
        self.momentum = momentum

        # input_layer after transforming by im2col
        self.input_cols = None
        self.input_layer = None
        self.output_layer = None
        self.kernel = None
        self.bias = None

        self.rf_height = 0
        self.rf_width = 0
        self.batch_size = 1

        self.bias_delta = None
        self.kernel_delta = None

    def set_input_layer(self, input_layer):
        self.input_layer = input_layer
        number, channel, height, width = self.input_layer.shape
        # each row is a filter
        self.kernel = normal(size=(
            self.kernel_cnt, channel, self.kernel_size, self.kernel_size))
        self.kernel_delta = zeros(self.kernel.shape).reshape(-1, self.kernel_cnt)

        self.bias = ones((self.kernel_cnt, 1))
        self.bias_delta = zeros((self.kernel_cnt, 1))

        # the counts of receptive filed
        self.rf_height = (height - self.kernel_size +
                          2 * self.padding) / self.stride + 1
        self.rf_width = (width - self.kernel_size +
                         2 * self.padding) / self.stride + 1
        self.batch_size = number

    def active(self):
        self.input_cols = im2col(self.input_layer, self.kernel_size,
                                 self.kernel_size, self.padding, self.stride)

        kernel_cols = self.kernel.reshape(self.kernel_cnt, -1)
        conv_sum = kernel_cols.dot(self.input_cols) + self.bias
        conv_sum = conv_sum.reshape(self.kernel_cnt, self.rf_height, self.rf_width, self.batch_size)

        conv_sum = conv_sum.transpose(3, 0, 1, 2)
        # fig = plt.figure()
        # for x in range(conv_sum.shape[1]):
        #     ax = fig.add_subplot(4, 4, x + 1)
        #     ax.matshow(conv_sum[0, x, :, :], cmap=matplotlib.cm.binary)
        #     plt.xticks(numpy.array([]))
        #     plt.yticks(numpy.array([]))
        # plt.show()
        return conv_sum

    def derivative(self):
        return 1

    def feed_forward(self):
        self.output_layer = self.active()

    def back_propagated(self, error):
        error_derv = error * self.derivative()

        error_sum = sum(error_derv, axis=(0, 2, 3))
        error_sum = error_sum.reshape(self.kernel_cnt, -1)
        self.bias_delta = \
            self.momentum * self.bias_delta + self.learning_rate * error_sum / self.batch_size
        self.bias -= self.bias_delta

        error_cols = error_derv.transpose(1, 2, 3, 0).reshape(self.kernel_cnt, -1)
        gradient = self.input_cols.dot(error_cols.T)

        kernel_cols = self.kernel.reshape(self.kernel_cnt, -1)
        prev_error_cols = kernel_cols.T.dot(error_cols)
        prev_error = col2im(prev_error_cols, self.input_layer.shape,
                            self.kernel_size, self.kernel_size,
                            padding=self.padding, stride=self.stride)

        self.kernel_delta = \
            self.momentum * self.kernel_delta + self.learning_rate * gradient / self.batch_size
        self.kernel -= self.kernel_delta.reshape(self.kernel.shape)

        return prev_error


class ReLUSynapse(Synapse):
    def __init__(self):
        self.output_layer = None
        self.input_layer = None
        self.input_reshaped = None

    def set_input_layer(self, input_layer):
        self.input_layer = input_layer

    def active(self):
        # should be W' @ X, but here weight is a matrix
        return maximum(self.input_layer, 0)

    def derivative(self):
        pass

    def feed_forward(self):
        self.output_layer = self.active()
        # fig = plt.figure()
        # for x in range(self.output_layer.shape[1]):
        #     ax = fig.add_subplot(4, 4, x + 1)
        #     ax.matshow(self.output_layer[0, x, :, :])
        #     plt.xticks(numpy.array([]))
        #     plt.yticks(numpy.array([]))
        # plt.show()

    def back_propagated(self, error):
        error_derv = error
        error_derv[self.input_layer <= 0] = 0
        # error_derv = error * self.derivative()

        return error_derv


class PoolingSynapse(Synapse):
    def __init__(self, pool_size=2, padding=0, stride=2):
        self.output_layer = None
        self.input_layer = None
        self.input_cols = None
        self.max_idx = None

        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding

        self.rf_height = 0
        self.rf_width = 0

    def set_input_layer(self, input_layer):
        self.input_layer = input_layer
        number, channel, height, width = self.input_layer.shape
        self.rf_width = (width - self.pool_size) / self.stride + 1
        self.rf_height = (height - self.pool_size) / self.stride + 1

    def pooling(self):
        pass

    def derivative(self):
        return 1

    def feed_forward(self):
        pass

    def back_propagated(self, error):
        pass


class MaxPoolingSynapse(PoolingSynapse):
    def __init__(self, pool_size, padding, stride):
        """
        MaxPooling select the maximum value of a matrix, we should remember the
        index of the maximum one and use it to back propagate the error to the
        correct neuron
        :param pool_size: the size of pooling layer
        """
        super(MaxPoolingSynapse, self).__init__(pool_size, padding, stride)

    def set_input_layer(self, input_layer):
        super(MaxPoolingSynapse, self).set_input_layer(input_layer)

    def pooling(self):
        number, channel, height, width = self.input_layer.shape
        input_reshaped = self.input_layer.reshape(number * channel, 1, height, width)

        self.input_cols = im2col(input_reshaped, self.pool_size,
                                 self.pool_size, self.padding, self.stride)
        self.max_idx = argmax(self.input_cols, axis=0)

        self.output_layer = self.input_cols[self.max_idx, range(self.max_idx.size)]
        self.output_layer = self.output_layer.reshape(self.rf_height, self.rf_width, number, channel)
        self.output_layer = self.output_layer.transpose(2, 3, 0, 1)
        #
        # fig = plt.figure()
        # for x in range(self.output_layer.shape[1]):
        #     ax = fig.add_subplot(4, 4, x + 1)
        #     ax.matshow(self.output_layer[0, x, :, :], cmap=matplotlib.cm.binary)
        #     plt.xticks(numpy.array([]))
        #     plt.yticks(numpy.array([]))
        # plt.show()

    def feed_forward(self):
        self.pooling()

    def back_propagated(self, error):
        number, channel, height, width = self.input_layer.shape
        input_delta = zeros_like(self.input_cols)
        error_flat = error.transpose(2, 3, 0, 1).ravel()

        input_delta[self.max_idx, range(error_flat.size)] = error_flat
        prev_error = col2im(input_delta, (number * channel, 1, height, width),
                            self.pool_size, self.pool_size, self.padding,
                            self.stride)
        prev_error = prev_error.reshape(self.input_layer.shape)

        return prev_error


class AvgPoolingSynapse(PoolingSynapse):
    def __init__(self, pool_size, stride):
        super(AvgPoolingSynapse, self).__init__(pool_size, stride)

    def set_input_layer(self, input_layer):
        super(AvgPoolingSynapse, self).set_input_layer(input_layer)

    def pooling(self):
        pass

    def feed_forward(self):
        pass

    def back_propagated(self, error):
        pass
