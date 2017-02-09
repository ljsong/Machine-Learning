#! /usr/bin/env python
# -*- coding: utf-8 -*-

from numpy.random import normal
from numpy import maximum
from numpy import concatenate
from numpy import ones
from numpy import sum
from im2col import im2col
from im2col import col2im


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

    def forward(self):
        pass

    def backward(self):
        pass


class ConvSyanpse(Synapse):

    def __init__(self, input_layer, kernel_size, kernel_cnt, padding=1, stride=1):
        """
        This class define a convolutional layer which has a filter matrix,
        input vector and output vector
        :param kernel_size: this should be a number which indicate the size of filter matrix
        :param kernel_cnt: the number of the filters
        :param input_layer: this vector store the value of input which should be 3-D image
        :param padding: how many zeros should be padded around the input value
        :param stride: the step which kernel go through the input value
        """
        self.input_layer = input_layer
        self.kernel_size = kernel_size
        self.kernel_cnt = kernel_cnt
        self.padding = padding
        self.stride = stride

        # input_layer after transforming by im2col
        self.input_cols = None
        self.output_layer = None

        depth, height, width = input_layer.shape
        # each row is a filter
        self.kernel = normal(size=(self.kernel_cnt, depth, self.kernel_size, self.kernel_size))
        self.bias = ones((self.kernel_cnt, 1))

        self.bias_delta = None
        self.kernel_delta = None

        # the counts of receptive filed
        self.rf_height = (height - self.kernel_size + 2 * self.padding) / self.stride + 1
        self.rf_width = (width - self.kernel_size + 2 * self.padding) / self.stride + 1

    def active(self):
        inputs = self.input_layer.reshape(1, self.input_layer.shape)
        self.input_cols = im2col(inputs, self.kernel_size,
                                 self.kernel_size, self.padding, self.stride)

        kernel_cols = self.kernel.reshape(self.kernel_cnt, -1)
        conv_sum = kernel_cols.dot(self.input_cols) + self.bias
        conv_sum = conv_sum.reshape(self.kernel_cnt, self.rf_height, self.rf_width, 1)

        conv_sum = conv_sum.transpose(3, 0, 1, 2)
        return conv_sum

    def forward(self):
        self.output_layer = self.active()

    def backward(self, error):
        self.bias_delta = sum(error, axis=(0, 2, 3))
        self.bias_delta = self.bias_delta.reshape(self.kernel_cnt, -1)

        error_cols = error.transpose(1, 2, 3, 0).reshape(self.kernel_cnt, -1)
        self.kernel_delta = error_cols.dot(self.input_cols.T)
        self.kernel_delta = self.kernel_delta(self.kernel.shape)

        kernel_cols = self.kernel.reshape(self.kernel_cnt, -1)
        prev_error_cols = kernel_cols.T.dot(error_cols)
        prev_error = col2im(prev_error_cols, self.input_layer.shape,
                            self.kernel_size, self.kernel_size,
                            padding=self.padding, stride=self.stride)

        return prev_error


class ReLUSynapse(Synapse):

    def __init__(self, input_layer):
        self.input_layer = input_layer
        self.output_layer = None

    def active(self):
        self.output_layer = maximum(self.input_layer, 0)  # here 0 has been broadcast

    def derivative(self):
        return 1


class PoolingSynapse(Synapse):

    def __init__(self, input_layer, pool_size, stride):
        self.input_layer = input_layer
        self.output_layer = None
        self.pool_size = pool_size
        self.stride = stride

        depth, height, width = self.input_layer.shape
        self.rf_width = (width - self.pool_size) * self.stride + 1
        self.rf_height = (height - self.pool_size) * self.stride + 1

    def pooling(self, func):
        depth, height, width = self.input_layer.shape

        for ix in range(depth):
            matrix = self.input_layer[ix, :, :].reshape(height, width)
            cols = im2col(matrix, self.pool_size, self.pool_size, 0, self.stride)

            pooling_matrix = func(cols)
            pooling_matrix = pooling_matrix.reshape((self.rf_height, self.rf_width, 1))
            if not self.output_layer:
                self.output_layer = pooling_matrix
            else:
                self.output_layer = concatenate((self.output_layer, pooling_matrix), axis=0)


class MaxPoolingSynapse(PoolingSynapse):

    def __init__(self, input_layer, output_layer, pool_size, stride):
        """
        MaxPooling select the maximum value of a matrix, we should remember the
        index of the maximum one and use it to back propagate the error to the
        correct neuron
        :param pool_size: the size of pooling layer
        """
        super(MaxPoolingSynapse, self).__init__(input_layer, output_layer, pool_size, stride)

    def pooling(self):
        pass


class AvgPoolingSynapse(PoolingSynapse):

    def __init__(self, input_layer, output_layer, pool_size, stride):
        super(AvgPoolingSynapse, self).__init__(input_layer, output_layer, pool_size, stride)

    def pooling(self):
        pass
