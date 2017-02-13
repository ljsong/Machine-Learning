#! /usr/bin/env python
# -*- coding: utf-8 -*-

from numpy.random import normal
from numpy import maximum
from numpy import minimum
from numpy import ones
from numpy import argmax
from numpy import sum
from numpy import zeros_like
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

    def feed_forward(self):
        pass

    def back_propagated(self):
        pass


class ConvSynapse(Synapse):

    def __init__(self, kernel_size=3, kernel_cnt=1, padding=1, stride=1):
        """
        This class define a convolutional layer which has a filter matrix,
        input vector and output vector
        :param kernel_size: this should be a number which indicate the size of filter matrix
        :param kernel_cnt: the number of the filters
        :param input_layer: this vector store the value of input which should be 3-D image
        :param padding: how many zeros should be padded around the input value
        :param stride: the step which kernel go through the input value
        """
        self.kernel_size = kernel_size
        self.kernel_cnt = kernel_cnt
        self.padding = padding
        self.stride = stride

        # input_layer after transforming by im2col
        self.input_cols = None
        self.output_layer = None

        self.bias_delta = None
        self.kernel_delta = None

    def __setattr__(self, key, value):
        if key == 'input_layer' and not hasattr(ConvSynapse, 'input_layer'):
            super(ConvSynapse, self).__setattr__('input_layer', value)
            number, channel, height, width = self.input_layer.shape
            # each row is a filter
            self.kernel = normal(size=(
                self.kernel_cnt, channel, self.kernel_size, self.kernel_size))
            self.bias = ones((self.kernel_cnt, 1))

            # the counts of receptive filed
            self.rf_height = (height - self.kernel_size +
                              2 * self.padding) / self.stride + 1
            self.rf_width = (width - self.kernel_size +
                             2 * self.padding) / self.stride + 1
        else:
            super(ConvSynapse, self).__setattr__(key, value)

    def active(self):
        inputs = self.input_layer.reshape(1, self.input_layer.shape)
        self.input_cols = im2col(inputs, self.kernel_size,
                                 self.kernel_size, self.padding, self.stride)

        kernel_cols = self.kernel.reshape(self.kernel_cnt, -1)
        conv_sum = kernel_cols.dot(self.input_cols) + self.bias
        conv_sum = conv_sum.reshape(self.kernel_cnt, self.rf_height, self.rf_width, 1)

        conv_sum = conv_sum.transpose(3, 0, 1, 2)
        return conv_sum

    def derivative(self):
        return 1

    def feed_forward(self):
        self.output_layer = self.active()

    def back_propagated(self, error):
        error_derv = error * self.derivative(self.output_layer.values)
        self.bias_delta = sum(error_derv, axis=(0, 2, 3))
        self.bias_delta = self.bias_delta.reshape(self.kernel_cnt, -1)

        error_cols = error_derv.transpose(1, 2, 3, 0).reshape(self.kernel_cnt, -1)
        self.kernel_delta = error_cols.dot(self.input_cols.T)
        self.kernel_delta = self.kernel_delta(self.kernel.shape)

        kernel_cols = self.kernel.reshape(self.kernel_cnt, -1)
        prev_error_cols = kernel_cols.T.dot(error_cols)
        prev_error = col2im(prev_error_cols, self.input_layer.shape,
                            self.kernel_size, self.kernel_size,
                            padding=self.padding, stride=self.stride)

        return prev_error


class ReLUSynapse(Synapse):

    def __init__(self):
        self.input_layer = None
        self.output_layer = None

    def active(self):
        self.output_layer = maximum(self.input_layer, 0)  # here 0 has been broadcast

    def derivative(self):
        return minimum(self.output_layer, 1)

    def feed_forward(self):
        pass

    def back_propagated(self, error):
        return self.derivative() * error


class PoolingSynapse(Synapse):

    def __init__(self, pool_size, stride):
        self.output_layer = None
        self.pool_size = pool_size
        self.stride = stride

        self.input_cols = None
        self.max_idx = None

    def __setattr__(self, key, value):
        if key == 'input_layer' and not hasattr(PoolingSynapse, 'input_layer'):
            super(PoolingSynapse, self).__setattr__('input_layer', value)
            number, channel, height, width = self.input_layer.shape
            self.rf_width = (width - self.pool_size) * self.stride + 1
            self.rf_height = (height - self.pool_size) * self.stride + 1
        else:
            super(PoolingSynapse, self).__setattr__(key, value)

    def pooling(self, func):
        number, channel, height, width = self.input_layer.shape
        input_reshaped = self.input_layer.reshape(number * channel, 1, height, width)

        self.input_cols = im2col(input_reshaped, self.pool_size, self.pool_size, 0, self.stride)
        self.max_idx = func(self.input_cols, axis=0)

        self.output_layer = self.input_cols[self.max_idx, range(self.max_idx.size)]
        self.output_layer = self.output_layer.reshape(self.rf_height, self.rf_width, number, channel)
        self.output_layer = self.output_layer.transpose(2, 3, 0, 1)

    def derivative(self):
        return 1

    def feed_forward(self):
        pass

    def back_propagated(self, error):
        pass


class MaxPoolingSynapse(PoolingSynapse):

    def __init__(self, pool_size, stride):
        """
        MaxPooling select the maximum value of a matrix, we should remember the
        index of the maximum one and use it to back propagate the error to the
        correct neuron
        :param pool_size: the size of pooling layer
        """
        super(MaxPoolingSynapse, self).__init__(pool_size, stride)

    def __setattr__(self, key, value):
        super(MaxPoolingSynapse, self).__setattr__(key, value)

    def pooling(self):
        super(MaxPoolingSynapse, self).pooling(lambda x: argmax(x, axis=0))

    def feed_forward(self):
        self.pooling()

    def back_propagated(self, error):
        number, channel, height, width = self.input_layer.shape
        input_delta = zeros_like(self.input_cols)
        error_flat = error.transpose(2, 3, 0, 1).ravel()

        input_delta[self.max_idx, range(self.max_idx.size)] = error_flat
        prev_error = col2im(input_delta, (number * channel, 1, height, width),
                            self.pool_size, self.pool_size, 0, self.stride)
        prev_error = prev_error.reshape(self.input_layer.shape)

        return prev_error


class AvgPoolingSynapse(PoolingSynapse):

    def __init__(self, pool_size, stride):
        super(AvgPoolingSynapse, self).__init__(pool_size, stride)

    def __setattr__(self, key, value):
        super(MaxPoolingSynapse, self).__setattr__(key, value)

    def pooling(self):
        super(AvgPoolingSynapse, self).pooling(lambda x: sum(x, axis=0) / x.shape[0])

    def feed_forward(self):
        pass

    def back_propagated(self, error):
        pass
