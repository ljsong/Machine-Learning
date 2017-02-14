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
    def __init__(self, kernel_size=3, kernel_cnt=1, padding=1, stride=1,
                 learning_rate=0.1, momentum=0.7):
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
        self.learning_rate = learning_rate
        self.momentum = momentum

        # input_layer after transforming by im2col
        self.input_cols = None
        self.output_layer = None

        self.bias_delta = None
        self.kernel_delta = None

    def __setattr__(self, key, value):
        if key == 'input_layer' and not hasattr(self, 'input_layer'):
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
            self.batch_size = number
        else:
            super(ConvSynapse, self).__setattr__(key, value)

    def active(self):
        self.input_cols = im2col(self.input_layer, self.kernel_size,
                                 self.kernel_size, self.padding, self.stride)

        kernel_cols = self.kernel.reshape(self.kernel_cnt, -1)
        conv_sum = kernel_cols.dot(self.input_cols) + self.bias
        print self.kernel_cnt, self.rf_height, self.rf_width, self.batch_size
        conv_sum = conv_sum.reshape(self.kernel_cnt, self.rf_height, self.rf_width, self.batch_size)

        conv_sum = conv_sum.transpose(3, 0, 1, 2)
        return conv_sum

    def derivative(self):
        return 1

    def feed_forward(self):
        self.output_layer = self.active()

    def back_propagated(self, error):
        error_derv = error * self.derivative(self.output_layer)

        error_sum = sum(error_derv, axis=(0, 2, 3))
        self.bias_delta = \
            self.momentum * self.bias_delta + error_sum / self.batch_size
        self.bias -= self.learning_rate * self.bias_delta

        error_cols = error_derv.transpose(1, 2, 3, 0).reshape(self.kernel_cnt, -1)
        gradient = self.input_cols.dot(error_cols.T)
        self.kernel_delta = \
            self.momentum * self.kernel_delta + gradient / self.batch_size
        self.kernel -= \
            self.learning_rate * self.kernel_delta.reshape(self.kernel.shape)

        kernel_cols = self.kernel.reshape(self.kernel_cnt, -1)
        prev_error_cols = kernel_cols.T.dot(error_cols)
        prev_error = col2im(prev_error_cols, self.input_layer.shape,
                            self.kernel_size, self.kernel_size,
                            padding=self.padding, stride=self.stride)

        return prev_error


class ReLUSynapse(Synapse):
    def __init__(self, learning_rate=0.1, momentum=0.7):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.output_layer = None
        self.weight_delta = None
        self.bias_delta = None

    def __setattr__(self, key, value):
        if key == 'input_layer' and not hasattr(self, 'input_layer'):
            super(ReLUSynapse, self).__setattr__('input_layer', value)
            number, channel, height, width = self.input_layer.shape
            self.input_reshaped = self.input_layer.reshape(number, -1)
            self.input_reshape = self.input_reshaped.T

            wsize = self.input_reshaped.shape[0]
            self.weight = normal(size=(wsize, wsize))
            self.bias = ones((wsize, 1))

            self.weight_delta = zeros((wsize, wsize))
            self.bias_delta = zeros((wsize, 1))
        else:
            super(ReLUSynapse, self).__setattr__(key, value)

    def active(self):
        # should be W' @ X, but here weight is a matrix
        outputs = self.weight.dot(self.input_reshaped)
        outputs += self.bias
        outputs = maximum(outputs, 0)

        return outputs.reshape(self.input_layer.shape)

    def derivative(self):
        cond_matrix = self.output_layer > 0

        return minimum(cond_matrix, 1)

    def feed_forward(self):
        self.output_layer = self.active()

    def back_propagated(self, error):
        error_derv = error * self.derivative()
        gradient = self.input_layer.dot(error_derv.T)

        prev_error = self.weight.dot(error_derv)

        self.weight_delta = \
            self.momentum * self.weight_delta + gradient / self.batch_size
        self.weight -= self.learning_rate * self.weight_delta

        error_sum = sum(error_derv, axis=1, keepdims=True)
        self.bias_delta = \
            self.momentum * self.bias_delta + error_sum / self.batch_size
        self.bias -= self.learning_rate * self.bias_delta

        return prev_error


class PoolingSynapse(Synapse):
    def __init__(self, pool_size=2, padding=0, stride=2):
        self.output_layer = None
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding

        self.input_cols = None
        self.max_idx = None

    def __setattr__(self, key, value):
        if key == 'input_layer' and not hasattr(PoolingSynapse, 'input_layer'):
            super(PoolingSynapse, self).__setattr__('input_layer', value)
            number, channel, height, width = self.input_layer.shape
            self.rf_width = (width - self.pool_size) / self.stride + 1
            self.rf_height = (height - self.pool_size) / self.stride + 1
        else:
            super(PoolingSynapse, self).__setattr__(key, value)

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

    def __setattr__(self, key, value):
        super(MaxPoolingSynapse, self).__setattr__(key, value)

    def pooling(self):
        number, channel, height, width = self.input_layer.shape
        input_reshaped = self.input_layer.reshape(number * channel, 1, height, width)

        self.input_cols = im2col(input_reshaped, self.pool_size,
                                 self.pool_size, self.padding, self.stride)
        self.max_idx = argmax(self.input_cols, axis=0)

        self.output_layer = self.input_cols[self.max_idx, range(self.max_idx.size)]
        self.output_layer = self.output_layer.reshape(self.rf_height, self.rf_width, number, channel)
        self.output_layer = self.output_layer.transpose(2, 3, 0, 1)

    def feed_forward(self):
        self.pooling()

    def back_propagated(self, error):
        number, channel, height, width = self.input_layer.shape
        input_delta = zeros_like(self.input_cols)
        error_flat = error.transpose(2, 3, 0, 1).ravel()

        input_delta[self.max_idx, range(self.max_idx.size)] = error_flat
        prev_error = col2im(input_delta, (number * channel, 1, height, width),
                            self.pool_size, self.pool_size, self.padding,
                            self.stride)
        prev_error = prev_error.reshape(self.input_layer.shape)

        return prev_error


class AvgPoolingSynapse(PoolingSynapse):
    def __init__(self, pool_size, stride):
        super(AvgPoolingSynapse, self).__init__(pool_size, stride)

    def __setattr__(self, key, value):
        super(AvgPoolingSynapse, self).__setattr__(key, value)

    def pooling(self):
        pass

    def feed_forward(self):
        pass

    def back_propagated(self, error):
        pass
