#! /usr/bin/env python
# -*- coding: utf-8 -*-

from numpy.random import normal
# from im2col import im2col


class ConvSyanpse(object):

    def __init__(self, kernel_size, input_layer, output_layer):
        """
        This class define a convolutional layer which has a filter matrix,
        input vector and output vector
        :param kernel_size: this should be a number which indicate the size of filter matrix
        :param input_layer: this vector store the value of input which should be 3-D image
        :param output_layer: convolutional sum
        """
        self.input_layer = input_layer
        self.output_layer = output_layer
        depth = input_layer.shape[0]

        self.kernel = normal(size=(depth, kernel_size, kernel_size))


class ReLUSynapse(object):

    def __init__(self):
        pass


class PoolingSynapse(object):

    def __init__(self, pool_size):
        pass


class MaxPoolingSynapse(PoolingSynapse):

    def __init__(self, pool_size):
        super(MaxPoolingSynapse, self).__init__(pool_size)


class AvgPoolingSynapse(PoolingSynapse):

    def __init__(self, pool_size):
        super(AvgPoolingSynapse, self).__init__(pool_size)
