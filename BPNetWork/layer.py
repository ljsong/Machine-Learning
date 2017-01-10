#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
from numpy.random import normal
from numpy import linalg


class Layer(object):
    def __init__(self, num_neurons, neurons=None):
        self.num_neurons = num_neurons
        self.neurons = normal(size=(num_neurons, 1)) if neurons is None else neurons

    def active(self, x):
        pass

    def derivative(self, y):
        pass

    def squared_error(self, target):
        """Return the cost associated with an output ``output`` and
        desired output ``target``"""

        return 0.5 * linalg.norm(output - target) ** 2

    def cross_entropy(self, target):
        pass

    def forward(self):
        pass

    def backward(self):
        pass


class SigmoidLayer(Layer):
    def __init__(self, num_neurons, neurons=None):
        super(SigmoidLayer, self).__init__(num_neurons, neurons)

    def active(self, x):
        return 1.0 / (1 + numpy.exp(-x))

    def derivative(self, y):
        return y * (1 - y)

    def forward(self):
        pass

    def backward(self):
        pass


class LinearLayer(Layer):
    """
    Output is a simple linear transformation of input, this layer can
    only deal with some linear classification problem
    """
    def __init__(self, num_neurons, neurons=None):
        super(LinearLayer, self).__init__(num_neurons, neurons)

    def forward(self):
        pass

    def backward(self):
        pass


class SoftmaxLayer(Layer):
    """
    This layer is often used as the last layer which is an output layer
    """
    def __init__(self, num_neurons, neurons=None):
        super(SoftmaxLayer, self).__init__(num_neurons, neurons)

    def forward(self):
        pass

    def backward(self):
        pass


class HyperTanLayer(Layer):
    """
    This class use hyperbolic tangent function as activation function
    and use its derivative to decide the direction of gradient
    """
    def __init__(self, num_neurons, neurons):
        super(HyperTanLayer, self).__init__(num_neurons, neurons)

    @staticmethod
    def tangent(x):
        pass

    def forward(self):
        pass

    def backward(self):
        pass
