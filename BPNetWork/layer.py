#! /usr/bin/env python
# -*- coding: utf-8 -*-

from numpy.random import normal


class Layer(object):
    def __init__(self, num_neurons, values=None):
        self.num_neurons = num_neurons
        # values means the input of this layer
        self.values = normal(size=(num_neurons, 1)) if values is None else values
        self.shape = values.shape if values is not None else [num_neurons, 1]

