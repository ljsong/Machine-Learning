#! /usr/bin/env python
# -*- coding:utf-8 -*-

import numpy
from numpy import random
from numpy import ones
from numpy import zeros
from numpy import exp


class NeuronNetwork(object):

    def __init__(self, num_layers, neurons_of_each_layer):
        self.num_layers = num_layers
        self.neurons_of_each_layer = neurons_of_each_layer
        self.layers = []
        self.activation_func = lambda x: 1 / (1 + exp(-x))

        self._create_layers()

    def _create_layers(self):
        for num_neurons in self.neurons_of_each_layer:
            self.layers.append(Layer(num_neurons))

    def set_activation_function(self, activation_function):
        self.activation_func = activation_function

    def train(self, inputs, target):
        pass

    def feed_forward(self):
        pass

    def back_propagated(self):
        pass

    def predict(self, inputs, target):
        pass


class Layer(object):
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.values = zeros((num_neurons, 1))


class Synapse(object):

    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.weight = random.random((features.num_neurons, target.num_neurons))
        self.bias = ones((target.num_neurons, 1))


def main():
    nerual_network = NeuronNetwork(3, [4, 6, 4])


if __name__ == '__main__':
    main()
