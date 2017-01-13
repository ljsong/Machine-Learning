#! /usr/bin/env python
# -*- coding:utf-8 -*-

from numpy import exp
from numpy import nan_to_num
from numpy import linalg
from numpy import log
from numpy import sum
from cPickle import dump
from cPickle import HIGHEST_PROTOCOL
from cPickle import load
from synapse import SynapseFactory
from layer import Layer


class NeuralNetwork(object):

    def __init__(self, neurons_of_each_layer, types_of_each_synapse,
                 cost_type='S', learning_rate=0.1, momentum=0.9):

        """
        The list ``neurons_of_each_layer`` contains the number of neurons in the
        respective layers of this network. For example, if the list was [4,6,4]
        then it would be a three-layer network, with the first layer containing
        4 neurons, 6-neurons-hidden-layer and 4-neurons-output-layer

        The list "types_of_each_synapse" contains the type of synapses in this
        network. Each synapse contains two layer: input layer and output layer,
        the values of output layer is determined by active function and the
        values of input layer. Default is Linear Synapse

        Currently, there are several valid types of synapse as follows:
        'S': Sigmoid Synapse
        'L': Linear Synapse(default)
        'T': Tangent Synapse
        'M': Softmax Synapse

        The cost_type means which cost function should be used to evaluate
        the error, currently there are two methods as follows:

        'S': Squared Error(default)
        'C': Cross Entropy
        """

        self.neurons_of_each_layer = neurons_of_each_layer
        self.synapses = []
        self.layers = []
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.cost_type = cost_type

        if types_of_each_synapse is None:
            self.types_of_each_synapse = 'L' * len(self.neurons_of_each_layer)
        else:
            self.types_of_each_synapse = types_of_each_synapse

        self._connect_layers()

    def _connect_layers(self):
        for num_neurons in self.neurons_of_each_layer:
            self.layers.append(Layer(num_neurons))

        this_layers = self.layers[: -1]
        next_layers = self.layers[1:]

        for this_layer, next_layer, synapse_type in zip(
                this_layers, next_layers, self.types_of_each_synapse):
            synapse = SynapseFactory.create_synapse(synapse_type)
            synapse.init(this_layer, next_layer,
                         self.learning_rate, self.momentum)
            self.synapses.append(synapse)

    def feed_forward(self, inputs):
        self.synapses[0].input_layer.values = inputs

        for synapse in self.synapses:
            synapse.feed_forward()

        return self.synapses[-1].output_layer.values

    def back_propagated(self, error):
        for synapse in reversed(self.synapses):
            error = synapse.back_propagated(error)

    def single_loop(self, inputs, target):
        # here inputs and target are both (n, 1) column vector
        outputs = self.feed_forward(inputs)
        error = outputs - target
        self.back_propagated(error)

        return self.error_cost(outputs, target, self.cost_type)

    @classmethod
    def _squared_error(cls, outputs, target):
        """Return the cost associated with an output `output` and
        desired output `target`"""

        batch_size = outputs.shape[1]
        return 0.5 * linalg.norm(outputs - target) / batch_size

    @classmethod
    def _cross_entropy(cls, outputs, target):
        tiny = exp(-30)
        batch_size = outputs.shape[1]
        return sum(nan_to_num(
            -target * log(outputs + tiny))) / batch_size

    @classmethod
    def error_cost(cls, outputs, target, func='S'):
        if func == 'S':
            return cls._squared_error(outputs, target)
        elif func == 'C':
            return cls._cross_entropy(outputs, target)
        else:
            raise AttributeError("Can't find a valid active function "
                                 "of key %s to compute the error cost!" % func)

    @staticmethod
    def to_file(network, file_path='.', file_name='network.pkl'):
        from os import sep as separator
        abs_path = separator.join((file_path, file_name))

        with open(abs_path, 'wb') as outputs:
            dump(network, outputs, HIGHEST_PROTOCOL)

    @staticmethod
    def from_file(file_path='./network.pkl'):
        with open(file_path, 'rb') as inputs:
            network = load(inputs)

        return network
