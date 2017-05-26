#! /usr/bin/env python
# -*- coding: utf-8 -*-

from cPickle import dump, load
from cPickle import HIGHEST_PROTOCOL
from conv_synapse import ConvSynapse
from conv_synapse import ReLUSynapse
from conv_synapse import MaxPoolingSynapse
from numpy import sum
from numpy import exp
from numpy import log
from numpy import linalg
from numpy import nan_to_num
import os
import sys


class ConvolutionalNet(object):

    def __init__(self, kernel_size, kernel_cnt, types_of_each_synapse=None,
                 padding=1, stride=1, learning_rate=0.01, momentum=0.7):
        self.kernel_size = kernel_size
        self.kernel_cnt = kernel_cnt
        self.padding = padding
        self.stride = stride
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.full_connected = None
        self._init_full_connected()

        # default architecture is:
        #  Conv -> ReLU -> Conv -> ReLu -> Pooling ->
        #  Conv -> ReLU -> Conv -> ReLu -> Pooling ->
        # Fully-connected multi-layer perceptron
        if types_of_each_synapse is None:
            self.types_of_each_synapse = "CRCRMCRCRM"
        else:
            self.types_of_each_synapse = types_of_each_synapse

        self.synapses = []
        self._connect_each_layer()

    def _connect_each_layer(self):
        for synapse_type in self.types_of_each_synapse:
            if synapse_type == 'C':
                synapse = ConvSynapse(self.kernel_size, self.kernel_cnt,
                                      self.padding, self.stride,
                                      self.learning_rate, self.momentum)
                self.synapses.append(synapse)
            elif synapse_type == 'R':
                synapse = ReLUSynapse()
                self.synapses.append(synapse)
            elif synapse_type == 'M':
                synapse = MaxPoolingSynapse(2, 0, 2)
                self.synapses.append(synapse)
            else:
                print "Unsupported type - '%s' of synapse" % synapse_type
                sys.exit(-1)

    def _init_full_connected(self):
        base_path = os.path.dirname(os.path.realpath('.')).split(os.sep)
        module_path = os.sep.join(base_path + ['mlp'])
        sys.path.append(module_path)

        from mlperceptron import MLPerceptron
        # FIXME: how to determin the size of multi layer perceptron
        self.full_connected = MLPerceptron([960, 480, 10], "SM", 'C', self.learning_rate, self.momentum)

    def feed_forward(self, inputs):
        prev_output = inputs
        self.synapses[0].input_layer = inputs

        for synapse in self.synapses:
            synapse.input_layer = prev_output
            synapse.feed_forward()
            prev_output = synapse.output_layer

        full_input = prev_output.reshape(prev_output.shape[0], -1)
        final_output = self.full_connected.feed_forward(full_input.T)
        return final_output

    def back_propagated(self, error):
        error = self.full_connected.back_propagated(error)
        error = error.reshape(self.synapses[-1].output_layer.shape)

        for synapse in reversed(self.synapses):
            error = synapse.back_propagated(error)

        return error

    def single_loop(self, inputs, target):
        # here inputs and target are both (n, 1) column vector
        outputs = self.feed_forward(inputs)
        error = outputs - target
        self.back_propagated(error)

        return self.error_cost(outputs, target, 'C')

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
        print outputs.shape

        cost = sum(nan_to_num(
            -target * log(outputs + tiny)), axis=1, keepdims=True) / batch_size
        return sum(cost)

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
