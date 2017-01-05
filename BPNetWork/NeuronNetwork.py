#! /usr/bin/env python
# -*- coding:utf-8 -*-

from numpy import random
from numpy import ones
from numpy import zeros
from numpy import exp
from Utils import Sigmoid
from Utils import QuadraticCost


class NeuronNetwork(object):

    def __init__(self, neurons_of_each_layer, activator=None, evaluator=None):

        """
        The list ``neurons_of_each_layer`` contains the number of neurons in the
        respective layers of this network. For example, if the list was [4, 6, 4]
        then it would be a three-layer network, with the first layer containing 4
        neurons, 6-neurons-hidden-layer and 4-neurons-output-layer
        """
        self.neurons_of_each_layer = neurons_of_each_layer
        self.layers = []
        self.synapses = []

        # default activated function is sigmoid
        self.activator = Sigmoid() if activator is None else activator
        # default error evaluated function is mean squared error
        self.evaluator = QuadraticCost(Sigmoid()) if evaluator is None else evaluator

        self._connect_layers()

    def _connect_layers(self):
        for num_neurons in self.neurons_of_each_layer:
            self.layers.append(Layer(num_neurons))

        input_layers = self.layers[: -1]
        output_layers = self.layers[1:]

        for input_layer, output_layer in zip(input_layers, output_layers):
            self.synapses.append(Synapse(input_layer, output_layer))

    def _single_loop(self, inputs, target):
        # here inputs and target are both (n, 1) column vector
        self.synapses[0].inputs = inputs

        outputs = self.feed_forward(inputs)
        error = self.evaluator.delta(outputs, target)

        self.back_propagated(error)

        return self.evaluator.evaluate(outputs, target)

    def feed_forward(self, inputs):
        prev_layer_output = inputs
        for idx, synapse in enumerate(self.synapses):
            self.synapses[idx].inputs = prev_layer_output
            prev_layer_output = synapse.feed_forward(self.activator)

        outputs = prev_layer_output
        return outputs

    def back_propagated(self, error):
        for synapse in reversed(self.synapses):
            error = synapse.back_propagated(self.activator, error)

    def train(self, inputs, target, epoch=3000):
        row, col = inputs.shape
        output_neurons = self.layers[-1].num_neurons

        times = 1
        idx = 0

        while times <= epoch:

            total_error = 0
            idx = 0
            while idx < row:
                each_input = inputs[idx, :].reshape(col, 1)
                each_target = target[idx, :].reshape(output_neurons, 1)

                error = self._single_loop(each_input, each_target)
                total_error += error
                idx += 1

            print total_error
            times += 1

    def validate(self, inputs, target):
        pass


class Layer(object):
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.values = zeros((num_neurons, 1))


class Synapse(object):

    def __init__(self, inputs, outputs):
        self.inputs = inputs.values
        self.outputs = outputs.values
        self.weight = random.random((inputs.num_neurons, outputs.num_neurons))
        self.bias = ones((outputs.num_neurons, 1))

    def feed_forward(self, activator):
        def func(x): return activator.activate(x)
        self.outputs = func(self.weight.T.dot(self.inputs) + self.bias)

        return self.outputs

    def back_propagated(self, activator, error):
        def derivative(x): return activator.derivative(x)
        error_derv = error * derivative(self.outputs)
        gradient = self.inputs.dot(error_derv.T)

        self.weight -= gradient
        self.bias -= error

        return self.weight.dot(error_derv)



def main():
    nerual_network = NeuronNetwork([4, 6, 3])


if __name__ == '__main__':
    main()
