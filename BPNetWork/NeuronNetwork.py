#! /usr/bin/env python
# -*- coding:utf-8 -*-

from numpy import random
from numpy import ones
from numpy import rint
from Utils import Sigmoid
from Utils import QuadraticCost


class NeuronNetwork(object):

    def __init__(self, neurons_of_each_layer, activator=None, evaluator=None, learning_rate=0.1):

        """
        The list ``neurons_of_each_layer`` contains the number of neurons in the
        respective layers of this network. For example, if the list was [4, 6, 4]
        then it would be a three-layer network, with the first layer containing 4
        neurons, 6-neurons-hidden-layer and 4-neurons-output-layer
        """
        self.neurons_of_each_layer = neurons_of_each_layer
        self.synapses = []
        self.learning_rate = learning_rate

        # default activated function is sigmoid
        self.activator = Sigmoid() if activator is None else activator
        # default error evaluated function is mean squared error
        self.evaluator = QuadraticCost(Sigmoid()) if evaluator is None else evaluator

        self._connect_layers()

    def _connect_layers(self):
        neurons_this_layer = self.neurons_of_each_layer[: -1]
        neurons_next_layer = self.neurons_of_each_layer[1:]

        for num_this_layer, num_next_layer in zip(neurons_this_layer, neurons_next_layer):
            self.synapses.append(Synapse(num_this_layer, num_next_layer, self.learning_rate))

    def feed_forward(self, inputs, z_axis):
        prev_layer_output = inputs
        for idx, synapse in enumerate(self.synapses):
            self.synapses[idx].inputs = prev_layer_output
            prev_layer_output = synapse.feed_forward(self.activator, z_axis)

        outputs = prev_layer_output
        return outputs

    def back_propagated(self, error, z_axis):
        for synapse in reversed(self.synapses):
            error = synapse.back_propagated(self.activator, error, z_axis)

    def _single_loop(self, inputs, target, z_axis):
        # here inputs and target are both (n, 1) column vector
        outputs = self.feed_forward(inputs, z_axis)
        error = self.evaluator.delta(outputs, target)

        self.back_propagated(error, z_axis)

        return self.evaluator.evaluate(outputs, target)

    def train(self, inputs, target, batch_size=1, epoch=3000):
        row, col = inputs.shape
        output_neurons = self.synapses[-1].output_neurons
        for synapse in self.synapses:
            synapse.add_bias(batch_size)

        times = 1
        while times <= epoch:
            total_error = 0
            idx = 0
            while idx < row:
                start = idx
                end = idx + batch_size if idx + batch_size < row else row
                z_axis = end - start

                each_input = inputs[start: end, :].reshape(col, z_axis)
                each_target = target[start: end, :].reshape(output_neurons, z_axis)

                error = self._single_loop(each_input, each_target, z_axis)
                total_error += error
                idx += z_axis

            if times % 100 == 0:
                print "Epoch %d" % times
                print "Total Error: %3.4f" % total_error
            times += 1

    def validate(self, inputs, target):
        row, col = inputs.shape
        output_neurons = self.synapses[-1].output_neurons
        correct = 0

        idx = 0
        while idx < row:
            each_input = inputs[idx, :].reshape(col, 1)
            each_target = target[idx, :].reshape(output_neurons, 1)

            outputs = self.feed_forward(each_input, 1)
            outputs = rint(outputs)
            if self.evaluator.evaluate(outputs, each_target) == 0:
                correct += 1
            else:
                print outputs, each_target

            idx += 1

        print "Correct Percentage: %3.4f%%" % (correct * 100.0 / row)


class Synapse(object):

    def __init__(self, input_neurons, output_neurons, learning_rate):
        self.weight = random.random((input_neurons, output_neurons))
        self.learning_rate = learning_rate
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons

        self.inputs = None
        self.outputs = None
        self.bias = None

    def add_bias(self, batch_size):
        """Add a bias matrix into this synapse"""
        self.bias = ones((self.output_neurons, batch_size))

    def feed_forward(self, activator, z_axis):
        """According to the specified active function to compute
        the output, this function will return the output and the
        network can use it as input to compute the next layer's output
        """
        def func(x): return activator.activate(x)
        self.outputs = func(self.weight.T.dot(self.inputs) + self.bias[:, :z_axis])

        return self.outputs

    def back_propagated(self, activator, error, z_axis):
        """According to the error comes from the next layer to update
        the weight matrix and bias vector, this function will compute
        the error of previous layer and return the error.
        The formula is as follows: δˡ = ((wˡ⁺¹)ᵀ * δˡ⁺¹)⊙σ'(zˡ)
        And the function will return (wˡ⁺¹)ᵀ * δˡ⁺¹)
        """
        def derivative(x): return activator.derivative(x)
        error_derv = error * derivative(self.outputs)
        gradient = self.inputs.dot(error_derv.T)

        prev_error = self.weight.dot(error_derv)
        self.weight -= self.learning_rate * gradient
        self.bias[:, :z_axis] -= self.learning_rate * error

        return prev_error
