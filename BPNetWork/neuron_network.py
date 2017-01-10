#! /usr/bin/env python
# -*- coding:utf-8 -*-

from numpy import random
from numpy import ones
from numpy import zeros
from numpy import rint
from numpy import all
from numpy import sum
from nn_utils import Sigmoid
from nn_utils import QuadraticCost


class NeuronNetwork(object):

    def __init__(self, neurons_of_each_layer, activator=None, evaluator=None, learning_rate=0.1, momentum=0.9):

        """
        The list ``neurons_of_each_layer`` contains the number of neurons in the
        respective layers of this network. For example, if the list was [4, 6, 4]
        then it would be a three-layer network, with the first layer containing 4
        neurons, 6-neurons-hidden-layer and 4-neurons-output-layer
        """
        self.neurons_of_each_layer = neurons_of_each_layer
        self.synapses = []
        self.learning_rate = learning_rate
        self.momentum = momentum

        # default activated function is sigmoid
        self.activator = Sigmoid() if activator is None else activator
        # default error evaluated function is mean squared error
        self.evaluator = QuadraticCost(Sigmoid()) if evaluator is None else evaluator

        self._connect_layers()

    def _connect_layers(self):
        neurons_this_layer = self.neurons_of_each_layer[: -1]
        neurons_next_layer = self.neurons_of_each_layer[1:]

        for num_this_layer, num_next_layer in zip(neurons_this_layer, neurons_next_layer):
            self.synapses.append(Synapse(num_this_layer, num_next_layer, self.learning_rate, self.momentum))

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

    def _single_loop(self, inputs, target):
        # here inputs and target are both (n, 1) column vector
        outputs = self.feed_forward(inputs)
        error = self.evaluator.delta(outputs, target)

        self.back_propagated(error)

        return self.evaluator.evaluate(outputs, target)

    def train(self, inputs, target, batch_size=1, epoch=3000):
        row, col = inputs.shape

        times = 1
        while times <= epoch:
            total_error = 0
            idx = 0
            while idx < row:
                start = idx
                end = idx + batch_size if idx + batch_size < row else row
                counts = end - start

                each_input = inputs[start: end, :].T
                each_target = target[start: end, :].T

                error = self._single_loop(each_input, each_target)
                total_error += error
                idx += counts

            # if times % 100 == 0:
            print "Epoch %d" % times
            print "Total Error: %3.4f" % total_error
            times += 1
        else:
            print "Epoch %d" % (times - 1)
            print "Total Error: %3.4f" % total_error

    def validate(self, inputs, target):
        row, col = inputs.shape
        output_neurons = self.synapses[-1].output_neurons
        correct = 0

        idx = 0
        while idx < row:
            each_input = inputs[idx, :].reshape(col, 1)
            each_target = target[idx, :].reshape(output_neurons, 1)

            outputs = self.feed_forward(each_input)
            outputs = rint(outputs)
            if all(outputs == each_target):
                correct += 1
            else:
                print outputs, each_target

            idx += 1

        print "Correct Percentage: %3.2f%%" % (correct * 100.0 / row)


class Synapse(object):

    def __init__(self, input_neurons, output_neurons, learning_rate, momentum):
        self.weight = learning_rate * random.normal(size=(input_neurons, output_neurons))
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.bias = ones((self.output_neurons, 1))
        self.bias_delta = zeros((self.output_neurons, 1))
        self.weight_delta = zeros((input_neurons, output_neurons))

        self.inputs = None
        self.outputs = None
        self.batch_size = 1

    def feed_forward(self, activator):
        """According to the specified active function to compute
        the output, this function will return the output and the
        network can use it as input to compute the next layer's output
        """
        def func(x): return activator.activate(x)
        self.batch_size = self.inputs.shape[1]
        self.outputs = func(self.weight.T.dot(self.inputs) + self.bias.dot(ones((1, self.batch_size))))

        return self.outputs

    def back_propagated(self, activator, error):
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

        self.weight_delta = self.momentum * self.weight_delta + gradient / self.batch_size
        self.weight -= self.learning_rate * self.weight_delta

        error_sum = sum(error_derv, axis=1).reshape((error_derv.shape[0], 1))
        self.bias_delta = self.momentum * self.bias_delta + error_sum / self.batch_size
        self.bias -= self.learning_rate * self.bias_delta

        return prev_error
