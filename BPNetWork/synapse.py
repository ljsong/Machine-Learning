#! /usr/bin/env python
# -*- coding: utf-8 -*-

from numpy.random import normal
from numpy import ones
from numpy import zeros
from numpy import exp
from numpy import amax
from numpy import sum
from numpy import errstate


class Synapse(object):

    def __init__(self, input_layer=None, output_layer=None,
                 learning_rate=0.1, momentum=0.5):
        self.batch_size = 1
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.input_neurons = self.output_neurons = 0
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight = self.bias = self.bias_delta = self.weight_delta = None

    def init(self, input_layer, output_layer, learning_rate, momentum):

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.input_layer = input_layer
        self.output_layer = output_layer

        if not self.input_layer or not self.output_layer:
            return

        if input_layer.shape[1] != output_layer.shape[1]:
            raise AttributeError("The second dimension of input and output is"
                                 " not matched which means they may have "
                                 "different batch size!")
        else:
            self.batch_size = self.input_layer.shape[1]

        self.input_neurons = input_layer.num_neurons
        self.output_neurons = output_layer.num_neurons
        self.weight = learning_rate * normal(
            size=(self.input_neurons, self.output_neurons))

        self.bias = ones((self.output_neurons, 1))
        self.bias_delta = zeros((self.output_neurons, 1))
        self.weight_delta = zeros((self.input_neurons, self.output_neurons))

    def active(self):
        """
        This function computes the result by using specified active function
        """
        return self.output_layer.values

    def derivative(self, y):
        """
        This function returns a derivative with respect to the active function
        """
        return 1

    def feed_forward(self):
        """According to the specified active function to compute
        the output, this function will return the output and the
        network can use it as input to compute the next layer's output
        """
        self.output_layer.values = self.active()

    def back_propagated(self, error):
        """According to the error comes from the next layer to update
        the weight matrix and bias vector, this function will compute
        the error of previous layer and return the error.
        The formula is as follows: δˡ = ((wˡ⁺¹)ᵀ * δˡ⁺¹)⊙σ'(zˡ)
        And the function will return (wˡ⁺¹)ᵀ * δˡ⁺¹)
        """
        # According the formula: δˡ = ((wˡ⁺¹)ᵀ * δˡ⁺¹)⊙σ'(zˡ), we
        # need to multiply the error with the derivative of itself
        # output
        error_derv = error * self.derivative(self.output_layer.values)
        gradient = self.input_layer.values.dot(error_derv.T)

        # We need to multiply the input layer's derivative in the current
        # synapse because the input layer of this synapse is the output
        # layer of previous synapse
        prev_error = self.weight.dot(error_derv)

        self.weight_delta = \
            self.momentum * self.weight_delta + gradient / self.batch_size
        self.weight -= self.learning_rate * self.weight_delta

        error_sum = sum(error_derv, axis=1, keepdims=True)
        self.bias_delta = \
            self.momentum * self.bias_delta + error_sum / self.batch_size
        self.bias -= self.learning_rate * self.bias_delta

        return prev_error


class SigmoidSynapse(Synapse):
    def __init__(self, input_layer=None, output_layer=None,
                 learning_rate=0.1, momentum=0.5):
        super(SigmoidSynapse, self).__init__(
            input_layer, output_layer, learning_rate, momentum)

    def active(self):
        outputs = self.weight.T.dot(self.input_layer.values)
        outputs += self.bias.dot(ones((1, self.batch_size)))

        return 1.0 / (1 + exp(-outputs))

    def derivative(self, y):
        return y * (1 - y)

    def feed_forward(self):
        super(SigmoidSynapse, self).feed_forward()

    def back_propagated(self, error_derv):
        return super(SigmoidSynapse, self).back_propagated(error_derv)


class LinearSynapse(Synapse):
    def __init__(self, input_layer=None, output_layer=None,
                 learning_rate=0.1, momentum=0.5):
        super(LinearSynapse, self).__init__(
            input_layer, output_layer, learning_rate, momentum)

    def active(self):
        outputs = self.weight.T.dot(self.input_layer.values)
        outputs += self.bias.dot(ones((1, self.batch_size)))

        return outputs

    def derivative(self, y):
        return 1

    def feed_forward(self):
        super(LinearSynapse, self).feed_forward()

    def back_propagated(self, error_derv):
        return super(LinearSynapse, self).back_propagated(error_derv)


class SoftmaxSynapse(Synapse):
    def __init__(self, input_layer=None, output_layer=None,
                 learning_rate=0.1, momentum=0.5):
        super(SoftmaxSynapse, self).__init__(
            input_layer, output_layer, learning_rate, momentum)

    def active(self):
        outputs = self.weight.T.dot(self.input_layer.values)
        outputs += self.bias.dot(ones((1, self.batch_size)))

        max_unit = amax(outputs, axis=0, keepdims=True)
        identity = ones((self.output_neurons, 1))
        outputs -= identity.dot(max_unit)
        outputs = exp(outputs)

        all_sums = identity.dot(sum(outputs, axis=0, keepdims=True))
        outputs = outputs / all_sums

        return outputs

    def derivative(self, y):
        return 1

    def feed_forward(self):
        super(SoftmaxSynapse, self).feed_forward()

    def back_propagated(self, error_derv):
        return super(SoftmaxSynapse, self).back_propagated(error_derv)


class TangentSynapse(Synapse):
    def __init__(self, input_layer=None, output_layer=None,
                 learning_rate=0.1, momentum=0.5):
        super(TangentSynapse, self).__init__(
            input_layer, output_layer, learning_rate, momentum)

    def active(self):
        pass

    def derivative(self, y):
        pass

    def feed_forward(self):
        super(TangentSynapse, self).feed_forward()

    def back_propagated(self, error_derv):
        return super(TangentSynapse, self).back_propagated(error_derv)


class SynapseFactory(object):
    @staticmethod
    def create_synapse(synapse_type):
        if synapse_type == 'S':
            return SigmoidSynapse()
        elif synapse_type == 'L':
            return LinearSynapse()
        elif synapse_type == 'T':
            return TangentSynapse()
        elif synapse_type == 'M':
            return SoftmaxSynapse()
        else:
            raise AttributeError("Unsupported type of synapse, currently"
                                 "we only support these four S, L, T, M types!")
