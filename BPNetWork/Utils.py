#! /usr/bin/env python
# -*- coding:utf-8 -*-

from numpy import exp
from numpy import linalg
from numpy import sum
from numpy import nan_to_num
from numpy import log


class Activation(object):

    def activate(self, x):
        return x

    def derivative(self, y):
        return 1


class Sigmoid(Activation):

    def activate(self, x):
        return 1.0 / (1.0 + exp(-x))

    def derivative(self, y):
        return y * (1 - y)


class Linear(Activation):

    def activate(self, x):
        super(Linear, self).activate(x)

    def derivative(self, y):
        super(Linear, self).derivative(y)


class Cost(object):

    def __init__(self, activator):
        self.activator = activator

    def evaluate(self, output, target):
        return output - target

    def delta(self, output, target):
        return output - target


class QuadraticCost(Cost):

    def __init__(self, activator):
        super(QuadraticCost, self).__init__(activator)

    def evaluate(self, output, target):
        """Return the cost associated with an output ``output`` and
        desired output ``target``"""

        return 0.5 * linalg.norm(output - target) ** 2

    def delta(self, output, target):
        """Return the error delta from the output layer. """

        return output - target


class CrossEntropyCost(Cost):

    def evaluate(self, output, target):
        """Return the cost associated with an output ``output`` and
        desirec output ``target``. Note that numpy.nan_to_num is used
        to ensure numerical stability. In particular, if both ``output``
        and ``target`` have a 1.0 in the same slot, then the expression
        (1 - target) * numpy.log(1 - output) returns nan. The numpy.nan_to_num
        ensures that is converte to the correct value0.0
        """

        return sum(nan_to_num(-target * log(output) - (1 - target) * log(1 - output)))

    def delta(self, output, target):
        """Return the error delta from the output layer. """

        return (output - target) / (output * (1 - output))
