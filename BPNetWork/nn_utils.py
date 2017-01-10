#! /usr/bin/env python
# -*- coding:utf-8 -*-

from numpy import exp
from numpy import linalg
from numpy import sum
from numpy import log
from numpy import nan_to_num


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

    def evaluate(self, output, target):
        return output - target

    def delta(self, output, target):
        return output - target


class QuadraticCost(Cost):

    def evaluate(self, output, target):
        """Return the cost associated with an output ``output`` and
        desired output ``target``"""

        return 0.5 * linalg.norm(output - target) ** 2

    def delta(self, output, target):
        """Return the error delta from the output layer. """

        return output - target


class CrossEntropyCost(Cost):

    tiny = exp(-30)

    def evaluate(self, output, target):
        """Return the cost associated with an output ``output`` and
        desired output ``target``. Note that numpy.nan_to_num is used
        to ensure numerical stability. In particular, if both ``output``
        and ``target`` have a 1.0 in the same slot, then the expression
        (1 - target) * numpy.log(1 - output) returns nan. The numpy.nan_to_num
        ensures that is convert to the correct value0.0
        """
        batch_size = output.shape[1]
        return sum(nan_to_num(-target * log(output + self.tiny))) / batch_size

    def delta(self, output, target):
        """Return the error delta from the output layer. """

        return (output - target) / (output * (1 - output))
