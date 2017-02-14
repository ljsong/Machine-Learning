#! /usr/bin/env python
# -*- coding: utf-8 -*-

import path_magic
from numpy import rint
from numpy import all


def train(network, inputs, target, batch_size=1, epoch=3000):
    number, channel, row, col = inputs.shape

    times = 1
    while times <= epoch:
        total_error = 0
        idx = 0
        while idx < row:
            start = idx
            end = idx + batch_size if idx + batch_size < number else number
            counts = end - start

            each_input = inputs[start: end, :, :, :]
            each_target = target[start: end, :].T

            error = network.single_loop(each_input, each_target)
            total_error += error
            idx += counts

        # if times % 100 == 0:
        print "Epoch %d" % times
        print "Total Error: %3.4f" % total_error
        times += 1
    else:
        print "Epoch %d" % (times - 1)
        print "Total Error: %3.4f" % total_error


def validate(network, inputs, target):
    number, channel, row, col = inputs.shape
    output_neurons = target.shape[1]
    correct = 0

    idx = 0
    while idx < row:
        each_input = inputs[idx, :, :, :]
        each_target = target[idx, :].reshape(output_neurons, 1)

        outputs = network.feed_forward(each_input)
        outputs = rint(outputs)
        if all(outputs == each_target):
            correct += 1
        # else:
        #     print outputs, each_target

        idx += 1

    print "Correct Percentage: %3.2f%%" % (correct * 100.0 / row)
