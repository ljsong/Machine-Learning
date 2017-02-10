#! /usr/bin/env python
# -*- coding: utf-8 -*-

from cPickle import dump, load
from cPickle import HIGHEST_PROTOCOL


class ConvolutionalNet(object):

    def __init__(self, kernel_size, padding=1, stride=1):
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

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
