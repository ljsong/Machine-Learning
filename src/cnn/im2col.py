#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def get_im2col_indices(
        x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    W = x_shape[-1]
    H = x_shape[-2]

    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1

    # field_height * field_width rows
    i0 = np.repeat(np.arange(field_height), field_width)
    j0 = np.tile(np.arange(field_width), field_height)

    # out_height * out_width columns
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j1 = stride * np.tile(np.arange(out_width), out_height)

    # broadcasting
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    return (i, j)


def im2col(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((p, p), (p, p)), mode='constant')

    indices = get_im2col_indices(
        x.shape, field_height, field_width, padding, stride)

    cols = x_padded[indices]

    return cols


def main():
    a = np.eye(3, dtype=int)
    print im2col(a, 2, 2)


if __name__ == '__main__':
    main()
