#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def get_im2col_indices(
        x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    # number means the counts of image, channel usually is 3(RGB)
    number, channel, height, width = x_shape

    assert (height + 2 * padding - field_height) % stride == 0
    assert (width + 2 * padding - field_height) % stride == 0
    out_height = int((height + 2 * padding - field_height) / stride + 1)
    out_width = int((width + 2 * padding - field_width) / stride + 1)

    # field_height * field_width rows
    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, channel)
    j0 = np.tile(np.arange(field_width), field_height * channel)

    # out_height * out_width columns
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j1 = stride * np.tile(np.arange(out_width), out_height)

    # broadcasting
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(channel), field_height * field_width).reshape(-1, 1)

    return i.astype(int), j.astype(int), k.astype(int)


def im2col(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding

    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    i, j, k = get_im2col_indices(
        x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    # channel(or depth) means the counts of color, usually it is 3(RGB)
    channel = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * channel, -1)

    return cols


def col2im(cols, x_shape, field_height, field_width, padding=1, stride=1):
    number, channel, height, width = x_shape
    padded_h = height + 2 * padding
    padded_w = width + 2 * padding

    x_padded = np.zeros((number, channel, padded_h, padded_w), dtype=cols.dtype)
    i, j, k = get_im2col_indices(x_shape, field_height, field_width, padding, stride)

    cols_reshaped = cols.reshape(channel * field_height * field_width, -1, number)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.fmax.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    if padding == 0:
        return x_padded

    return x_padded[:, :, padding:-padding, padding:-padding]


def main():
    a = np.arange(8).reshape((1, 1, 4, 2))
    cols = im2col(a, 2, 2, padding=0)
    print cols
    b = col2im(cols, a.shape, 2, 2, padding=0)

    print b


if __name__ == '__main__':
    main()
