import numpy as np


def cast(a, b):

    # TODO: allow for b to be just a tuple
    if len(a.shape) == 1:
        if a.shape[0] == b.shape[0]:
            a = a.repeat(b.shape[1]).reshape(b.shape)
        elif a.shape[0] == b.shape[1]:
            a = np.tile(a, b.shape[0]).reshape(b.shape)
        else:
            raise ValueError('operands could not be broadcast together with shapes {} {}'
                             .format(a.shape, b.shape))
        return a
    elif len(a.shape) == 2:
        if a.shape[0] == b.shape[0]:
            a = a.repeat(b.shape[1]).reshape(b.shape)
        elif a.shape[0] == b.shape[1]:
            a.repeat(b.shape[0]).reshape(b.shape).T
        return a


def add_to_rows(x, add):

    """
    Add numbers to each row in a vectorized fashion. Very useful for arrays with many many rows.
    Broken
    :param x:
    :param add:
    :return:
    """

    rows = x.shape[0]
    cols = x.shape[1]
    add = np.array(add).repeat(cols).reshape(rows, -1)

    return x + add