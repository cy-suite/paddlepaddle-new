import unittest

import numpy as np

import paddle

class TestNewaxis(unittest.TestCase):
    def test_none_index(self):
        # `None` index adds newaxis
        a = np.array([1, 2, 3])
        np.testing.assert_equal(a[None], a[paddle.newaxis])
        np.testing.assert_equal(a[None].ndim, a.ndim + 1)
