import unittest

import numpy as np

import paddle

class TestInfRepr(unittest.TestCase):
    def test_inf(self):
        x = np.array([paddle.inf])
        np.testing.assert_equal(repr(x), 'array([inf])')
