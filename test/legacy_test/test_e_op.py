import unittest
import math
import paddle

paddle.enable_static()

class TestPaddleE(unittest.TestCase):
    def setUp(self):
        # Verifying paddle.e against math.e
        self.expected_value = math.e
        self.e_value = paddle.e

    def test_check_value(self):
        self.assertAlmostEqual(self.e_value, self.expected_value, places=6, msg="paddle.e does not match math.e")

    def test_check_type(self):
        self.assertIsInstance(self.e_value, float, msg="paddle.e is not of type float")

    def test_check_value_with_precision(self):
        self.assertTrue(abs(self.e_value - self.expected_value) < 1e-6, msg="paddle.e value mismatch with math.e")

if __name__ == "__main__":
    unittest.main()
