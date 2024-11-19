import unittest
import math
import paddle

class TestConstants(unittest.TestCase):
    def test_pi(self):
        self.assertEqual(math.pi, paddle.pi)

if __name__ == "__main__":
    unittest.main()