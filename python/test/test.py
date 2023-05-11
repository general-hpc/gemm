import unittest

import tensor


class Test(unittest.TestCase):

    def test(self):
        left = tensor.Matrix([1, 2, 3, 4, 5, 6], 2, 3)
        right = tensor.Matrix([1, 2, 3, 4, 5, 6], 3, 2)
        actual = tensor.Mul(left, right).eval()
        expect = tensor.Matrix([22, 28, 49, 64], 2, 2)
        self.assertEqual(actual, expect)
