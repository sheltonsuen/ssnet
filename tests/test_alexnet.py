import unittest

import torch

from ssnet.alexnet import AlexNet


class TestAlexNet(unittest.TestCase):
    def setUp(self):
        self.alexnet = AlexNet()

    def test_forward_should_return_a_tensor_as_well(self):
        x = self.alexnet(torch.rand((3, 256, 256)))
        self.assertTrue(isinstance(x, torch.Tensor))


if __name__ == '__main__':
    unittest.main()
