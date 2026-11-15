import unittest

import torch
from torch import nn

from hard_xxs.model import Config, HARDXXS


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_spatial_shapes_and_backward(self):
        model = HARDXXS(Config(classes=11)).train()
        image = torch.randn(2, 3, 65, 97, requires_grad=True)
        logits = model(image)
        self.assertEqual(tuple(logits.shape), (2, 11, 65, 97))
        target = torch.randint(0, 11, (2, 65, 97))
        nn.functional.cross_entropy(logits, target).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_serial_dilated_depthwise_convolutions(self):
        model = HARDXXS()
        blocks = model.context.blocks
        self.assertEqual(len(blocks), 3)
        for block in blocks:
            depthwise = [m for m in block.modules() if isinstance(m, nn.Conv2d)
                         and m.groups == m.in_channels and m.kernel_size == (3, 3)]
            self.assertEqual([m.dilation for m in depthwise], [(18, 18), (12, 12), (6, 6)])
            self.assertTrue(block.residual)

    def test_full_cityscapes_resolution(self):
        model = HARDXXS().eval()
        with torch.inference_mode():
            logits = model(torch.randn(1, 3, 360, 640))
        self.assertEqual(tuple(logits.shape), (1, 19, 360, 640))
        self.assertTrue(torch.isfinite(logits).all())

    def test_state_dict_round_trip(self):
        a, b = HARDXXS().eval(), HARDXXS().eval()
        b.load_state_dict(a.state_dict(), strict=True)
        image = torch.randn(1, 3, 64, 96)
        with torch.inference_mode():
            torch.testing.assert_close(a(image), b(image), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
