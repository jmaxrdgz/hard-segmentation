import unittest

import torch
from torch import nn

from hard_segmentation.models import Config, HARD, HARDXXS, HARDXS, HARDS, PRESETS, build_model


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

    def test_all_variants_forward_backward_and_checkpoint(self):
        for variant, expected_class in [('xxs', HARDXXS), ('xs', HARDXS), ('s', HARDS)]:
            with self.subTest(variant=variant):
                model = build_model(variant, classes=11).train()
                self.assertIsInstance(model, HARD)
                self.assertIsInstance(model, expected_class)
                logits = model(torch.randn(2, 3, 65, 97))
                self.assertEqual(tuple(logits.shape), (2, 11, 65, 97))
                target = torch.randint(0, 11, (2, 65, 97))
                nn.functional.cross_entropy(logits, target).backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                restored = build_model(variant, classes=11).eval()
                restored.load_state_dict(model.state_dict(), strict=True)
                model.eval()
                image = torch.randn(1, 3, 64, 96)
                with torch.inference_mode():
                    torch.testing.assert_close(model(image), restored(image), rtol=0, atol=0)

    def test_all_variant_shapes_and_architecture(self):
        counts = []
        for variant in PRESETS:
            with self.subTest(variant=variant):
                model = build_model(variant).eval()
                counts.append(sum(p.numel() for p in model.parameters()))
                self.assertEqual(model.configuration()['widths'], PRESETS[variant].widths)
                for block in model.context.blocks:
                    depthwise = [m for m in block.modules() if isinstance(m, nn.Conv2d)
                                 and m.groups == m.in_channels and m.kernel_size == (3, 3)]
                    self.assertEqual([m.dilation for m in depthwise], [(18, 18), (12, 12), (6, 6)])
                with torch.inference_mode():
                    logits = model(torch.randn(1, 3, 360, 640))
                self.assertEqual(tuple(logits.shape), (1, 19, 360, 640))
                self.assertTrue(torch.isfinite(logits).all())
        self.assertLess(counts[0], counts[1])
        self.assertLess(counts[1], counts[2])

    def test_default_classes_and_invalid_variant(self):
        self.assertEqual(HARDXS().config, PRESETS['xs'])
        self.assertEqual(HARDS().config, PRESETS['s'])
        self.assertIsInstance(build_model('XS'), HARDXS)
        with self.assertRaises(ValueError):
            build_model('unknown')

    def test_xxs_configuration_and_checkpoint_compatibility(self):
        legacy = HARDXXS()
        current = build_model()
        self.assertEqual(legacy.configuration(), current.configuration())
        self.assertEqual(sum(p.numel() for p in current.parameters()), 109571)
        current.load_state_dict(legacy.state_dict(), strict=True)


if __name__ == '__main__':
    unittest.main()
