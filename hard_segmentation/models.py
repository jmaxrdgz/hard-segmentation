"""Embedded HARD paper reconstructions; inferred configurations, not author code.

See README.md for every architectural assumption. Returns full-resolution logits.
"""
from dataclasses import asdict, dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    classes: int = 19
    widths: tuple[int, int, int] = (16, 24, 40)
    repeats: tuple[int, int, int] = (1, 2, 3)
    expansion: int = 4
    head_width: int = 32
    dilations: tuple[int, int, int] = (18, 12, 6)


# Engineering presets, not configurations recovered from author code.
PRESETS = {
    'xxs': Config(),
    'xs': Config(widths=(32, 48, 80), head_width=64),
    's': Config(widths=(40, 64, 116), head_width=96),
}


def config_for_variant(variant, classes=19):
    try:
        return replace(PRESETS[variant.lower()], classes=classes)
    except KeyError:
        raise ValueError(f'unknown HARD variant: {variant!r}; choose {tuple(PRESETS)}') from None


def conv_bn_relu(cin, cout, kernel=1, stride=1, dilation=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel, stride, dilation * (kernel // 2),
                  dilation=dilation, groups=groups, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU6(inplace=True))


class InvertedBottleneck(nn.Module):
    def __init__(self, cin, cout, stride=1, expansion=4, dilations=(1,)):
        super().__init__()
        hidden = cin * expansion
        self.residual = stride == 1 and cin == cout
        layers = [conv_bn_relu(cin, hidden)]
        for index, dilation in enumerate(dilations):
            layers.append(conv_bn_relu(hidden, hidden, 3,
                                       stride if index == 0 else 1,
                                       dilation, hidden))
        layers.extend([nn.Conv2d(hidden, cout, 1, bias=False), nn.BatchNorm2d(cout)])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)
        return x + y if self.residual else y


class SerializedAtrous(nn.Module):
    def __init__(self, width, expansion, dilations):
        super().__init__()
        self.blocks = nn.Sequential(*[
            InvertedBottleneck(width, width, expansion=expansion, dilations=dilations)
            for _ in range(3)])

    def forward(self, x):
        return torch.cat((x, self.blocks(x)), dim=1)


class HARD(nn.Module):
    """Shared embedded HARD architecture with a stride-four segmentation head."""
    def __init__(self, config=Config()):
        super().__init__()
        if (config.classes < 1 or config.expansion < 1 or config.head_width < 1
                or len(config.widths) != 3 or len(config.repeats) != 3
                or len(config.dilations) != 3
                or min(*config.widths, *config.repeats, *config.dilations) < 1):
            raise ValueError('invalid HARD configuration')
        self.config = config
        a, b, c = config.widths
        self.stem = conv_bn_relu(3, a, 3, 2)
        def stage(cin, cout, repeats, stride):
            return nn.Sequential(
                InvertedBottleneck(cin, cout, stride, config.expansion),
                *[InvertedBottleneck(cout, cout, expansion=config.expansion)
                  for _ in range(repeats - 1)])
        self.stage1 = stage(a, a, config.repeats[0], 1)
        self.stage2 = stage(a, b, config.repeats[1], 2)
        self.stage3 = stage(b, c, config.repeats[2], 2)
        self.context = SerializedAtrous(c, config.expansion, config.dilations)
        self.classifier = nn.Sequential(
            conv_bn_relu(2 * c + b, config.head_width),
            nn.Conv2d(config.head_width, config.classes, 1))

    def forward(self, image):
        detail = self.stage2(self.stage1(self.stem(image)))
        context = self.context(self.stage3(detail))
        context = F.interpolate(context, size=detail.shape[-2:], mode='bilinear',
                                align_corners=False)
        logits = self.classifier(torch.cat((detail, context), dim=1))
        return F.interpolate(logits, size=image.shape[-2:], mode='bilinear',
                             align_corners=False)

    def configuration(self):
        return asdict(self.config)


class HARDXXS(HARD):
    """HARD-XXS reconstruction with the inferred XXS preset by default."""
    def __init__(self, config=None):
        super().__init__(config_for_variant('xxs') if config is None else config)


class HARDXS(HARD):
    """HARD-XS reconstruction with the inferred XS preset by default."""
    def __init__(self, config=None):
        super().__init__(config_for_variant('xs') if config is None else config)


class HARDS(HARD):
    """HARD-S reconstruction with the inferred S preset by default."""
    def __init__(self, config=None):
        super().__init__(config_for_variant('s') if config is None else config)


def build_model(variant='xxs', classes=19):
    config = config_for_variant(variant, classes)
    model_class = {'xxs': HARDXXS, 'xs': HARDXS, 's': HARDS}[variant.lower()]
    return model_class(config)
