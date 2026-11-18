# HARD embedded segmentation models

Standalone PyTorch reconstructions of HARD-XXS, HARD-XS and HARD-S semantic
segmentation. Unspecified architecture choices are
[documented explicitly](hard_xxs/README.md). This is not author code or a
verified reproduction of the paper's accuracy. No trained weights are included.

| Variant | Encoder widths | Encoder repeats | Expansion | Classifier width | Parameters (19 classes) |
| --- | --- | --- | ---: | ---: | ---: |
| HARD-XXS | 16, 24, 40 | 1, 2, 3 | 4 | 32 | 109,571 |
| HARD-XS | 32, 48, 80 | 1, 2, 3 | 4 | 64 | 384,499 |
| HARD-S | 40, 64, 116 | 1, 2, 3 | 4 | 96 | 751,467 |

All variants use the shared stride-eight inverted-bottleneck encoder,
three-block serialized atrous module, detail fusion and stride-four classifier.
The XS/S widths and classifier sizes are engineering presets chosen near the
paper's nominal parameter budgets, not recovered author configurations.

```python
import torch
from hard_xxs import HARDXXS, HARDXS, HARDS, build_model

model = build_model('xs', classes=19).eval()  # 'xxs', 'xs', or 's'
with torch.inference_mode():
    logits = model(torch.randn(1, 3, 360, 640))  # (1, 19, 360, 640)
```

`HARDXXS()`, `HARDXS()` and `HARDS()` construct the corresponding default
19-class models. `Config()` and the default benchmark still select XXS.
The `hard_xxs` package name is retained for existing imports and XXS checkpoints.

## Raspberry Pi 5 results

The results below are for **HARD-XXS**. XS and S have passed local forward,
training-gradient, checkpoint and benchmark-runner checks. Their Pi FPS is
**not yet measured**: the configured Pi SSH endpoint timed out on 18 November
2026. No local smoke timing is presented as Pi performance.

Two CPU cores (2-3), two intra-op threads, one inter-op thread, FP32 eager,
batch one, random weights, 30 warmups and 1,000 timed forwards over 1,000
distinct pregenerated inputs per resolution. Full-resolution 19-class logits
are included in timing.

| Input (H x W) | Mean latency (ms) | FPS |
| --- | ---: | ---: |
| 360 x 640 | 326.84 | **3.06** |
| 192 x 640 | 171.26 | **5.84** |

Measured 15 November 2026. All sampled clocks were about 2.4 GHz, with no
active throttling flags observed. Historical undervoltage/throttling flags
were already set before the runs. These are latency measurements of this
reconstruction, not trained segmentation accuracy or camera throughput.

[Architecture, tests and reproduction commands](hard_xxs/README.md) |
[Raw results and telemetry](benchmarks/) |
[CSV summary](benchmarks/hard-xxs-rpi5-summary.csv)

## Setup and test

Requires Python 3.10+ and PyTorch. The published runs used Python 3.13.5
and PyTorch 2.14.1+cpu. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch
.venv/bin/python -m unittest hard_xxs.test_model -v
```

Run on the Pi:

```bash
taskset -c 2,3 .venv/bin/python hard_xxs/benchmark.py \
  --variant xs --cpus 2,3 --threads 2 --height 360 --width 640 --classes 19 \
  --warmup 30 --iterations 1000 --noise-images 1000 \
  --output benchmarks/new-xs-rpi5-360x640.json
```

At 360x640, the pregenerated inputs occupy about 2.76 GB RAM.
Use `--variant s` for HARD-S and `--height 192` for the smaller benchmark size.
Each JSON records the selected variant, full configuration, parameter count,
source hashes, individual timings and telemetry.
The original run commands and revision in the archived metadata predate
this repository migration. The archived XXS records are unchanged; their source
hashes refer to the files at commit `bc28f8e`, before the XS/S extension.
