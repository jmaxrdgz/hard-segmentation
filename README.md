# HARD segmentation

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

## Apple M3 results

Measured on this computer on **18 November 2026**: Apple M3, 8 GB RAM,
macOS 15.7.3, FP32 PyTorch CPU, two intra-op threads and one inter-op thread,
batch one, random weights, contiguous layout. Each variant and resolution ran
sequentially in a fresh process with 30 warmups and 1,000 timed forwards over
1,000 distinct pregenerated inputs. macOS scheduling was not pinned to cores.
Timing includes the full-resolution 19-class logit resize, excluding input
generation, preprocessing and argmax.

| Model | 360x640 mean (ms) | 360x640 FPS | 192x640 mean (ms) | 192x640 FPS |
| --- | ---: | ---: | ---: | ---: |
| HARD-XXS | 68.10 | **14.68** | 39.63 | **25.23** |
| HARD-XS | 141.93 | **7.05** | 72.49 | **13.79** |
| HARD-S | 211.77 | **4.72** | 115.85 | **8.63** |

These are observed timings on the current desktop. A summary-file write initially
failed with low disk space after all six benchmark processes had exited
successfully; it succeeded on retry. Post-run swap was in use. No per-frame
memory-pressure telemetry was captured, so the effect of swapping on latency
cannot be established. All raw JSON records were validated and preserved.

[M3 CSV summary](benchmarks/m3-2026-11-18-summary.csv) |
[M3 hardware, exact commands and post-run conditions](benchmarks/m3-2026-11-18-hardware.json).
The six `m3-2026-11-18-{variant}-{height}x640.json` files contain all timings,
configuration, source hashes and runtime details. Source revision: `d979d3b`.

## Raspberry Pi 5 results

XS and S passed the full eight-test suite on the Pi on 18 November 2026.
Their completed 360x640 measurements use the same protocol as below, with
source revision `d979d3b`, PyTorch 2.14.1+cpu and Python 3.13.5.

| Model | 360x640 mean (ms) | 360x640 FPS | 192x640 |
| --- | ---: | ---: | --- |
| HARD-XS | 762.35 | **1.31** | In progress |
| HARD-S | 1,045.35 | **0.96** | In progress |

**Active undervoltage/throttling was sampled during both new Pi runs**:
1 of 76 telemetry samples for XS and 10 of 103 for S reported active flags.
These results describe the current power conditions, not a verified stable-power
baseline. Sampling cannot establish the full duration or latency impact of events.

[XS 360x640 raw timings and telemetry](benchmarks/rpi5-2026-11-18-xs-360x640.json) |
[S 360x640 raw timings and telemetry](benchmarks/rpi5-2026-11-18-s-360x640.json).

### Earlier HARD-XXS measurements

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
