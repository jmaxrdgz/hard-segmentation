# HARD segmentation

Standalone PyTorch reconstructions of HARD-XXS, HARD-XS and HARD-S semantic
segmentation. Unspecified architecture choices are
[documented explicitly](hard_segmentation/README.md). This is not author code or a
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
from hard_segmentation import HARDXXS, HARDXS, HARDS, build_model

model = build_model('xs', classes=19).eval()  # 'xxs', 'xs', or 's'
with torch.inference_mode():
    logits = model(torch.randn(1, 3, 360, 640))  # (1, 19, 360, 640)
```

`HARDXXS()`, `HARDXS()` and `HARDS()` construct the corresponding default
19-class models. `Config()` and the default benchmark still select XXS.
The shared implementation is `hard_segmentation/models.py`, with a generic
`HARD` base and the three variant classes. Tests live in
`hard_segmentation/test_models.py`. Imports now use `hard_segmentation`;
model state-dictionary keys and tensor shapes are unchanged.

## Apple M3 results

Measured on this computer on **18 November 2026**: Apple M3, 8 GB RAM,
macOS 15.7.3, Python 3.13.3, PyTorch 2.14.1, FP32 CPU,
two intra-op threads and one inter-op thread,
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

All three variants were measured on a Raspberry Pi 5 Model B Rev 1.1 with
8 GB RAM, PyTorch 2.14.1+cpu and Python 3.13.5. XXS was measured on
**15 November 2026** (source at `bc28f8e`); XS and S on **18 November 2026**
(source revision `d979d3b`). XS and S passed the full eight-test suite on the Pi.

All runs used two CPU cores (2-3), two intra-op threads, one inter-op thread,
FP32 eager, batch one, random weights, 30 warmups and 1,000 timed forwards over
1,000 distinct pregenerated inputs per resolution. Full-resolution 19-class
logits are included in timing. Runs were sequential in fresh processes, with
all native thread affinities verified as restricted to cores 2-3. The governor
was `ondemand`, with a configured maximum CPU frequency of 2.4 GHz.

| Model | 360x640 mean (ms) | 360x640 FPS | 192x640 mean (ms) | 192x640 FPS |
| --- | ---: | ---: | ---: | ---: |
| HARD-XXS | 326.84 | **3.06** | 171.26 | **5.84** |
| HARD-XS | 762.35 | **1.31** | 417.16 | **2.40** |
| HARD-S | 1,045.35 | **0.96** | 566.34 | **1.77** |

XXS had no active throttling flags observed, with sampled clocks about 2.4 GHz;
historical undervoltage/throttling flags were already set before its runs.
**Active undervoltage/throttling was sampled during all four XS/S runs**.
Recorded XS/S temperatures ranged from 34.5 to 54.9 C.

| Model | Input | Samples with active power flags / telemetry samples |
| --- | --- | ---: |
| HARD-XXS | 360x640 | 0 / 35 |
| HARD-XXS | 192x640 | 0 / 19 |
| HARD-XS | 360x640 | 1 / 76 |
| HARD-S | 360x640 | 10 / 103 |
| HARD-XS | 192x640 | 5 / 43 |
| HARD-S | 192x640 | 7 / 58 |

The rows reflect their observed power conditions. XS/S results are not a verified
stable-power baseline. Sampling cannot establish the full duration or latency impact
of events.

[XXS 360x640 raw timings and telemetry](benchmarks/rpi5-2026-11-15-xxs-360x640.json) |
[XXS 192x640](benchmarks/rpi5-2026-11-15-xxs-192x640.json) |
[XS 360x640](benchmarks/rpi5-2026-11-18-xs-360x640.json) |
[S 360x640](benchmarks/rpi5-2026-11-18-s-360x640.json) |
[XS 192x640](benchmarks/rpi5-2026-11-18-xs-192x640.json) |
[S 192x640](benchmarks/rpi5-2026-11-18-s-192x640.json) |
[XXS CSV summary](benchmarks/rpi5-2026-11-15-summary.csv) |
[XS/S CSV summary](benchmarks/rpi5-2026-11-18-summary.csv) |
[XXS hardware and exact commands](benchmarks/rpi5-2026-11-15-hardware.json) |
[XS/S hardware and exact commands](benchmarks/rpi5-2026-11-18-hardware.json).

These are latency measurements of the reconstructions, not trained segmentation
accuracy or camera throughput.

The **ten 18 November runs** (six M3 and four Pi) completed 1,000 timed forwards each.
Sample counts, mean/median/FPS calculations, source hashes, thread settings,
input/output shapes and Pi native thread affinities were verified before
publication. A [combined CSV](benchmarks/2026-11-18-summary.csv) includes timings
and the available power/temperature sample summaries. No M3 power or temperature
telemetry was collected; empty fields in its CSV rows indicate unavailable data.

[Architecture, tests and reproduction commands](hard_segmentation/README.md) |
[Raw results and telemetry](benchmarks/)

## Setup and test

Requires Python 3.10+ and PyTorch. The Pi runs used Python 3.13.5 and
PyTorch 2.14.1+cpu; the M3 runs used Python 3.13.3 and PyTorch 2.14.1.
From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch
.venv/bin/python -m unittest hard_segmentation.test_models -v
```

Run on the Pi:

```bash
taskset -c 2,3 .venv/bin/python hard_segmentation/benchmark.py \
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
Archived commands retain their original paths; see the
[benchmark filename and provenance notes](benchmarks/README.md) for the rename map.
