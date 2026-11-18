# HARD-XXS, HARD-XS and HARD-S reconstructions

Standalone PyTorch semantic segmentation implementations, with **109,571 / 384,499 /
751,467 trainable parameters** for XXS / XS / S with 19 classes. These are paper
reconstructions with inferred layer configurations, not official author code,
trained models, or a verified
reproduction of the published accuracy or speed. A similar parameter count does
not establish architectural equivalence.

Source: Kwon, Kim and Kim, ACCV 2024,
[HARD paper](https://openaccess.thecvf.com/content/ACCV2024/papers/Kwon_HARD__Hardware-Aware_lightweight_Real-time_semantic_segmentation_model_Deployable_from_ACCV_2024_paper.pdf),
sections 3.2-3.3 and Figures 2 and 5;
[supplement](https://openaccess.thecvf.com/content/ACCV2024/supplemental/Kwon_HARD__Hardware-Aware_ACCV_2024_supplemental.pdf).
These sources leave the embedded variants' channel widths, repeats, expansion and detailed
normalization/activation configuration unspecified. The earlier repository review
located no author code or checkpoint. This implementation fills those gaps
explicitly; changing them can change CPU latency substantially.

## Implemented architecture

The table below details XXS. XS and S use the same operations and skip routing,
with these explicitly inferred presets:

| Variant | Widths at 1/2, 1/4, 1/8 | Repeats | Expansion | Classifier width | Parameters (19 classes) |
| --- | --- | --- | ---: | ---: | ---: |
| XXS | 16, 24, 40 | 1, 2, 3 | 4 | 32 | 109,571 |
| XS | 32, 48, 80 | 1, 2, 3 | 4 | 64 | 384,499 |
| S | 40, 64, 116 | 1, 2, 3 | 4 | 96 | 751,467 |

These widths and classifier sizes approximate the nominal model sizes in the
paper. Parameter proximity does not verify architectural equivalence. Repeats,
expansion, normalization, activations, skip routing and dilation interpretation
remain the same assumptions as the original XXS reconstruction below.

| Component | Configuration | Output scale |
| --- | --- | --- |
| Stem | 3x3 conv, 3 to 16, stride 2 | 1/2 |
| Encoder 1 | 1 inverted bottleneck, 16 channels | 1/2 |
| Encoder 2 | 2 inverted bottlenecks, 24 channels; first stride 2 | 1/4 |
| Encoder 3 | 3 inverted bottlenecks, 40 channels; first stride 2 | 1/8 |
| SA | 3 residual AIVBs, 40 channels; each has serial depthwise dilations 18, 12, 6 | 1/8 |
| Context concat | Encoder 3 output and final SA output, 80 channels | 1/8 |
| Detail fusion | Bilinear context resize; concatenate Encoder 2, 104 channels | 1/4 |
| Classifier | 1x1 conv to 32; 1x1 conv to class count | 1/4 |
| Output | Bilinear logits resize to exact input size | 1/1 |

Paper-supported structure: inverted-bottleneck encoder down to 1/8, three AIVBs
with three serial dilated depthwise convolutions apiece (nine total), residual
connections shown in Figure 2, context concatenation, classifier and final resize.
Figure 5 also depicts a 1/4 decoder and encoder skip connection.

Engineering assumptions: all channel widths and encoder repeats above; expansion
factor 4 everywhere; 3x3 stride-two stem; downsampling in the first depthwise
convolution of each encoder stage; BatchNorm and ReLU6 after expansion and each
depthwise convolution; linear Conv-BN projection; residuals only at matching
shapes; exact skip routing to the 1/4 classifier; concatenation of the input and
last SA output rather than all intermediate AIVB outputs; 32-channel pointwise
classifier; bilinear resize with `align_corners=False`. All three AIVBs use the
same decreasing dilation triplet, following Figure 2. The text alone is ambiguous
about decreasing rates across modules versus inside each module.

`Config` exposes widths, repeats, expansion, head width, classes and dilations.
`build_model(variant, classes=19)` selects the `xxs`, `xs` or `s` preset; the public
`HARDXXS`, `HARDXS` and `HARDS` classes also select their corresponding defaults.
Supply `config_for_variant('s', classes=11)` to obtain a modified preset.
No DAP module or GPU auxiliary heads are included in this embedded variant.
The default returns `(N, 19, H, W)` logits, including for odd input dimensions.
Training with cross-entropy and saving/loading a normal `state_dict` are supported.
There is no training pipeline, dataset or pretrained checkpoint in this addition.

## Validation

From the repository root, using the existing PyTorch environment:

```bash
.venv/bin/python -m unittest hard_xxs.test_model -v
```

Four tests passed locally on 15 November 2026: full 360x640 output and finite
logits, odd-size output and cross-entropy backward through every parameter,
all nine depthwise dilation settings, and exact state-dictionary round-trip.
A five-iteration local runner smoke check also passed. Its timing is not a Pi
measurement and is deliberately not published as a performance result.
The same four model tests subsequently passed on the Raspberry Pi 5.

On 18 November 2026, the expanded **eight-test suite passed locally** for all
three variants. It checks full 360x640 finite logits, odd-size forward and
cross-entropy gradients for every parameter, per-variant checkpoint round-trips,
the nine depthwise dilation settings, preset selection and XXS compatibility.
XS and S also passed short benchmark-runner smoke checks, using the script and
module entry points respectively. The full eight-test suite subsequently passed
on the Pi on 18 November 2026. All three variants completed M3 benchmarks and
XS/S completed Pi benchmarks at both 360x640 and 192x640, with 1,000 timed
forwards per configuration; see the [results tables](../README.md).

The 18 November runs use two intra-op threads and one inter-op thread, FP32
eager CPU, batch one, random weights, 30 warmups and 1,000 distinct pregenerated
inputs. Pi threads are pinned to cores 2-3; M3 threads use macOS scheduling.
All four new Pi runs captured active undervoltage/throttling flags, and the M3
had low disk space and swap in use after its runs. These observed conditions
are documented with the results. The [combined CSV](../benchmarks/2026-11-18-summary.csv)
and per-run JSON files preserve the measurements and provenance.

## Earlier XXS two-core Raspberry Pi 5 benchmark

Measured on **15 November 2026** on a **Raspberry Pi 5 Model B Rev 1.1, 8 GB**,
over Tailscale SSH. Debian 13.7 ARM64, Linux 6.18.50+rpt-rpi-2712,
Python 3.13.5, PyTorch 2.14.1+cpu. Both runs used cores **2-3** only,
two intra-op threads, one inter-op thread, batch one, FP32 eager CPU, random
weights, contiguous memory format, 30 warmups and **1,000 timed forwards over
1,000 distinct pregenerated inputs**. Runs were sequential, each in a fresh
process. Original benchmark workspace revision: `c894c5a8d64d34a1cd83a5396a2bc42e2b9a3719`.
The code and results were subsequently moved to the standalone `jmaxrdgz/hard-segmentation`
repository. The recorded hashes verify the same model and benchmark source bytes;
the archived commands and revision describe the original run before migration.
Those original files remain available at commit `bc28f8e`. The later XS/S
extension changes the shared source files but keeps the XXS layer configuration
and state-dictionary layout compatible. Original XXS result files are retained
without edits.

| Input (H x W) | Parameters | Mean latency (ms) | FPS |
| --- | ---: | ---: | ---: |
| 360 x 640 | 109,571 | 326.84 | **3.06** |
| 192 x 640 | 109,571 | 171.26 | **5.84** |

Both runs verified the affinity of every native thread as restricted to cores
2-3. All telemetry samples reported about 2.4 GHz ARM clock and unchanged
`get_throttled=0x50000`: historical undervoltage and throttling flags, with no
active flags observed. Sampled temperatures were 44.4-54.9 C at 360x640 and
50.5-56.0 C at 192x640. The governor was `ondemand`. Sampling does not rule out
brief events between measurements; the historical flags predate these runs.

Records: [360x640 timings and telemetry](../benchmarks/hard-xxs-rpi5-360x640.json),
[192x640 timings and telemetry](../benchmarks/hard-xxs-rpi5-192x640.json),
[CSV summary](../benchmarks/hard-xxs-rpi5-summary.csv), and
[hardware and exact commands](../benchmarks/hard-xxs-rpi5-hardware.json).
Each JSON contains all 1,000 frame timings and source hashes. Sample counts,
source hashes, mean/FPS calculations, thread settings and all recorded thread
affinities were checked before publication.

These are measurements of this reconstruction with the assumptions above, not
validation of the paper's exact HARD-XXS or its trained accuracy.

The runner depends only on PyTorch and the Python standard library. In a Pi
checkout of this repository with a PyTorch environment at `.venv`:

```bash
taskset -c 2,3 .venv/bin/python hard_xxs/benchmark.py \
  --cpus 2,3 --threads 2 --height 360 --width 640 --classes 19 \
  --warmup 30 --iterations 1000 --noise-images 1000 \
  --output benchmarks/hard-xxs-rpi5-360x640.json
```

For comparison with the existing depth CPU timings, run a fresh process at
192x640 as well:

```bash
taskset -c 2,3 .venv/bin/python hard_xxs/benchmark.py \
  --cpus 2,3 --threads 2 --height 192 --width 640 --classes 19 \
  --warmup 30 --iterations 1000 --noise-images 1000 \
  --output benchmarks/hard-xxs-rpi5-192x640.json
```

These commands pin the entire process before PyTorch imports and verify native
thread affinity. Both use two intra-op threads, one inter-op thread, batch one,
FP32 eager CPU, contiguous layout, random weights, seed zero, 30 warmups and 1,000
timed forwards over 1,000 distinct pregenerated uniform-noise images. At 360x640,
the inputs occupy about 2.76 GB (decimal) RAM; reduce `--noise-images` if necessary,
and retain its value in reported results. The classifier produces 19 segmentation
channels rather than the single depth channel of the separate depth benchmark repo.

To measure the new variants with the same protocol, add `--variant xs` or
`--variant s`, and use a separate output filename for each variant and resolution.

Each result JSON includes all timings, mean/median/p90/p99 latency, FPS calculated
as `1000 / mean_ms`, model configuration, source SHA256 hashes, PyTorch build,
thread counts and affinities, input/output shapes, and Pi temperature, ARM clock
and throttling telemetry before/after and approximately every ten seconds between
forwards. Telemetry is outside the timed intervals. Inspect `get_throttled` for
active and historical flags; historical flags alone do not date an event to this
run. Sampling cannot exclude short power or thermal events between samples.

Timing includes the full-resolution logit interpolation. It excludes input
generation, preprocessing, camera/network I/O and argmax. Conv-BN fusion,
quantization and alternate runtimes are not used. Random weights measure operator
latency only, not segmentation accuracy or useful end-to-end camera FPS.
