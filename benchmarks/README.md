# Benchmark records

Result filenames identify the host, date, variant and input size:
`{host}-{date}-{variant}-{height}x{width}.json`. Shared hardware metadata and
summary CSVs identify the host and date without a variant. The root README
links to the current result tables.

The original 15 November XXS files were renamed to match this scheme:

| Original filename | Current filename |
| --- | --- |
| `hard-xxs-rpi5-360x640.json` | `rpi5-2026-11-15-xxs-360x640.json` |
| `hard-xxs-rpi5-192x640.json` | `rpi5-2026-11-15-xxs-192x640.json` |
| `hard-xxs-rpi5-hardware.json` | `rpi5-2026-11-15-hardware.json` |
| `hard-xxs-rpi5-summary.csv` | `rpi5-2026-11-15-summary.csv` |

The contents of all archived result and hardware files are unchanged. Commands
inside those records intentionally retain the paths used when the measurements
were taken, including `hard_xxs/benchmark.py` and the former checkout folder.
Recorded source hashes refer to the original `model.py` and `benchmark.py`:
commit `bc28f8e` for 15 November, and `d979d3b` for 18 November.

The current code lives in `hard_segmentation/models.py` and
`hard_segmentation/benchmark.py`. New benchmark records hash those filenames.
Use the current README commands to run the renamed package, and the recorded
revision when checking historical source hashes. Renaming files does not create
new performance measurements.
