# Post-Processing Pipeline — MR-OOD Anomaly Detection

## Project Overview

This repository is one part of a three-stage pipeline for MR-only radiotherapy anomaly detection:

**1. Anomaly extraction** (`ood-train/`)
Flow-based models (CFlow, FastFlow) are trained on normal MR pelvis slices.
At inference, the model produces per-slice **anomaly maps** (continuous scores, `.npy`)
and binary **prediction masks** (PNG) for the test set.
See `../ood-train/README.md` for training and extraction commands.

**2. Post-processing + evaluation** ← this repository
Raw prediction masks are refined through a multi-stage post-processing pipeline:
body masking → small-component filtering → morphological closing → consecutive-slice
(3D persistence) filtering. Pixel-, slice-, and patient-level metrics are computed
on both raw and post-processed masks and saved as a JSON summary.

**3. Visualization** (scripts included here)
Scripts to reproduce qualitative figures from the report and presentation
are located in the root of this repository (see [Visualization](#visualization)).
Additional figure notebooks live in `../mp_visualizations/`.

---

## Relationship to Other Repositories

| Repository | Role |
|---|---|
| `../ood-train` | Model training (CFlow, FastFlow) and anomaly map extraction |
| `../OOD-Data-Preprocessing` | Dataset preparation (NIfTI→PNG conversion, split creation, body mask generation) |
| `../mp_visualizations` | Additional analysis notebooks and presentation figures |
| **this repo** | Post-processing, evaluation, and visualization of model outputs |

---

## Expected Inputs

This repository takes as input the outputs of `ood-train/extract_fastflow.py` (or the CFlow equivalent) — collectively referred to as the model extraction output:

```
<extraction_output_root>/
  anomaly_maps/
    test/
      good/img/          ← per-slice anomaly maps as .npy (float32 scores)
      Ungood/img/        ← per-slice anomaly maps as .npy
  prediction_masks/
    test/
      good/img/          ← binary prediction masks as PNG (0 or 255)
      Ungood/img/
```

Body masks (from `OOD-Data-Preprocessing`) must be available at a separate dataset root:

```
<dataset_root>/
  test/
    good/
      img/               ← source MR slices (PNG or NIfTI)
      bodymask/          ← binary body masks (PNG), same filenames as img/
    Ungood/
      img/
      bodymask/
      label/             ← ground-truth anomaly masks (PNG), required for metrics
```

---

## Output Artifacts

After running the pipeline, the output root contains:

```
<output_root>/
  01_body_masked_png/    ← prediction masks after body-mask application
  02_morphology_png/     ← after small-component filter + morphological closing
  03_consecutive_filtered_png/  ← final post-processed masks (3D persistence filter)
  volumes/
    raw/                 ← 3D NIfTI volumes from raw prediction masks
    post_processed/      ← 3D NIfTI volumes from post-processed masks
    ground_truth/        ← 3D NIfTI volumes from ground-truth masks (if provided)
  metrics/
    metrics_summary.json ← pixel/slice/patient metrics for both raw and post-processed
```

---

## Post-Processing Stages

The pipeline applies five sequential stages to the raw binary prediction masks:

| Stage | Script / Module | Description |
|---|---|---|
| 0 | `apply_bodymask.py` | Multiply prediction masks element-wise by the anatomical body mask to remove out-of-body detections |
| 1 | `morphology/processor.py` | Binarize (threshold=0.5) and remove connected components smaller than **τ_area = 3 pixels** |
| 2 | `morphology/processor.py` | Morphological closing: dilation × N followed by erosion × N (default kernel: 5×5 ellipse, 1 round). Fills small intra-region gaps and smooths contours |
| 3 | `morphology/stack_to_3d.py` | Stack 2D PNG slices into patient-wise 3D NIfTI volumes |
| 4 | `filter_prediction_masks_consecutive.py` | 3D persistence filter: discard any 2D connected component that does not overlap with an anomaly region in at least one neighbouring slice |

Default morphology parameters (configurable via CLI or `config/morpho_val.yaml`):

```
--min-component-size 3      # τ_area: minimum CC size in pixels
--kernel-size 5             # structuring element size (must be odd)
--kernel-shape ellipse      # or rect
--dilate-iterations 1       # dilation passes per round
--erode-iterations 1        # erosion passes per round
--num-rounds 1              # number of (dilate+erode) rounds
```

---

## Evaluation

### Before vs. After Post-Processing

`evaluate_model_outputs.py` computes metrics at three levels:

- **Pixel level**: precision, recall, Dice score, false negative rate, balanced accuracy
  (aggregated over all prediction–ground-truth pixel pairs)
- **Slice level**: each 2D slice is classified as positive/negative based on whether
  any predicted or ground-truth anomaly pixel exists; standard binary classification
  metrics are reported
- **Patient level**: mean positive fraction (α_mean) — the average fraction of predicted
  positive pixels per patient across all their slices. Patients are classified as
  anomalous if α_mean ≥ threshold. Metrics are reported for multiple thresholds
  (default: α_mean ∈ {0.0, 0.02, 0.05, 0.1})

`main_pipeline.py` automatically runs evaluation on both the raw input masks and the
**stage 02 output** (`02_morphology_png`, i.e. after morphological closing), then saves
both to `metrics/metrics_summary.json`. Stage 03 (`03_consecutive_filtered_png`) is
the final mask used for 3D NIfTI export and qualitative inspection; metrics are not
re-evaluated on it (this matches the pipeline design in the report).

### Standalone Metrics

```bash
python evaluate_model_outputs.py \
  --prediction-dir post_process_outputs/02_morphology_png \
  --ground-truth-dir /path/to/dataset/test \
  --ground-truth-replace img:label \
  --mean-fraction-thresholds 0.0 0.02 0.05 0.1 \
  --output-json metrics.json
```

---

## How to Run

### Full Pipeline (body mask → morphology → 3D filter → metrics)

```bash
python main_pipeline.py \
  --input-dir /path/to/prediction_masks/test \
  --body-mask-dir /path/to/dataset/root \
  --output-root post_process_outputs \
  --path-replace prediction_masks:test \
  --path-replace img:bodymask \
  --ground-truth-dir /path/to/dataset/root/test \
  --metrics-mean-fraction-thresholds 0.0 0.02 0.05 0.1 \
  --skip-missing-body-mask
```

Stage-by-stage PNG outputs land under `post_process_outputs/0*/`, NIfTI volumes
under `post_process_outputs/volumes/`, and the full metrics summary at
`post_process_outputs/metrics/metrics_summary.json`.

### Morphology Parameter Tuning (validation set)

Edit `config/morpho_val.yaml` to set the path to your body-masked validation masks
and the parameter combinations to test, then run:

```bash
python morphology/tune_morpho.py
```

Results are saved to `reports/morphology_tuning/tuning_report.json`.

---

## Visualization

Visualization scripts live in `visualization/`. They can be run directly from that folder
(they add the repository root to the Python path automatically).

| Script | Output |
|---|---|
| `visualization/visualize_processed_anomaly_maps.py` | Side-by-side comparison of original vs. body-masked anomaly maps |
| `visualization/visualize_processed_prediction_masks.py` | Raw, body-masked, and filtered prediction mask panels |
| `visualization/visualize_anomaly_thresholded_outputs.py` | Anomaly map next to its thresholded binary output |
| `visualization/convert_to_bone_colormap.py` | Convert NIfTI slices to bone-colormap PNGs for inspection |

Example — visualize before/after body masking:

```bash
python visualization/visualize_processed_anomaly_maps.py \
  --anomaly-dir /path/to/anomaly_maps/test \
  --masked-dir post_process_outputs/01_body_masked_png \
  --image-dir /path/to/dataset \
  --image-replace anomaly_maps:test \
  --comparison-dir comparisons \
  --overlay-dir overlays \
  --comparison-cmap magma \
  --overlay-alpha 0.6
```

Example — visualize raw vs. filtered prediction masks:

```bash
python visualization/visualize_processed_prediction_masks.py \
  --raw-dir /path/to/prediction_masks/test \
  --masked-dir post_process_outputs/01_body_masked_png \
  --image-dir /path/to/dataset \
  --image-replace prediction_masks:test \
  --output-dir prediction_mask_comparisons
```

---

## Repository Structure

```
Post-Processing-Pipeline/
│
├── main_pipeline.py                        # End-to-end pipeline entrypoint
├── apply_bodymask.py                       # Stage 0: body mask application
├── postprocess_utils.py                    # Shared I/O and array utilities
├── filter_prediction_masks_consecutive.py  # Stage 4: 3D persistence filter
├── evaluate_model_outputs.py               # Pixel/slice/patient metrics
├── compute_pixel_metrics.py                # Per-slice metric primitives
│
├── morphology/                             # Stages 1–3
│   ├── processor.py                        # MorphologyProcessor, BatchProcessor
│   ├── stack_to_3d.py                      # BatchNIfTIStacker (2D PNG → 3D NIfTI)
│   ├── slice_metrics.py                    # Metric helpers
│   ├── pipeline_tuning.py                  # Tuning pipeline logic
│   ├── tune_morpho.py                      # Tuning entrypoint
│   ├── apply_morpho.py                     # Standalone batch apply (legacy, kept)
│   └── README.md                           # Detailed morphology documentation
│
├── visualization/                          # Report and presentation figures
│   ├── visualize_processed_anomaly_maps.py    # Before/after body-mask comparison
│   ├── visualize_processed_prediction_masks.py # Raw vs. body-masked vs. filtered masks
│   ├── visualize_anomaly_thresholded_outputs.py # Anomaly map + threshold overlay
│   └── convert_to_bone_colormap.py            # NIfTI → bone-colormap PNG
│
├── config/
│   └── morpho_val.yaml                     # Morphology tuning configuration
│
├── results/                                # Report figures (comparison.png, table.png)
├── requirements.txt
├── README.md
│
└── old_code/                               # Unused / superseded files (not deleted)
    ├── train_cflow.py                      # Training scripts (ood-train domain)
    ├── train_fastflow.py
    ├── extract_fastflow.py
    ├── extract_cflow.py
    ├── fastflow_dataset.py
    ├── radimagenet_utils.py
    ├── apply_bodymask_pred.py              # Superseded by apply_bodymask_fastflow.py
    ├── PROJECT_BRIEF.md                    # Refactor planning notes
    ├── config.json                         # Unused placeholder config
    ├── filtered_metrics.json               # Stale output artifact
    └── flowchart2.md                       # Draft diagram scratch file
```

---

## Dependencies

```bash
pip install -r requirements.txt
```

Key packages: `anomalib==2.2.0`, `torch==2.8.0`, `nibabel==5.3.2`, `opencv-python==4.8.1.78`, `scipy==1.10.1`, `scikit-learn`.
