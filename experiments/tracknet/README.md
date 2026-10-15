# Temporal handball ball-localization experiment

This experiment fine-tunes pretrained TrackNet features on the companion project's
hand-clicked ball centres. It does not alter the production detection/tracking pipeline.

## Data

Source: `../rfdetr-handball-finetune/datasets/expansion-v1/`.
Each example contains three consecutive BGR frames, newest first, resized to
640x360 and divided by 255. Only the newest frame is supervised. All target
images must match the labelled PNG pixel-for-pixel; inconsistent video seeks
are recovered by exact matching within a short continuously decoded window.
Original match-separated train/valid/test assignments are preserved.
Native frame intervals are preserved (sources range from about 25 to 50 fps).
No-ball labels produce empty heatmaps; they do not distinguish occlusion from
out-of-frame or otherwise invisible balls.

## Model and training

`model.py` is the downloaded architecture from
https://github.com/yastrebksv/TrackNet/blob/main/model.py (upstream revision recorded
in `runs/tracknet_smoke/upstream_commit.json`). Weights:
https://huggingface.co/vishnushenoy09/tracknet-v1-tennis .

All 17 convolutional feature blocks are initialized from the tennis checkpoint.
The 256-class intensity head is replaced by a newly initialized one-channel
heatmap head. This is a TrackNet-derived model, not an exact reproduction of the
original training recipe. Gaussian centre targets have sigma 2 input pixels.
Gaussian focal loss follows the objective described in CenterNet's implementation:
https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/losses.py .
Pretrained batch-normalization running statistics stay fixed; affine parameters
and all convolution weights are fine-tuned. Only synchronized horizontal flips
are used for augmentation. Seed: 42; AdamW; learning rate 1e-4.

## Commands

Run from this repository using its existing CUDA environment (do not sync dependencies):

```bash
.venv/bin/python -m experiments.tracknet.experiment prepare
.venv/bin/python -m experiments.tracknet.experiment train --smoke
.venv/bin/python -m experiments.tracknet.experiment train
.venv/bin/python -m experiments.tracknet.experiment evaluate
```

The smoke run fits 12 training examples and is an optimization/geometry check,
not an accuracy measurement. Full training selects the checkpoint and confidence
threshold on validation F1 only, using a predetermined threshold grid. Training
stops at 30 epochs or after 7 epochs without validation-F1 improvement, with at
least 10 epochs. Existing best checkpoints are protected against overwrite.

A correct prediction is within 10 pixels at 640x360 of the labelled centre
(30 native pixels on 1920x1080). Wrong-location predictions count both as false
positives and missed balls. On non-16:9 sources the normalized distance is
anisotropic in native pixels. Confidence is a heatmap peak, not a calibrated
probability. At most one ball is emitted per frame; no interpolation or smoothing.

Generated triplets and provenance: `data/cache/tracknet-handball-v1/`.
Checkpoints, logs, predictions and evaluation: `runs/tracknet_handball_v1/`.

For a like-for-like comparison with the existing detector and a review video:

```bash
../rfdetr-handball-finetune/.venv/bin/python experiments/tracknet/detector_baseline.py
.venv/bin/python -m experiments.tracknet.report
```

RF-DETR runs in the companion project's existing environment and reads its
checkpoint; neither the companion data nor environment is modified. Its
confidence threshold is selected on the same validation set using the same
centre-distance metric. This metric differs from detector box-IoU mAP/F1.
The BHC-FAG review video uses the existing 500-frame detector cache, whose
confidence floor is 0.20; effective video thresholds are saved separately.
The video is qualitative and is not used to select hyperparameters.

## Result: 2026-09-18

All 1,050 target images were matched exactly to source video frames. Three
required recovery from a one-frame seek discrepancy. Split sizes were
550/300/200; visible-ball counts were 353/218/138. No additional manual labels
were needed. The 12-example smoke check reached 8/8 positive localizations and
0 false positives on four absent targets at its selected threshold.

Full training stopped after epoch 13 (about 8 minutes of train/validation time).
The selected checkpoint was epoch 6, validation F1 0.3064, threshold 0.20.
RF-DETR's independently validation-selected threshold was 0.25.

On the 200 untouched test targets (138 visible balls), with a 10-input-pixel
centre-distance tolerance:

| Model | Correct ball positions | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Existing RF-DETR ball-v1 | 81/138 | 66.4% | 58.7% | 62.3% |
| Fine-tuned TrackNet-derived heatmap model | 21/138 | 28.0% | 15.2% | 19.7% |

Nineteen correct positions were shared, two were TrackNet-only, 62 were
detector-only, and 55 were missed by both. These overlap counts do not measure
an implemented fusion method. TrackNet emitted 54 false positions, including
19 on absent-ball images. RF-DETR emitted 41 false positions, including 20 on
absent-ball images.

This configuration does not justify integration. The result is specific to this
640x360, adapted single-heatmap-head fine-tune, dataset, and optimization recipe;
it does not establish a limit for temporal handball detection. Labels lack a
reliable held/flight category, so the aggregate test score does not separately
answer moving-ball performance. The test set is now inspected and should not be
used for tuning subsequent runs.

Artifacts:
- `runs/tracknet_handball_v1/train/best.pt`
- `runs/tracknet_handball_v1/comparison.json`
- `runs/tracknet_handball_v1/test_predictions.json`
- `runs/tracknet_handball_v1/test_contact_sheet.jpg`
- `runs/tracknet_handball_v1/BHC-FAG_comparison.mp4` (500 frames, 50 fps;
  magenta TrackNet, cyan RF-DETR; qualitative, no ground truth overlay)

Verification: 66 focused tests passed (three experiment-contract tests and 63
existing ball tests). The saved H.264 video was reopened and verified to contain
500 frames at 50 fps. The existing ball pipeline was not changed.
