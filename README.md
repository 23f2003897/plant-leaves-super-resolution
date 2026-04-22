# Plant Leaves Super-Resolution Challenge

A Conditional GAN (cGAN) solution for 4x super-resolution of degraded crop leaf images, built for the IITM DLGenAI NPPE2 competition on Kaggle.

## Problem Statement

Automated drones and cheap mobile sensors deployed to photograph crop leaves transmit heavily degraded 32x32 images due to hardware limitations and compression. The task is to reconstruct high-fidelity 128x128 images from these degraded inputs using a Conditional GAN.

- **Input:** Degraded 32x32 RGB leaf images
- **Output:** Reconstructed 128x128 RGB leaf images
- **Metric:** Mean Absolute Error (MAE) — lower is better
- **Competition Baseline:** 17.3529644
- **Final Score:** 15.9

## Architecture

### Generator — RCAB with PixelShuffle

The generator uses Residual Channel Attention Blocks (RCAB) to process features at LR resolution (32x32), then upsamples to 128x128 using two sequential PixelShuffle x2 operations.

```
Input (3, 32, 32)
    -> Initial Conv
    -> 16 x RCAB (Residual Channel Attention Block)
        -> Conv -> ReLU -> Conv -> Channel Attention -> Residual Scale
    -> Body Conv
    -> PixelShuffle x2 -> (3, 64, 64)
    -> PixelShuffle x2 -> (3, 128, 128)
    -> Final Conv
    -> Add Bicubic Upscaled Input (global residual)
    -> Clamp [0, 1]
Output (3, 128, 128)
```

### Discriminator — Conditional PatchGAN style

The discriminator is conditioned on the bicubic-upscaled LR image. It receives a 6-channel input (3 channels from LR upscaled + 3 channels from SR or HR image) and outputs a real/fake score.

```
Input: concat(LR_bicubic, SR_or_HR) -> (6, 128, 128)
    -> Conv blocks with stride
    -> AdaptiveAvgPool
    -> Linear -> scalar score
```

## Training Details

| Parameter | Value |
|---|---|
| Batch size | 8 |
| Epochs | 150 |
| Generator LR | 2e-4 |
| Discriminator LR | 2e-5 |
| Optimizer | AdamW (weight_decay=1e-4) |
| Scheduler | CosineAnnealingWarmRestarts (T_0=50) |
| Loss G | Charbonnier + 0.005 x Perceptual + 0.001 x Adversarial |
| Loss D | BCEWithLogits |
| Gradient clipping | max_norm=1.0 |

## Loss Functions

- **Charbonnier Loss** — smooth L1 variant, better than plain L1 for super-resolution
- **Perceptual Loss** — computed using VGG19 features (loaded from the provided dataset file, no internet required)
- **Adversarial Loss** — standard GAN loss with BCEWithLogits

## Inference

8-way Test Time Augmentation (TTA) is applied during inference:
- 4 rotations (0, 90, 180, 270 degrees)
- Each rotation with and without horizontal flip
- Predictions are averaged after undoing all transforms

## Dataset

| Split | Images |
|---|---|
| Train HR (128x128) | 1642 |
| Train LR (32x32) | 1642 |
| Test LR (32x32) | 495 |

Augmentations applied during training: horizontal flip, vertical flip, random rotation (0/90/180/270 degrees). The same transformation is applied identically to both LR and HR images to preserve spatial correspondence.

## Results

| Approach | Leaderboard MAE |
|---|---|
| Bicubic baseline | ~19.5 |
| Competition baseline | 17.35 |
| UNet + RRDB (L1 + Perceptual) | 16.1 |
| RCAB + PixelShuffle cGAN | **15.9** |

## Rule Compliance

- Generator and Discriminator are randomly initialized. No pretrained weights are used for the core task.
- VGG19 weights are loaded only from the provided `vgg19_weights.pth` file in the competition dataset, used only for perceptual loss computation.
- The notebook runs with Internet Access OFF and External Data DISALLOWED on Kaggle.
- Submission format follows the competition specification: space-separated pixel values, 49152 integers per row, range [0, 255].

## Repository Structure

```
plant-leaves-super-resolution/
    final_cgan_notebook.ipynb   # Main notebook with EDA, training, inference
    README.md                   # This file
```

## How to Run

1. Add the competition dataset to your Kaggle notebook
2. Set Internet Access to OFF
3. Run all cells in order
4. Submit the generated `submission.csv`

## Key Findings

**What worked:**
- Raw [0,1] pixel values with no normalization
- 8-way TTA using torch.rot90 on tensors
- Charbonnier loss as the primary objective
- Residual scaling (res_scale=0.1) inside RCAB blocks
- Global bicubic residual connection
- AdamW with weight decay for regularization

**What did not work:**
- [-1,1] normalization with Tanh output
- GAN training without warmup
- Frequency loss with high perceptual loss weight
- Training PixelShuffle on full data for 500 epochs without a validation split
