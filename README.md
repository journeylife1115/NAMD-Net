# NAMD-Net

Selected key implementation code for **NAMD-Net: A Dual-Branch Model for Significant Wave Height Prediction Based on Noise-Assisted Multivariate Empirical Mode Decomposition**.

This is intentionally a compact research-code release for transparency and reproducibility, rather than a complete archive of exploratory experiments, plotting scripts, benchmark implementations, trained weights, or intermediate files.

## Method overview

1. Use SWH, WSPD, GST, and APD as inputs.
2. Jointly decompose the multivariate series using NA-MEMD.
3. Use permutation entropy to characterize/group decomposed components.
4. Model high-frequency components with WaveNet and low-frequency components with LSTM.
5. Combine component forecasts to obtain the final multi-step SWH forecast.

The main forecasting setup uses a 48-hour historical window and a direct 24-step output.

## Contents

- `models.py` — WaveNet and LSTM modules used in the dual-branch framework.
- `data_utils.py` — chronological splitting, gap-aware sliding windows, and training-only scaling utilities.
- `permutation_entropy.py` — permutation-entropy calculation and grouping.
- `config.py` — key experimental hyperparameters.
- `requirements.txt` — main dependencies.

## NA-MEMD

The NA-MEMD algorithm itself is **not redistributed here**. The study used the publicly available implementation from:

**PiethonProgram / NA-MEMD-and-MEMD**  
https://github.com/PiethonProgram/NA-MEMD-and-MEMD

Please obtain the NA-MEMD/MEMD implementation from the original repository and follow its applicable license terms.

Principal settings used in the study included 50 direction vectors, noise intensity 0.1, four auxiliary noise channels, and random seed 42.

## Data

The study used hourly NDBC observations from buoy stations 41013 and 41046. Raw observations are not redistributed in this repository. Prepare the four input variables (SWH, WSPD, GST, APD) in chronological order.

Windows crossing temporal discontinuities should be excluded. Scaling parameters are estimated from the training subset and then applied to validation/test data.

## Scope

This repository documents the key implementation logic of NAMD-Net. It does not include the authors' complete experimental workspace, exploratory scripts, all comparison models, generated figures, local paths, trained weights, or intermediate decomposition files.

## Environment

The experiments were conducted using the following environment:

- Python 3.9.20
- PyTorch 2.2.1
- CUDA 11.8
- cuDNN 8.7.0
- Windows 10

The main Python dependencies are listed in `requirements.txt`.

## Citation

If you use this code, please cite the associated NAMD-Net manuscript.
