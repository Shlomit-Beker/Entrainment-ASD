
# ASD Entrainment Analysis Code

## Overview
This repository contains MATLAB code for preprocessing and analyzing EEG and behavioral data from a study on cortical entrainment in children with Autism Spectrum Disorder (ASD) and typically developing (TD) peers.

## Structure

- `preprocessing/`: Preprocessing script (`Preprocess_EEG_All.m`) for EEG data.
- `analysis/`: Scripts for ERP, frequency analysis, phase-behavior coupling, TRF models, and EEG-clinical correlations.
- `params/`: Parameter definition files for different components.
- `triggers/`: Trigger extraction scripts.
- `layout/`: EEG layout file (`64_lay.mat`).
- `docs/`: Original Readme and related documentation.

## Main Scripts

1. `Preprocess_EEG_All.m` — EEG preprocessing using FieldTrip.
2. `Run_Entrainment_Analyses.m` — Calls functions for:
   - ERP analysis
   - Time-frequency and phase-locking analyses
   - Inactive mode
   - Behavior–EEG phase coupling
   - Temporal Response Function (TRF) modeling
   - Correlation with clinical measures

## Dependencies

- [FieldTrip toolbox](https://www.fieldtriptoolbox.org/) (required for all analyses)

