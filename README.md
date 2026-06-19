# TSFNet Code for Disdrometer-Based Precipitation Nowcasting

This repository provides the source code for the manuscript:

**A Bin-Aware Deep Learning Framework for Precipitation Nowcasting from Disdrometer Spectra**

The code implements TSFNet, a deep learning framework for minute-scale precipitation nowcasting from Parsivel disdrometer drop-size distribution (DSD) spectra. It also includes the baseline models and experiment scripts used in the manuscript.

## Repository Structure

```text
baselines/      Baseline models for comparison
data_loader/    Data loading and preprocessing scripts
exp/            Training and evaluation scripts
my_model/       TSFNet model implementation
utils/          Utility functions
run.py          Main running script
```

## Requirements

The code was developed with Python 3.9 and PyTorch. Main packages include:

```text
pytorch
numpy
pandas
scikit-learn
lightgbm
matplotlib
```

## Usage

After installing the required packages, run:

```bash
python run.py
```

## Citation

If you use this code, please cite the associated manuscript:

```text
Mao, W., Kou, M., Guo, P., Ren, J., Chen, P.
A Bin-Aware Deep Learning Framework for Precipitation Nowcasting from Disdrometer Spectra.
Submitted to Computers & Geosciences.
```
