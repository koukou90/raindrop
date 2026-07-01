# Raindrop: TSFNet for Precipitation Nowcasting from Disdrometer Spectra

This repository contains the source code used for the manuscript:

**A Bin-Aware Deep Learning Framework for Precipitation Nowcasting from Disdrometer Spectra**

The code implements the Triple-Stream Fusion Network (TSFNet) and the baseline models used to evaluate minute-scale precipitation nowcasting from OTT Parsivel2 drop-size distribution (DSD) spectra.

Repository: <https://github.com/koukou90/raindrop>

## What is included

The public repository is provided as individual source files, not as a compressed archive. It contains the code needed to define, train, evaluate and compare the models described in the manuscript.

Expected repository layout:

```text
raindrop/
+-- baselines/       # Baseline and ablation models
+-- data_loader/     # Data loading, preprocessing and sample construction
+-- exp/             # Training, evaluation and recipe utilities
+-- my_model/        # TSFNet model definitions
+-- utils/           # Metrics and feature-processing utilities
+-- .gitignore
+-- README.md
+-- run.py           # Main command-line entry point
```

## Data availability

The observational dataset is not included in this repository because it contains non-public disdrometer observations from field stations. The repository therefore provides the model and experiment code, but not the private data files required to reproduce the numerical results directly.

Users who have access to data in the same format can run the training and evaluation pipeline by placing the preprocessed files under a local data directory. The expected structure is:

```text
data/
+-- <site_name>/
    +-- <site_name>_dsd.csv
    +-- <site_name>_params.csv
```

where `<site_name>` is one of:

```text
W2127_Haichaoba
W2128_Haichaoyinsi
W2129_buligou
```

The DSD file should contain a `Timestamp` column and spectral columns named `data1`, `data2`, ..., `data64`. The first 32 spectral columns represent bin-wise concentration features and the last 32 represent fall-velocity features. The parameter file should contain `Timestamp`, `RainRate`, `Dm`, `LogNw`, `LWC` and `Z`.

## Main models

The main model used in the manuscript is:

```text
TripleStreamV3
```

The repository also includes baseline and ablation models, including:

```text
BaselineLightGBM
BaselineMLP
BaselineLSTM
BaselineCNNTransformer
BaselineiTransformer
BaselinePatchTST
BaselineDLinear
AblateV3NoPersistence
AblateV3NoMixGate
AblateV3NoBinAware
AblateV3NoAux
AblateV3NoStreamGate
```

## Environment

The experiments in the manuscript were run with Python 3.9 and PyTorch. A typical environment can be prepared with:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Install the required Python packages:

```bash
pip install numpy pandas scipy scikit-learn matplotlib torch lightgbm
```

If a GPU is available, install the PyTorch build that matches your local CUDA version by following the official PyTorch installation instructions.

## Quick test without private data

Because the dataset is not public, the quickest repository-level test is a Python source check. It verifies that the individual source files are present and syntactically valid, without requiring access to the private data.

Run from the repository root:

```bash
python -m compileall baselines data_loader exp my_model utils run.py
```

A successful quick test should finish without syntax errors.

You can also inspect the command-line interface:

```bash
python run.py --help
```

If this command reports a missing local module, please check that all source files imported by `run.py` and `exp/exp.py` are present in the public repository.

## Running an experiment with accessible data

After placing preprocessed data in the expected directory structure, train and evaluate TSFNet with:

```bash
python run.py 
  --model_name TripleStreamV3 
  --v3_recipe t5_boost 
  --site_name W2127_Haichaoba 
  --data_dir data 
  --seq_len 10 
  --pred_len 5 
  --batch_size 64 
  --epochs 80 
  --out_path out
```

On Linux or macOS, replace the line-continuation character `^` with `\`.

To run another site, change `--site_name` to `W2128_Haichaoyinsi` or `W2129_buligou`.

To run a baseline model, change `--model_name`, for example:

```bash
python run.py --model_name BaselineLightGBM --site_name W2127_Haichaoba --data_dir data
```

Outputs are written to the directory specified by `--out_path`.

## Notes on reproducibility

The code uses chronological event-level splitting to reduce temporal leakage between training, validation and test sets. Random seeds can be controlled with:

```bash
python run.py --seed 2025
```

Exact numerical reproduction of the manuscript results requires the private observational dataset and the same preprocessing choices described in the paper.

## License

None
