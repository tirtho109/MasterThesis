# Competition Model Training

## Overview
This Python script, `PySINDy_SaturatedGrowthModel.py`, is designed to train a Saturated Growth Model using PySINDy package. It allows for flexible configuration of parameters and is suitable for parameter estimation and model prediction of SG model.

## Prerequisites
- python=3.9.19
- cvxpy==1.4.2
- h5py==3.10.0
- clarabel==0.7.1
- ipykernel==6.29.3
- ipython==8.12.3
- matplotlib==3.5.3
- numpy==1.26.1
- pandas==1.3.5
- scikit-learn==1.3.2
- scipy==1.13.0
- pysindy==1.7.5

## Features
- Customizable optimizer, currently it has SR3 and SINDyPI.
- Feature Library, only poly, fourier, combined(poly+fourier) and custom. 
- Options for generating training data with noise.
- Visualization capabilities for model analysis.

## Usage
Execute with:
```bash
python PySINDy_SaturatedGrowthModel.py

## Additional Arguments
- ´-ic´, ´--initial_conditions´: Initial conditions (u0) for the model. Default: [0.01].
- ´--tend´: End time for the model simulation. Default: 24.

## Training Data Generator Arguments
- ´-nx´, ´--numx´: Number of nodes in X. Default: 100.
- ´-nT´, ´--nTest´: Number of test points. Default: 200.
- ´--sparse´: Sparsity of training data. Default: False.
- ´-tl´, ´--time_limit´: Time window for the training data. Default: [10, 24].
- ´-nl´, ´--noise_level´: Level of noise in training data. Default: 0.005.
- ´-sf´, ´--show_figure´: Show training data plot. Default: False.
- ´-nt´, ´--num_threshold´: Number of thresholds to be scanned. Default: 10.
- ´-mt´, ´--max_threshold´: Maximum threshold to be scanned. Default: 1.

## Output Arguments
- ´-op´, ´--outputpath´: Output path. Default: ./CompModels.
- ´-of´, ´--outputprefix´: Output prefix. Default: res.

## Optimizer Arguments
- ´-fl´, ´--feature_library´: Feature library to choose for SINDy algorithm. Default: custom.
- ´-po´, ´--poly_order´: Polynomial order for polynomial library. Default: 2.
- ´-opt´, ´--optimizer´: Optimizer for SINDy algorithm. Default: SR3.
- ´-th´, ´--thresholder´: Thresholder for SINDy optimizer algorithm. Default: l1.

## Miscellaneous Arguments
- ´--plot´: Plot the model. Default: False.

## Outputs
- Training data plots (optional).
- Model comparison.
- Pareto curve.
- MSE and MAE.



