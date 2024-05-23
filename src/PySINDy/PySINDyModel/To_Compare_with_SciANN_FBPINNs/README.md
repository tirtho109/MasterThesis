# Model learning using SINDy algorithm (PySINDy package)

## Introduction

This project involves running a series of Jupyter notebooks to model and analyze data using various setup. The outputs include model coefficients, error metrics, and comparisons to evaluate the performance of different models.

## Prerequisites

- Python environment with Jupyter Notebook installed.
- Necessary libraries: [See Dependencies](https://github.com/tirtho109/MasterThesis/blob/TryPySINDy/src/PySINDy/installation.md)

## Instructions

1. **Run the Following Notebooks in Order:**

   - `SGModelFD.ipynb`
   - `SGModelSFD.ipynb`
   - `CompCoexModelFD.ipynb`
   - `CompCoexModelSFD.ipynb`
   - `CompSurvModelFD.ipynb`
   - `CompSurvModelSFD.ipynb`

2. **Outputs for Each Notebook:**

   - **Model Coefficients**
   - **Mean Squared Error (MSE) and Mean Absolute Error (MAE)**
   - **Threshold**
   - **Training Data Plot**

3. **Run the Final Notebook:**

   - `analysis.ipynb`

4. **Outputs for the Final Notebook:**

   - CSV file with the following columns:
     - Model Type
     - Number of Training Data
     - Time Limit
     - Best Differentiation Method
     - MSE
     - Coefficient
     - Lambda (Threshold)

   - Model Comparison Plot
   - Parameter Comparison Plot

## Detailed Descriptions

### Notebooks

- **SGModelFD.ipynb**: Runs the SG model with FiniteDifference(FD) as differentiation method.
- **SGModelSFD.ipynb**: Runs the SG model with SmoothedFiniteDifference(SFD) as differentiation method.
- **CompCoexModelFD.ipynb**: Compares coexistence models FD.
- **CompCoexModelSFD.ipynb**: Compares coexistence models with SFD.
- **CompSurvModelFD.ipynb**: Compares survival models with FD.
- **CompSurvModelSFD.ipynb**: Compares survival models with SFD.
- **analysis.ipynb**: Analyzes and compares the results from the previous notebooks.

## Running the Notebooks

1. Open each notebook in Jupyter Notebook.
2. Run all cells in the notebook to ensure all computations are completed.
3. Verify that the expected outputs are generated and saved.

## Notes

- Ensure that all necessary data files are in place before running the notebooks.
- Check the dependencies and install any missing packages using `pip install <package-name>`.
