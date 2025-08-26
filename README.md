# Cox Proportional Hazards Model for ICD Risk Prediction

This repository contains a Cox Proportional Hazards model implementation that replaces Random Forest models for predicting ICD (Implantable Cardioverter-Defibrillator) risk in patients.

## Overview

The implementation evaluates multiple model configurations:
- Sex-agnostic models (with and without undersampling)
- Sex-specific models (separate models for males and females)
- Different feature sets (guideline, benchmark, proposed, real proposed)

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Required data files

Ensure the following data files are in the correct locations:
- `/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff/icd_survival.xlsx`
- `/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff/no_icd_survival.csv`
- `/home/sunx/data/aiiih/projects/sunx/projects/ICD_sex_diff/NICM.xlsx`

## Running the Analysis

### Full evaluation (50 iterations)

```bash
python3 cox_model.py
```

This will:
- Load and preprocess the data
- Run 50 random train-test splits
- Evaluate all model configurations
- Save results to `cox_results.xlsx`

### Quick test (3 iterations)

```bash
python3 cox_test.py
```

This runs a quick test with only 3 iterations to verify the setup.

## Output

The analysis produces:
- **cox_results.xlsx**: Excel file with model performance metrics
- **Feature importance plots**: Hazard ratios and coefficients
- **Kaplan-Meier survival curves**: Risk stratification visualization

## Model Configurations

### Feature Sets

1. **Guideline**: NYHA Class, LVEF
2. **Benchmark**: Basic clinical and CMR features
3. **Proposed**: Extended feature set including advanced CMR parameters
4. **Real Proposed**: Modified proposed set with NYHA Class included

### Model Types

1. **Guideline model**: Rule-based (NYHA ≥ 2 & LVEF ≤ 35)
2. **Sex-agnostic**: Trained on all data
3. **Sex-agnostic (undersampled)**: Balanced training for fair comparison
4. **Sex-specific**: Separate models for males and females

## Key Metrics

- **C-index**: Concordance index for survival prediction
- **Accuracy, AUC, F1-score**: Classification metrics
- **Sensitivity, Specificity**: Clinical performance measures
- **Sex-specific metrics**: Performance by gender
- **Incidence rates**: Event rates by predicted risk group

## Statistical Methods

- **Cox Proportional Hazards regression** with L2 penalization
- **Stratified train-test splits** maintaining class balance
- **Per-sex undersampling** for fair model comparison
- **95% confidence intervals** using standard errors
- **Log-rank tests** for survival curve comparisons

## Files Description

- `cox_model.py`: Main implementation file
- `cox_test.py`: Quick test script
- `requirements.txt`: Python dependencies
- `cox_model_summary.md`: Detailed implementation summary
- `README.md`: This file

## Notes

- The code is optimized for clarity and statistical rigor
- All models use consistent random seeds for reproducibility
- Results include confidence intervals for robust inference
- Feature importance uses hazard ratios for interpretability