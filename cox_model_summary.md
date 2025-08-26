# Cox Proportional Hazards Model Implementation Summary

This implementation replaces the Random Forest models from `a.py` with Cox Proportional Hazards models for survival analysis, following the requirements:

## Key Features

### 1. Model Types Implemented
- **Sex-agnostic models**: Train on all data
- **Sex-agnostic with undersampling**: Ensures fair comparison with sex-specific models
- **Sex-specific models**: Separate models for males and females
- **Guideline model**: Rule-based using NYHA Class ≥ 2 and LVEF ≤ 35

### 2. Feature Selection Sets
- **Guideline**: NYHA Class, LVEF
- **Benchmark**: Age by decade, BMI, AF, Beta Blocker, CrCl>45, LVEF, QTc, NYHA>2, CRT, AAD, Significant LGE
- **Proposed**: Benchmark + DM, HTN, HLP, LVEDVi, LV Mass Index, RVEDVi, RVEF, LA EF, LAVi, MRF (%), Sphericity Index, Relative Wall Thickness, MV Annular Diameter, ACEi/ARB/ARNi, Aldosterone Antagonist
- **Real Proposed**: Similar to Proposed but includes NYHA Class and uses LGE Burden 5SD

### 3. Evaluation Process
- **50 random train-test splits** (70% train, 30% validation)
- **Stratified sampling** to maintain class balance
- **Undersampling** for sex-agnostic models to ensure fair comparison

### 4. Metrics Calculated
- Accuracy, AUC, F1-score, Sensitivity, Specificity
- Sex-specific metrics (male/female performance)
- Incidence rates by gender
- **Concordance index (C-index)** specific to Cox models

### 5. Key Differences from Random Forest Implementation

#### Model Training
```python
# Cox model training
cph = CoxPHFitter(penalizer=0.1)
cph.fit(cox_df, duration_col=time_col, event_col=event_col)

# Risk prediction using hazard ratios
hazard_ratios = model.predict_partial_hazard(X_test)
```

#### Feature Importance
- Uses **hazard ratios** and **coefficients** instead of RF feature importances
- Hazard ratio > 1: increased risk
- Hazard ratio < 1: decreased risk
- Coefficient magnitude indicates feature importance

#### Visualization
- Dual plots showing both hazard ratios and coefficients
- Color coding: red (guideline features), gray (standard CMR), blue (advanced CMR)

### 6. Survival Analysis
- Kaplan-Meier curves for risk stratification
- Log-rank tests for comparing survival between groups
- Primary endpoint: Appropriate ICD therapy or VT/VF/SCD
- Secondary endpoint: Death

### 7. Output
Results are exported to an Excel file containing:
- Model performance metrics with 95% confidence intervals
- Format: mean (CI_lower, CI_upper)
- One row per model-metric combination

## Usage

```python
# Run the full evaluation with 50 iterations
results, summary = multiple_random_splits_cox(clean_df, N=50)

# Save results to Excel
summary.to_excel('cox_results.xlsx', index=True, index_label='Model/Metric')

# Run inference for visualization
cox_inference_df = sex_specific_cox_inference(
    train_df=train_df,
    test_df=test_df,
    features=inference_features,
    label_col="VT/VF/SCD",
    survival_df=survival_df,
    seed=42,
    gray_features=gray_features,
    red_features=red_features
)
```

## Statistical Compliance
- Proper handling of censored data
- Concordance index for model evaluation
- Stratified sampling maintains class balance
- Confidence intervals based on standard errors
- Log-rank tests for survival comparisons