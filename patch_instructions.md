# Fix for Sex-Agnostic Model Prediction Bias in a.py

## Problem Summary
Your sex-agnostic model predicts almost all test samples as 0 (only 3/158 as high risk) due to:
1. **F1-score optimization bias** with class imbalance
2. **Overly conservative threshold selection**
3. **Inadequate handling of rare positive class**

## Root Cause Analysis

### Current Problematic Code (Lines 309-314):
```python
def find_best_threshold(y_true, y_scores):
    """Find the probability threshold that maximizes the F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.nanargmax(f1_scores[:-1])
    return thresholds[best_idx]  # ← This returns very high thresholds!
```

**Why this fails:**
- F1-score optimization favors high precision over high recall
- With class imbalance, it's easier to get high precision by predicting fewer positives
- Results in thresholds that classify almost everything as negative

## Solution: Replace with Improved Threshold Selection

### PATCH 1: Replace find_best_threshold function

**Location:** Lines 309-314 in a.py

**Replace this:**
```python
def find_best_threshold(y_true, y_scores):
    """Find the probability threshold that maximizes the F1 score."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.nanargmax(f1_scores[:-1])
    return thresholds[best_idx]
```

**With this:**
```python
def find_best_threshold(y_true, y_scores, method='youden'):
    """Find optimal threshold using multiple strategies to handle class imbalance."""
    if method == 'youden':
        # Youden's J statistic (sensitivity + specificity - 1)
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return thresholds[best_idx]
    
    elif method == 'f1':
        # Original F1-based method (kept for comparison)
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        best_idx = np.nanargmax(f1_scores[:-1])
        return thresholds[best_idx]
    
    elif method == 'fixed_recall':
        # Target a specific recall (e.g., 0.8 to ensure we catch most positive cases)
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        target_recall = 0.8
        valid_indices = recalls[:-1] >= target_recall
        if not np.any(valid_indices):
            return thresholds.min()
        valid_precisions = precisions[:-1][valid_indices]
        valid_thresholds = thresholds[valid_indices]
        best_idx = np.argmax(valid_precisions)
        return valid_thresholds[best_idx]
    
    elif method == 'percentile':
        # Use a percentile-based threshold (e.g., top 10% of predictions)
        percentile = 90  # Top 10%
        return np.percentile(y_scores, percentile)
    
    else:
        # Default to youden if method not recognized
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return thresholds[best_idx]
```

### PATCH 2: Update function calls to use new threshold method

**Location:** Line 395 in rf_evaluate function

**Replace this:**
```python
threshold = find_best_threshold(y_train.values, oof_proba)
```

**With this:**
```python
threshold = find_best_threshold(y_train.values, oof_proba, method='youden')
```

**Location:** Line 825 in train_sex_specific_model function

**Replace this:**
```python
best_threshold = find_best_threshold(y_train, cv_probs)
```

**With this:**
```python
best_threshold = find_best_threshold(y_train, cv_probs, method='youden')
```

**Location:** Line 892 in train_sex_agnostic_model function

**Replace this:**
```python
best_threshold = find_best_threshold(y_train, cv_probs)
```

**With this:**
```python
best_threshold = find_best_threshold(y_train, cv_probs, method='youden')
```

### PATCH 3: Add diagnostic information to sex_agnostic_model_inference

**Location:** After line 1410 in sex_agnostic_model_inference function

**Add this diagnostic code:**
```python
    # Add diagnostic information
    print(f"Optimal probability threshold determined from training data: {best_threshold:.4f}")
    
    # Analyze probability distribution
    print(f"Test probabilities – min: {test_probs.min():.4f}, max: {test_probs.max():.4f}, mean: {test_probs.mean():.4f}")
    
    # Show prediction distribution with current threshold
    n_high_risk = test_preds.sum()
    n_low_risk = len(test_preds) - n_high_risk
    print(f"Using threshold {best_threshold:.4f}:")
    print(f"  High risk predictions: {n_high_risk} ({n_high_risk/len(test_preds)*100:.1f}%)")
    print(f"  Low risk predictions: {n_low_risk} ({n_low_risk/len(test_preds)*100:.1f}%)")
    
    # Try alternative thresholds for comparison
    print(f"\nAlternative threshold analysis:")
    for alt_method in ['f1', 'fixed_recall', 'percentile']:
        try:
            # You would need to store training probabilities to compute this properly
            # For now, just show different percentile thresholds
            if alt_method == 'percentile':
                alt_threshold = np.percentile(test_probs, 90)
                alt_preds = (test_probs >= alt_threshold).astype(int)
                alt_high = alt_preds.sum()
                print(f"  {alt_method:>12} (t={alt_threshold:.4f}): {alt_high} high risk ({alt_high/len(test_preds)*100:.1f}%)")
        except:
            continue
```

## Quick Test Method

After making these changes, test different threshold methods:

```python
# Test with different methods
methods_to_try = ['youden', 'fixed_recall', 'percentile']

for method in methods_to_try:
    print(f"\n=== Testing with {method} method ===")
    result_df = sex_agnostic_model_inference(
        train_df=train_df,
        test_df=test_df,
        features=features,
        label_col='VT/VF/SCD',
        survival_df=survival_df,
        seed=42,
        # You'll need to modify the function to accept this parameter
    )
```

## Expected Results

**Before fix:**
- High Risk: 3/158 samples (1.9%)
- Low Risk: 155/158 samples (98.1%)

**After fix (expected):**
- **Youden method**: ~15-25% high risk (better balance)
- **Fixed recall method**: Ensures 80%+ of true positives are caught
- **Percentile method**: Exactly 10% high risk (top decile)

## Verification Steps

1. Check that more samples are predicted as high risk (>3)
2. Verify that true positive cases have higher prediction probabilities
3. Ensure survival analysis shows meaningful differences between risk groups
4. Compare performance metrics across different threshold methods

## Additional Improvements (Optional)

For even better results, consider:

1. **SMOTE or ADASYN** for handling class imbalance during training
2. **Stratified sampling** to ensure balanced train/test splits
3. **Cost-sensitive learning** with custom loss functions
4. **Ensemble methods** combining multiple threshold strategies

The key insight is that **F1-score optimization is inappropriate for severely imbalanced datasets** in clinical applications where missing positive cases (low recall) is more costly than false alarms (low precision).