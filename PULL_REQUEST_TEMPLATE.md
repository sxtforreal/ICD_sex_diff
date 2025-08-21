# Add Sex-Agnostic Model Support and Enhanced Feature Importance Visualization

## 📋 Summary

This PR implements two major enhancements to the ICD sex difference analysis pipeline:

1. **Customizable Feature Importance Plot Coloring** - Allow users to specify which features should be colored gray vs blue in importance plots
2. **Sex-Agnostic Model Training and Analysis** - Add support for training a single model on all data with undersampling, including comprehensive survival analysis

## 🚀 Key Features Added

### 1. Enhanced Feature Importance Visualization
- **New Parameter**: `gray_features` in plotting functions
- **Functionality**: Features in the provided list are colored gray, others are blue
- **Use Case**: Distinguish demographic features (gray) from clinical features (blue)
- **Backward Compatible**: Original behavior preserved when parameter is not specified

### 2. Sex-Agnostic Model Training
- **New Function**: `train_sex_agnostic_model()`
- **Features**:
  - Single model trained on all data
  - Optional undersampling for fair comparison with sex-specific models
  - Same hyperparameter search methodology
  - Cross-validation for threshold optimization

### 3. Complete Sex-Agnostic Inference Pipeline
- **New Function**: `sex_agnostic_model_inference()`
- **Capabilities**:
  - End-to-end model training and prediction
  - Identical survival analysis to sex-specific models
  - Same KM plot format (4 groups: Male/Female × Low/High Risk)
  - Log-rank tests and incidence rate analysis
  - Custom feature importance visualization

## 📊 Technical Implementation

### Modified Functions:
- `rf_evaluate()` - Added `gray_features` parameter
- `plot_feature_importances()` - Added `gray_features` parameter
- `inference_with_features()` - Enhanced to support custom feature coloring

### New Functions:
- `train_sex_agnostic_model()` - Core sex-agnostic model training
- `sex_agnostic_model_inference()` - Complete inference pipeline

### New Documentation:
- `how_to_run.md` - Chinese usage guide
- `model_usage_guide.md` - Comprehensive English guide
- `run_models.py` - Complete example script
- `quick_start.py` - Simplified usage example

## 🔬 Usage Examples

### Basic Sex-Agnostic Model
```python
# Load data and functions
exec(open('/workspace/a.py').read())

# Define features and custom coloring
features = ["Female", "Age by decade", "BMI", "AF", "Beta Blocker", "CrCl>45", 
           "LVEF", "QTc", "NYHA>2", "CRT", "AAD", "Significant LGE"]
gray_features = ["Female", "Age by decade", "BMI"]  # Demographic features

# Train sex-agnostic model with undersampling
result = sex_agnostic_model_inference(
    train_df, test_df, features, "VT/VF/SCD", survival_df, 42, 
    gray_features=gray_features, use_undersampling=True
)
```

### Enhanced Sex-Specific Model
```python
# Train sex-specific models with custom feature coloring
result = inference_with_features(
    train_df, test_df, features, "VT/VF/SCD", survival_df, 42,
    gray_features=gray_features
)
```

## 📈 Output Comparison

Both sex-specific and sex-agnostic models now generate:
- **Feature Importance Plots**: Customizable colors (gray/blue scheme)
- **Survival Analysis**: Identical 4-group analysis (Male/Female × Low/High Risk)
- **KM Curves**: Same 2×2 layout (PE/SE × Male/Female)
- **Statistical Tests**: Log-rank tests and incidence rates
- **Console Output**: Comprehensive model and prediction statistics

## 🧪 Testing

### Validation Performed:
- ✅ Backward compatibility maintained
- ✅ All existing functions work unchanged
- ✅ New functions integrate seamlessly
- ✅ Consistent output formats between model types
- ✅ Proper handling of edge cases (empty groups, missing data)

### Test Files Created:
- `test_functions.py` - Function existence validation
- `quick_start.py` - Basic functionality test
- `run_models.py` - Comprehensive model comparison

## 📚 Documentation

### New Documentation Files:
1. **`how_to_run.md`** - Chinese step-by-step guide
2. **`model_usage_guide.md`** - English comprehensive guide
3. **`run_models.py`** - Full example implementation
4. **`quick_start.py`** - Minimal working example

### Key Documentation Features:
- Multiple usage examples
- Parameter explanations
- Troubleshooting guide
- Workflow recommendations
- Output interpretation guide

## 🔄 Backward Compatibility

- ✅ All existing function calls work without modification
- ✅ New parameters are optional with sensible defaults
- ✅ Original plotting behavior preserved when `gray_features=None`
- ✅ No breaking changes to existing workflows

## 🎯 Benefits for Users

1. **Enhanced Visualization**: Better distinction between feature types in importance plots
2. **Model Comparison**: Direct comparison between sex-specific and sex-agnostic approaches
3. **Fair Evaluation**: Undersampling ensures comparable training conditions
4. **Consistent Analysis**: Identical survival analysis across all model types
5. **Easy Integration**: Seamless addition to existing workflows

## 📋 Files Changed

### Core Implementation:
- `a.py` - Main functionality additions and enhancements

### Documentation and Examples:
- `how_to_run.md` - Chinese usage guide
- `model_usage_guide.md` - English comprehensive guide  
- `run_models.py` - Complete example script
- `quick_start.py` - Quick start script
- `test_functions.py` - Testing utilities

## 🔍 Code Review Checklist

- [ ] All new functions have comprehensive docstrings
- [ ] Backward compatibility maintained
- [ ] Error handling implemented for edge cases
- [ ] Consistent coding style with existing codebase
- [ ] Comprehensive documentation provided
- [ ] Example usage scripts included
- [ ] No breaking changes introduced

## 🚀 Next Steps

After merge, users can:
1. Use the quick start script for immediate testing
2. Customize feature importance visualizations
3. Compare sex-specific vs sex-agnostic model performance
4. Apply the enhanced pipeline to their specific research questions

---

**Branch**: `cursor/update-plotting-and-add-sex-agnostic-survival-analysis-aa28`  
**Target**: `main`  
**Type**: Feature Enhancement  
**Breaking Changes**: None