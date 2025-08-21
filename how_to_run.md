# 如何运行Sex-Specific和Sex-Agnostic模型

## 🚀 快速开始

### 第一步：确保环境准备就绪
```python
# 在Python环境中，首先加载a.py中的所有函数和数据
exec(open('/workspace/a.py').read())

# 这会加载：
# - train_df, test_df (训练和测试数据)
# - survival_df (生存数据)  
# - 所有新的模型函数
```

### 第二步：定义特征和参数
```python
# 选择特征集（推荐从benchmark开始）
features = [
    "Female", "Age by decade", "BMI", "AF", "Beta Blocker", "CrCl>45", 
    "LVEF", "QTc", "NYHA>2", "CRT", "AAD", "Significant LGE"
]

# 定义要在特征重要性图中标为灰色的特征（可选）
gray_features = ["Female", "Age by decade", "BMI"]  # 人口统计学特征

# 设置随机种子
seed = 42
```

## 🔬 运行模型

### 方法1：Sex-Specific模型（男女分别训练）
```python
print("运行Sex-Specific模型...")
sex_specific_result = inference_with_features(
    train_df=train_df,
    test_df=test_df,
    features=features,
    labels="VT/VF/SCD",
    survival_df=survival_df,
    seed=seed,
    gray_features=gray_features  # 新功能：自定义特征颜色
)
```

**这会输出：**
- 男性模型特征重要性图（灰色=人口统计学特征，蓝色=临床特征）
- 女性模型特征重要性图
- 4组生存分析：Male-Low Risk, Male-High Risk, Female-Low Risk, Female-High Risk
- Kaplan-Meier生存曲线（2x2布局）
- Log-rank检验p值

### 方法2：Sex-Agnostic模型（单一模型，使用欠采样）
```python
print("运行Sex-Agnostic模型（欠采样）...")
sex_agnostic_result = sex_agnostic_model_inference(
    train_df=train_df,
    test_df=test_df,
    features=features,
    label_col="VT/VF/SCD",
    survival_df=survival_df,
    seed=seed,
    gray_features=gray_features,    # 新功能：自定义特征颜色
    use_undersampling=True          # 新功能：使用欠采样公平比较
)
```

**这会输出：**
- Sex-agnostic模型特征重要性图（相同的颜色方案）
- 相同的4组生存分析（但使用单一模型的预测）
- 相同格式的Kaplan-Meier曲线和统计分析

### 方法3：Sex-Agnostic模型（不使用欠采样）
```python
print("运行Sex-Agnostic模型（无欠采样）...")
sex_agnostic_no_undersample = sex_agnostic_model_inference(
    train_df=train_df,
    test_df=test_df,
    features=features,
    label_col="VT/VF/SCD",
    survival_df=survival_df,
    seed=seed,
    gray_features=gray_features,
    use_undersampling=False         # 不使用欠采样
)
```

## 📊 完整示例脚本

```python
# 完整的运行脚本
def run_all_models():
    # 1. 加载数据和函数
    exec(open('/workspace/a.py').read())
    
    # 2. 设置参数
    features = ["Female", "Age by decade", "BMI", "AF", "Beta Blocker", "CrCl>45", 
               "LVEF", "QTc", "NYHA>2", "CRT", "AAD", "Significant LGE"]
    gray_features = ["Female", "Age by decade", "BMI"]
    seed = 42
    
    print("🔬 开始模型比较分析")
    print("=" * 60)
    
    # 3. Sex-Specific模型
    print("\n1️⃣ Sex-Specific模型（男女分别训练）")
    print("-" * 40)
    sex_specific = inference_with_features(
        train_df, test_df, features, "VT/VF/SCD", survival_df, seed, gray_features
    )
    
    # 4. Sex-Agnostic模型（欠采样）
    print("\n2️⃣ Sex-Agnostic模型（使用欠采样）")
    print("-" * 40)
    sex_agnostic_us = sex_agnostic_model_inference(
        train_df, test_df, features, "VT/VF/SCD", survival_df, seed, gray_features, True
    )
    
    # 5. Sex-Agnostic模型（无欠采样）
    print("\n3️⃣ Sex-Agnostic模型（无欠采样）")
    print("-" * 40)
    sex_agnostic_no_us = sex_agnostic_model_inference(
        train_df, test_df, features, "VT/VF/SCD", survival_df, seed, gray_features, False
    )
    
    print("\n✅ 所有模型训练完成！")
    print(f"Sex-specific结果: {len(sex_specific)} 样本")
    print(f"Sex-agnostic(欠采样)结果: {len(sex_agnostic_us)} 样本") 
    print(f"Sex-agnostic(无欠采样)结果: {len(sex_agnostic_no_us)} 样本")
    
    return sex_specific, sex_agnostic_us, sex_agnostic_no_us

# 运行所有模型
# results = run_all_models()
```

## 🎯 主要改进功能

### 1. 特征重要性图颜色定制
- **灰色**：`gray_features`列表中的特征（如人口统计学特征）
- **蓝色**：其他特征（如临床特征）
- 帮助区分不同类型的特征

### 2. Sex-Agnostic模型训练
- 在所有数据上训练单一模型
- 支持欠采样以便与sex-specific模型公平比较
- 使用相同的超参数搜索和阈值优化

### 3. 一致的生存分析
- 所有模型使用相同的生存分析流程
- 相同的KM图布局和统计检验
- 便于直接比较不同建模方法的结果

## 💡 使用建议

1. **开始时**：使用benchmark特征集和默认参数
2. **特征选择**：根据研究目标定制gray_features列表
3. **模型比较**：运行所有三种方法进行全面比较
4. **结果解释**：重点关注生存曲线的分离度和p值
5. **临床应用**：考虑模型的可解释性和实用性

## 🔍 输出文件

每个模型会生成：
- 特征重要性图（matplotlib图表）
- 生存分析图（2x2 KM曲线）
- 控制台统计输出
- 返回的结果数据框（包含预测和生存数据）