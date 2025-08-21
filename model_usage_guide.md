# Sex-Specific vs Sex-Agnostic模型使用指南

## 快速开始

### 方法1: 运行快速开始脚本
```bash
cd /workspace
python quick_start.py
```

### 方法2: 运行完整示例
```bash
cd /workspace
python run_models.py
```

## 详细使用说明

### 1. 准备数据
确保已经运行了`a.py`来加载和预处理数据：
```python
exec(open('/workspace/a.py').read())
```
这会加载：
- `train_df`: 训练数据
- `test_df`: 测试数据  
- `survival_df`: 生存数据
- 所有必要的函数

### 2. 定义特征

#### 可用的预定义特征集：
```python
# 基准特征集（推荐开始使用）
benchmark_features = [
    "Female", "Age by decade", "BMI", "AF", "Beta Blocker", "CrCl>45", 
    "LVEF", "QTc", "NYHA>2", "CRT", "AAD", "Significant LGE"
]

# 完整提议特征集
proposed_features = [
    "Female", "Age by decade", "BMI", "AF", "Beta Blocker", "CrCl>45", 
    "LVEF", "QTc", "NYHA>2", "CRT", "AAD", "Significant LGE", "DM", "HTN", 
    "HLP", "LVEDVi", "LV Mass Index", "RVEDVi", "RVEF", "LA EF", "LAVi", 
    "MRF (%)", "Sphericity Index", "Relative Wall Thickness", 
    "MV Annular Diameter", "ACEi/ARB/ARNi", "Aldosterone Antagonist"
]

# 指南特征集（最简单）
guideline_features = ["NYHA Class", "LVEF"]
```

#### 自定义特征颜色：
```python
# 定义在特征重要性图中要标为灰色的特征
gray_features = ["Female", "Age by decade", "BMI"]  # 人口统计学特征
# 其他特征将显示为蓝色（临床特征）
```

### 3. 运行Sex-Specific模型

```python
# 为男性和女性分别训练模型
sex_specific_result = inference_with_features(
    train_df=train_df,
    test_df=test_df,
    features=benchmark_features,  # 或使用其他特征集
    labels="VT/VF/SCD",
    survival_df=survival_df,
    seed=42,
    gray_features=gray_features  # 可选：自定义特征颜色
)
```

**输出：**
- 男性模型特征重要性图
- 女性模型特征重要性图
- 按性别和风险的生存分析（4组）
- Kaplan-Meier曲线
- Log-rank检验结果

### 4. 运行Sex-Agnostic模型

```python
# 在所有数据上训练单一模型
sex_agnostic_result = sex_agnostic_model_inference(
    train_df=train_df,
    test_df=test_df,
    features=benchmark_features,
    label_col="VT/VF/SCD",
    survival_df=survival_df,
    seed=42,
    gray_features=gray_features,      # 可选：自定义特征颜色
    use_undersampling=True           # 推荐：使用欠采样以便公平比较
)
```

**输出：**
- Sex-agnostic模型特征重要性图
- 按性别和风险的生存分析（4组）
- Kaplan-Meier曲线
- Log-rank检验结果

### 5. 比较不同方法

```python
# 比较三种方法
features = benchmark_features
gray_features = ["Female", "Age by decade", "BMI"]

# 方法1：Sex-specific（推荐用于主要分析）
result1 = inference_with_features(
    train_df, test_df, features, "VT/VF/SCD", survival_df, 42, gray_features
)

# 方法2：Sex-agnostic with undersampling（公平比较）
result2 = sex_agnostic_model_inference(
    train_df, test_df, features, "VT/VF/SCD", survival_df, 42, gray_features, True
)

# 方法3：Sex-agnostic without undersampling（标准训练）
result3 = sex_agnostic_model_inference(
    train_df, test_df, features, "VT/VF/SCD", survival_df, 42, gray_features, False
)
```

## 参数说明

### 通用参数：
- `train_df`: 训练数据框
- `test_df`: 测试数据框
- `features`: 特征列表
- `survival_df`: 生存数据
- `seed`: 随机种子（建议42）
- `gray_features`: 要在重要性图中标为灰色的特征列表

### Sex-agnostic特有参数：
- `label_col`: 目标变量列名（"VT/VF/SCD"）
- `use_undersampling`: 是否使用欠采样（True推荐用于公平比较）

## 输出解释

### 每个模型都会生成：

1. **特征重要性图**
   - 灰色：gray_features中指定的特征
   - 蓝色：其他特征
   - 按重要性降序排列

2. **生存分析统计**
   - 4个组的发病率：Male-Low, Male-High, Female-Low, Female-High
   - 主要终点(PE)和次要终点(SE)的事件率

3. **Kaplan-Meier生存曲线**
   - 2x2布局：PE/SE × Male/Female
   - 每个子图显示低风险vs高风险
   - 包含Log-rank检验p值

4. **控制台输出**
   - 模型超参数
   - 样本统计
   - 发病率分析

## 建议工作流程

1. **开始**: 使用`quick_start.py`快速测试
2. **探索**: 尝试不同的特征集
3. **比较**: 运行sex-specific和sex-agnostic模型
4. **分析**: 比较特征重要性和生存曲线
5. **报告**: 使用生成的图表和统计结果

## 故障排除

如果遇到错误：
1. 确保已运行`a.py`加载所有数据和函数
2. 检查特征名称是否在数据中存在
3. 确保survival_df包含必要的列（MRN, PE, SE, PE_Time, SE_Time）
4. 验证train_df和test_df包含所有指定的特征