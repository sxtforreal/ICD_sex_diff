# LGE性别特异性阈值分析完整框架 (Python版)

## 第一阶段：数据准备与描述性分析

### 1.1 导入必要库
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder
from patsy import dmatrix
from scipy import optimize
from scipy.interpolate import UnivariateSpline
import warnings
warnings.filterwarnings('ignore')

# 竞争风险分析
from lifelines import AalenAdditiveFitter
# 或者使用 pycox 包进行Fine-Gray分析
```

### 1.2 结局定义与时间窗设置
```python
def prepare_endpoint_data(df, T_main=3*365.25):
    """
    准备终点事件数据
    """
    df = df.copy()
    
    # 定义复合结局
    df['composite_endpoint'] = ((df['scd'] == 1) | (df['appropriate_icd'] == 1)).astype(int)
    
    # 计算首次事件时间
    df['time_to_first_event'] = np.minimum(
        df['scd_time'].fillna(np.inf), 
        df['icd_time'].fillna(np.inf)
    )
    df['time_to_first_event'] = np.where(
        df['time_to_first_event'] == np.inf,
        df['followup_time'],
        df['time_to_first_event']
    )
    
    # 截断到主要评估时间窗
    df['time_truncated'] = np.minimum(df['time_to_first_event'], T_main)
    df['event_truncated'] = np.where(
        df['time_to_first_event'] <= T_main,
        df['composite_endpoint'],
        0
    )
    
    # 竞争风险标记
    df['competing_death'] = (df['non_arrhythmic_death'] == 1).astype(int)
    
    return df

# 应用函数
data = prepare_endpoint_data(your_dataframe)
```

### 1.3 基线特征描述
```python
def descriptive_analysis(df):
    """
    描述性分析按性别分层
    """
    # 基本统计
    vars_continuous = ['age', 'ef', 'lge_percent', 'followup_time']
    vars_categorical = ['composite_endpoint', 'competing_death']
    
    print("=== 连续变量（均数±标准差）===")
    for var in vars_continuous:
        female_stats = df[df['sex'] == 'Female'][var].agg(['mean', 'std', 'count'])
        male_stats = df[df['sex'] == 'Male'][var].agg(['mean', 'std', 'count'])
        
        print(f"{var}:")
        print(f"  Female: {female_stats['mean']:.2f}±{female_stats['std']:.2f} (n={female_stats['count']})")
        print(f"  Male: {male_stats['mean']:.2f}±{male_stats['std']:.2f} (n={male_stats['count']})")
    
    # 事件率
    print("\n=== 事件率 ===")
    event_rate = df.groupby('sex')['composite_endpoint'].agg(['sum', 'count', 'mean'])
    print(event_rate)
    
    # LGE分布可视化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x='lge_percent', hue='sex', alpha=0.7)
    plt.title('LGE分布按性别')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='sex', y='lge_percent')
    plt.title('LGE分布箱线图')
    
    plt.tight_layout()
    plt.show()

# 执行描述性分析
descriptive_analysis(data)
```

## 第二阶段：限制性三次样条函数构建

### 2.1 RCS基函数生成
```python
from patsy import dmatrix

def create_rcs_basis(x, knots=None, df=4):
    """
    创建限制性三次样条基函数
    """
    if knots is None:
        # 默认使用分位数作为knots
        quantiles = np.linspace(0.1, 0.9, df-1)
        knots = np.quantile(x.dropna(), quantiles)
    
    # 使用patsy创建RCS基
    x_clean = x.dropna()
    basis_formula = f"cr(x_clean, knots={knots.tolist()}, constraints='center')"
    basis_matrix = dmatrix(basis_formula, {"x_clean": x_clean}, return_type='dataframe')
    
    return basis_matrix, knots

def apply_rcs_transform(df, var_name, knots, include_interaction=True):
    """
    对变量应用RCS变换并创建交互项
    """
    # 创建RCS基函数
    x_values = df[var_name].values
    basis_formula = f"cr({var_name}, knots={knots.tolist()}, constraints='center')"
    rcs_basis = dmatrix(basis_formula, df, return_type='dataframe')
    
    # 重命名列
    rcs_cols = [f"{var_name}_rcs_{i}" for i in range(rcs_basis.shape[1]-1)]  # 排除截距
    rcs_basis.columns = ['Intercept'] + rcs_cols
    rcs_basis = rcs_basis.drop('Intercept', axis=1)
    
    # 添加到原数据框
    result_df = pd.concat([df.reset_index(drop=True), rcs_basis.reset_index(drop=True)], axis=1)
    
    # 创建与性别的交互项
    if include_interaction:
        sex_male = (result_df['sex'] == 'Male').astype(int)
        for col in rcs_cols:
            result_df[f"{col}_x_male"] = result_df[col] * sex_male
    
    return result_df, rcs_cols

# 应用RCS变换
lge_basis, lge_knots = create_rcs_basis(data['lge_percent'], df=4)
data_rcs, rcs_column_names = apply_rcs_transform(data, 'lge_percent', lge_knots)

print(f"RCS knots位置: {lge_knots}")
print(f"生成的RCS列: {rcs_column_names}")
```

### 2.2 Cox回归建模
```python
def fit_cox_model_with_rcs(df, rcs_cols, other_covars=None, interaction=True):
    """
    拟合包含RCS和交互项的Cox模型
    """
    if other_covars is None:
        other_covars = ['age', 'ef']  # 根据你的协变量调整
    
    # 构建协变量列表
    covars = rcs_cols + other_covars + ['sex_male']
    
    if interaction:
        interaction_cols = [col for col in df.columns if '_x_male' in col]
        covars.extend(interaction_cols)
    
    # 准备数据
    df_model = df[['time_truncated', 'event_truncated'] + covars].dropna()
    
    # 转换为lifelines需要的格式
    cph = CoxPHFitter()
    cph.fit(df_model, duration_col='time_truncated', event_col='event_truncated')
    
    return cph, df_model, covars

# 添加性别指示变量
data_rcs['sex_male'] = (data_rcs['sex'] == 'Male').astype(int)

# 拟合主要模型
cox_model, model_data, model_covars = fit_cox_model_with_rcs(
    data_rcs, rcs_column_names, 
    other_covars=['age', 'ef'],  # 根据实际协变量调整
    interaction=True
)

print("Cox模型拟合完成")
print(f"C-index: {cox_model.concordance_index_:.3f}")
print("\n模型系数：")
print(cox_model.summary[['coef', 'p']])
```

## 第三阶段：标准化风险曲线生成

### 3.1 风险预测函数
```python
def predict_standardized_risk(model, original_data, lge_values, sex_value, 
                            time_point, rcs_knots, rcs_cols):
    """
    计算标准化风险预测
    """
    risks = []
    
    for lge_val in lge_values:
        # 为每个个体创建预测数据
        pred_risks_individual = []
        
        for idx, row in original_data.iterrows():
            # 创建预测数据点
            pred_data = row.copy()
            pred_data['lge_percent'] = lge_val
            pred_data['sex'] = sex_value
            pred_data['sex_male'] = 1 if sex_value == 'Male' else 0
            
            # 重新计算RCS基函数
            lge_array = np.array([lge_val])
            basis_formula = f"cr(lge_array, knots={rcs_knots.tolist()}, constraints='center')"
            rcs_values = dmatrix(basis_formula, {"lge_array": lge_array}, return_type='dataframe')
            rcs_values = rcs_values.iloc[0, 1:].values  # 去除截距
            
            # 更新RCS列
            for i, col in enumerate(rcs_cols):
                pred_data[col] = rcs_values[i]
                if f"{col}_x_male" in pred_data:
                    pred_data[f"{col}_x_male"] = rcs_values[i] * pred_data['sex_male']
            
            # 预测风险
            pred_df = pd.DataFrame([pred_data])[model_covars]
            survival_func = model.predict_survival_function(pred_df, times=[time_point])
            risk = 1 - survival_func.iloc[0, 0]
            pred_risks_individual.append(risk)
        
        # 计算标准化风险（平均）
        standardized_risk = np.mean(pred_risks_individual)
        risks.append(standardized_risk)
    
    return np.array(risks)

def generate_risk_curves(model, data, time_point=3*365.25, n_points=50):
    """
    生成男女标准化风险曲线
    """
    # LGE评估网格
    lge_min, lge_max = data['lge_percent'].quantile([0.01, 0.99])
    lge_grid = np.linspace(lge_min, lge_max, n_points)
    
    # 计算女性风险曲线
    print("计算女性风险曲线...")
    female_risks = predict_standardized_risk(
        model, model_data, lge_grid, 'Female', 
        time_point, lge_knots, rcs_column_names
    )
    
    # 计算男性风险曲线
    print("计算男性风险曲线...")
    male_risks = predict_standardized_risk(
        model, model_data, lge_grid, 'Male', 
        time_point, lge_knots, rcs_column_names
    )
    
    return lge_grid, female_risks, male_risks

# 生成风险曲线
lge_grid, female_risk_curve, male_risk_curve = generate_risk_curves(cox_model, data_rcs)

# 可视化风险曲线
plt.figure(figsize=(10, 6))
plt.plot(lge_grid, female_risk_curve * 100, label='Female', linewidth=2, color='red')
plt.plot(lge_grid, male_risk_curve * 100, label='Male', linewidth=2, color='blue')
plt.xlabel('LGE (%)')
plt.ylabel('3年累积风险 (%)')
plt.title('标准化风险曲线按性别分层')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 第四阶段：阈值确定

### 4.1 方案A：基于临床目标风险
```python
def find_risk_based_cutoffs(lge_grid, risk_curve, target_risks=[0.03, 0.10]):
    """
    基于目标风险水平找到LGE切点
    """
    cutoffs = []
    
    for target_risk in target_risks:
        # 找到最接近目标风险的LGE值
        idx = np.argmin(np.abs(risk_curve - target_risk))
        cutoff = lge_grid[idx]
        actual_risk = risk_curve[idx]
        
        cutoffs.append({
            'target_risk': target_risk,
            'cutoff': cutoff,
            'actual_risk': actual_risk
        })
    
    return cutoffs

# 设定目标风险水平（3年风险 <3% 低风险, 3-10% 中风险, ≥10% 高风险）
target_risks = [0.03, 0.10]

# 计算女性阈值
female_cutoffs = find_risk_based_cutoffs(lge_grid, female_risk_curve, target_risks)
male_cutoffs = find_risk_based_cutoffs(lge_grid, male_risk_curve, target_risks)

print("=== 基于风险的阈值 ===")
print("女性阈值:")
for i, cutoff in enumerate(female_cutoffs):
    print(f"  切点{i+1}: LGE {cutoff['cutoff']:.1f}% (目标风险{cutoff['target_risk']*100:.0f}%, 实际风险{cutoff['actual_risk']*100:.1f}%)")

print("男性阈值:")
for i, cutoff in enumerate(male_cutoffs):
    print(f"  切点{i+1}: LGE {cutoff['cutoff']:.1f}% (目标风险{cutoff['target_risk']*100:.0f}%, 实际风险{cutoff['actual_risk']*100:.1f}%)")
```

### 4.2 方案B：数据驱动优化
```python
from lifelines.statistics import logrank_test

def evaluate_stratification(df, lge_cutoffs, sex_group):
    """
    评估风险分层的区分度
    """
    # 筛选特定性别
    df_sex = df[df['sex'] == sex_group].copy()
    
    # 创建风险组
    df_sex['risk_group'] = pd.cut(
        df_sex['lge_percent'], 
        bins=[0] + lge_cutoffs + [100], 
        labels=['Low', 'Intermediate', 'High'],
        right=False
    )
    
    # 计算各组样本量
    group_counts = df_sex['risk_group'].value_counts()
    min_group_size = group_counts.min() / len(df_sex)
    
    # 如果任何组少于10%，返回惩罚分数
    if min_group_size < 0.10:
        return -1000
    
    # 计算log-rank统计量
    groups = df_sex.groupby('risk_group')
    chi_square = 0
    
    for name1, group1 in groups:
        for name2, group2 in groups:
            if name1 < name2:  # 避免重复比较
                try:
                    result = logrank_test(
                        group1['time_truncated'], group2['time_truncated'],
                        group1['event_truncated'], group2['event_truncated']
                    )
                    chi_square += result.test_statistic
                except:
                    return -1000
    
    return chi_square

def optimize_cutoffs_data_driven(df, sex_group, lge_range):
    """
    数据驱动的阈值优化
    """
    def objective(cutoffs):
        if cutoffs[0] >= cutoffs[1]:  # 确保切点顺序
            return -1000
        return -evaluate_stratification(df, cutoffs, sex_group)  # 负号因为要最大化
    
    # 初始猜测
    lge_min, lge_max = lge_range
    initial_guess = [lge_min + (lge_max-lge_min)*0.33, lge_min + (lge_max-lge_min)*0.67]
    
    # 优化
    bounds = [(lge_min, lge_max-1), (lge_min+1, lge_max)]
    
    result = optimize.minimize(
        objective, 
        initial_guess,
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    if result.success:
        return result.x
    else:
        return initial_guess

# 数据驱动优化
lge_range = (data['lge_percent'].quantile(0.05), data['lge_percent'].quantile(0.95))

female_optimal_cutoffs = optimize_cutoffs_data_driven(data, 'Female', lge_range)
male_optimal_cutoffs = optimize_cutoffs_data_driven(data, 'Male', lge_range)

print("\n=== 数据驱动阈值 ===")
print(f"女性最优切点: {female_optimal_cutoffs[0]:.1f}%, {female_optimal_cutoffs[1]:.1f}%")
print(f"男性最优切点: {male_optimal_cutoffs[0]:.1f}%, {male_optimal_cutoffs[1]:.1f}%")
```

## 第五阶段：统计检验与验证

### 5.1 交互作用检验
```python
def test_sex_lge_interaction(model):
    """
    检验LGE与性别的交互作用
    """
    # 获取交互项系数
    interaction_terms = [col for col in model.summary.index if '_x_male' in col]
    
    if len(interaction_terms) == 0:
        print("模型中未找到交互项")
        return None
    
    # 联合Wald检验
    interaction_coefs = model.summary.loc[interaction_terms, 'coef'].values
    interaction_cov = model.variance_matrix_.loc[interaction_terms, interaction_terms].values
    
    # 计算Wald统计量
    wald_stat = np.dot(interaction_coefs, np.dot(np.linalg.inv(interaction_cov), interaction_coefs))
    p_value = 1 - stats.chi2.cdf(wald_stat, df=len(interaction_terms))
    
    print(f"LGE×性别交互作用检验:")
    print(f"Wald χ² = {wald_stat:.3f}, df = {len(interaction_terms)}, p = {p_value:.4f}")
    
    return p_value

# 执行交互检验
from scipy import stats
interaction_p = test_sex_lge_interaction(cox_model)
```

### 5.2 Bootstrap置信区间
```python
def bootstrap_cutoffs(df, model_func, n_bootstrap=1000):
    """
    Bootstrap估计阈值的置信区间
    """
    np.random.seed(42)
    n_samples = len(df)
    
    female_cutoffs_boot = []
    male_cutoffs_boot = []
    
    for i in range(n_bootstrap):
        if i % 100 == 0:
            print(f"Bootstrap进度: {i}/{n_bootstrap}")
        
        # 重抽样
        boot_indices = np.random.choice(n_samples, n_samples, replace=True)
        boot_df = df.iloc[boot_indices].reset_index(drop=True)
        
        try:
            # 重新拟合模型和计算阈值
            boot_model, _, _ = fit_cox_model_with_rcs(boot_df, rcs_column_names)
            boot_lge_grid, boot_female_risks, boot_male_risks = generate_risk_curves(boot_model, boot_df)
            
            # 计算阈值
            boot_female_cutoffs = find_risk_based_cutoffs(boot_lge_grid, boot_female_risks, target_risks)
            boot_male_cutoffs = find_risk_based_cutoffs(boot_lge_grid, boot_male_risks, target_risks)
            
            female_cutoffs_boot.append([c['cutoff'] for c in boot_female_cutoffs])
            male_cutoffs_boot.append([c['cutoff'] for c in boot_male_cutoffs])
            
        except:
            continue
    
    # 计算置信区间
    female_cutoffs_boot = np.array(female_cutoffs_boot)
    male_cutoffs_boot = np.array(male_cutoffs_boot)
    
    female_ci = np.percentile(female_cutoffs_boot, [2.5, 97.5], axis=0)
    male_ci = np.percentile(male_cutoffs_boot, [2.5, 97.5], axis=0)
    
    return female_ci, male_ci, female_cutoffs_boot, male_cutoffs_boot

# 执行Bootstrap（注意：这会比较耗时）
print("开始Bootstrap分析...")
female_ci, male_ci, female_boot, male_boot = bootstrap_cutoffs(data_rcs, fit_cox_model_with_rcs)

print("\n=== Bootstrap 95%置信区间 ===")
for i in range(len(target_risks)):
    print(f"切点{i+1}:")
    print(f"  女性: {female_cutoffs[i]['cutoff']:.1f}% (95% CI: {female_ci[0,i]:.1f}%-{female_ci[1,i]:.1f}%)")
    print(f"  男性: {male_cutoffs[i]['cutoff']:.1f}% (95% CI: {male_ci[0,i]:.1f}%-{male_ci[1,i]:.1f}%)")
    
    # 计算差异及其置信区间
    diff_boot = male_boot[:, i] - female_boot[:, i]
    diff_ci = np.percentile(diff_boot, [2.5, 97.5])
    print(f"  差异(男-女): {male_cutoffs[i]['cutoff'] - female_cutoffs[i]['cutoff']:.1f}% (95% CI: {diff_ci[0]:.1f}%-{diff_ci[1]:.1f}%)")
```

## 第六阶段：模型验证与可视化

### 6.1 风险分层验证
```python
def validate_stratification(df, female_cutoffs_vals, male_cutoffs_vals):
    """
    验证风险分层效果
    """
    results = {}
    
    for sex, cutoffs_vals in [('Female', female_cutoffs_vals), ('Male', male_cutoffs_vals)]:
        df_sex = df[df['sex'] == sex].copy()
        
        # 创建风险组
        df_sex['risk_group'] = pd.cut(
            df_sex['lge_percent'], 
            bins=[0] + cutoffs_vals + [100], 
            labels=['Low', 'Intermediate', 'High'],
            right=False
        )
        
        # 计算各组事件率
        group_stats = df_sex.groupby('risk_group').agg({
            'event_truncated': ['sum', 'count', 'mean'],
            'time_truncated': 'mean'
        }).round(3)
        
        print(f"\n{sex}风险分层结果:")
        print(group_stats)
        
        # KM生存分析
        plt.figure(figsize=(10, 6))
        for group in ['Low', 'Intermediate', 'High']:
            group_data = df_sex[df_sex['risk_group'] == group]
            if len(group_data) > 0:
                kmf = KaplanMeierFitter()
                kmf.fit(group_data['time_truncated'], group_data['event_truncated'], label=f'{group} (n={len(group_data)})')
                kmf.plot_survival_function()
        
        plt.title(f'{sex} - Kaplan-Meier生存曲线')
        plt.ylabel('生存概率')
        plt.xlabel('时间(天)')
        plt.show()
        
        results[sex] = group_stats
    
    return results

# 应用选定的阈值进行验证
female_cutoffs_vals = [c['cutoff'] for c in female_cutoffs]
male_cutoffs_vals = [c['cutoff'] for c in male_cutoffs]

validation_results = validate_stratification(data, female_cutoffs_vals, male_cutoffs_vals)
```

### 6.2 最终可视化
```python
def final_visualization(lge_grid, female_risks, male_risks, 
                       female_cutoffs_vals, male_cutoffs_vals, target_risks):
    """
    最终结果可视化
    """
    plt.figure(figsize=(12, 8))
    
    # 绘制风险曲线
    plt.plot(lge_grid, female_risks * 100, label='Female', linewidth=3, color='red')
    plt.plot(lge_grid, male_risks * 100, label='Male', linewidth=3, color='blue')
    
    # 添加目标风险水平线
    for i, target_risk in enumerate(target_risks):
        plt.axhline(y=target_risk*100, color='gray', linestyle='--', alpha=0.7)
        plt.text(lge_grid[-1]*0.95, target_risk*100+0.5, f'{target_risk*100:.0f}%', 
                ha='right', va='bottom')
    
    # 添加女性阈值线
    for i, cutoff in enumerate(female_cutoffs_vals):
        plt.axvline(x=cutoff, color='red', linestyle=':', alpha=0.8)
        plt.text(cutoff, plt.ylim()[1]*0.9-i*2, f'F: {cutoff:.1f}%', 
                rotation=90, ha='right', va='top', color='red')
    
    # 添加男性阈值线
    for i, cutoff in enumerate(male_cutoffs_vals):
        plt.axvline(x=cutoff, color='blue', linestyle=':', alpha=0.8)
        plt.text(cutoff, plt.ylim()[1]*0.8-i*2, f'M: {cutoff:.1f}%', 
                rotation=90, ha='right', va='top', color='blue')
    
    plt.xlabel('LGE (%)')
    plt.ylabel('3年累积风险 (%)')
    plt.title('LGE性别特异性风险阈值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 生成最终可视化
final_visualization(lge_grid, female_risk_curve, male_risk_curve,
                   female_cutoffs_vals, male_cutoffs_vals, target_risks)
```

## 执行总结

### 完整执行流程
```python
# 1. 数据准备
data = prepare_endpoint_data(your_dataframe)
descriptive_analysis(data)

# 2. RCS建模
data_rcs, rcs_column_names = apply_rcs_transform(data, 'lge_percent', lge_knots)
data_rcs['sex_male'] = (data_rcs['sex'] == 'Male').astype(int)
cox_model, model_data, model_covars = fit_cox_model_with_rcs(data_rcs, rcs_column_names)

# 3. 风险曲线生成
lge_grid, female_risk_curve, male_risk_curve = generate_risk_curves(cox_model, data_rcs)

# 4. 阈值确定
female_cutoffs = find_risk_based_cutoffs(lge_grid, female_risk_curve, [0.03, 0.10])
male_cutoffs = find_risk_based_cutoffs(lge_grid, male_risk_curve, [0.03, 0.10])

# 5. 统计检验
interaction_p = test_sex_lge_interaction(cox_model)
female_ci, male_ci, female_boot, male_boot = bootstrap_cutoffs(data_rcs, fit_cox_model_with_rcs)

# 6. 验证和可视化
validation_results = validate_stratification(data, female_cutoffs_vals, male_cutoffs_vals)
final_visualization(lge_grid, female_risk_curve, male_risk_curve, 
                   female_cutoffs_vals, male_cutoffs_vals, [0.03, 0.10])
```

这个完整框架提供了所有必要的分析步骤。根据您的具体数据情况，可能需要调整某些参数（如协变量列表、时间窗、目标风险水平等）。
