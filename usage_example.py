#!/usr/bin/env python3
"""
使用示例：展示如何使用修改后的 sexspecificinference 函数（sex_specific_model_inference）
的 red_features 参数
"""

# 示例用法：
"""
# 定义特征列表
features = [
    "Female",
    "Age by decade", 
    "BMI",
    "AF",
    "Beta Blocker",
    "CrCl>45",
    "LVEF",
    "QTc", 
    "NYHA>2",
    "CRT",
    "AAD",
    "Significant LGE",
    "DM",
    "HTN",
    "HLP"
]

# 定义要显示为灰色的特征
gray_features = ["Female", "Age by decade"]

# 定义要显示为红色的特征（新功能）
red_features = ["LVEF", "NYHA>2", "Significant LGE"]

# 调用 sex_specific_model_inference 函数，现在支持 red_features 参数
result_df = sex_specific_model_inference(
    train_df=train_df,
    test_df=test_df, 
    features=features,
    labels="VT/VF/SCD",  # 或你的目标标签
    survival_df=survival_df,
    seed=42,
    gray_features=gray_features,  # 灰色特征
    red_features=red_features     # 红色特征（新参数）
)

# 特征重要性图表中：
# - red_features 中的特征将显示为红色
# - gray_features 中的特征将显示为灰色  
# - 其他特征将显示为蓝色

# 同样适用于其他函数：
result_df2 = sex_specific_full_inference(
    train_df=train_df,
    test_df=test_df, 
    features=features,
    labels="VT/VF/SCD",
    survival_df=survival_df,
    seed=42,
    gray_features=gray_features,
    red_features=red_features
)

result_df3 = sex_agnostic_model_inference(
    train_df=train_df,
    test_df=test_df, 
    features=features,
    label_col="VT/VF/SCD",
    survival_df=survival_df,
    seed=42,
    gray_features=gray_features,
    red_features=red_features,
    use_undersampling=True
)
"""

print("修改完成！")
print("\n新增功能：")
print("1. 所有相关函数现在都支持 red_features 参数")
print("2. 在特征重要性图表中，red_features 中的特征将显示为红色")
print("3. gray_features 中的特征显示为灰色")
print("4. 其他特征显示为蓝色")
print("\n修改的函数包括：")
print("- sex_specific_model_inference()")
print("- sex_specific_full_inference()")  
print("- sex_agnostic_model_inference()")
print("- plot_feature_importances()")
print("- rf_evaluate()")