# 运行sex-specific和sex-agnostic模型的示例代码
# 这个脚本展示如何使用a.py中的新功能

# 首先需要运行a.py来加载数据和函数
exec(open('/workspace/a.py').read())

print("=" * 80)
print("开始训练和比较Sex-Specific vs Sex-Agnostic模型")
print("=" * 80)

# 1. 定义特征集
# 使用benchmark特征集作为示例
features = [
    "Female", "Age by decade", "BMI", "AF", "Beta Blocker", "CrCl>45", 
    "LVEF", "QTc", "NYHA>2", "CRT", "AAD", "Significant LGE"
]

# 也可以使用更完整的proposed特征集
# features = [
#     "Female", "Age by decade", "BMI", "AF", "Beta Blocker", "CrCl>45", 
#     "LVEF", "QTc", "NYHA>2", "CRT", "AAD", "Significant LGE", "DM", "HTN", 
#     "HLP", "LVEDVi", "LV Mass Index", "RVEDVi", "RVEF", "LA EF", "LAVi", 
#     "MRF (%)", "Sphericity Index", "Relative Wall Thickness", 
#     "MV Annular Diameter", "ACEi/ARB/ARNi", "Aldosterone Antagonist"
# ]

print(f"使用特征数量: {len(features)}")
print(f"特征列表: {features}")

# 2. 定义要在feature importance图中标为灰色的特征
# 例如：人口统计学特征标为灰色，临床特征保持蓝色
gray_features = ["Female", "Age by decade", "BMI"]
print(f"将标为灰色的特征: {gray_features}")

# 3. 设置随机种子
seed = 42

print("\n" + "=" * 50)
print("方法1: 运行Sex-Specific模型")
print("=" * 50)

# 运行sex-specific模型（男女分别训练）
sex_specific_result = inference_with_features(
    train_df=train_df,
    test_df=test_df,
    features=features,
    labels="VT/VF/SCD",
    survival_df=survival_df,
    seed=seed,
    gray_features=gray_features
)

print("\n" + "=" * 50)
print("方法2: 运行Sex-Agnostic模型（使用欠采样）")
print("=" * 50)

# 运行sex-agnostic模型（单一模型，使用欠采样）
sex_agnostic_result = sex_agnostic_model_inference(
    train_df=train_df,
    test_df=test_df,
    features=features,
    label_col="VT/VF/SCD",
    survival_df=survival_df,
    seed=seed,
    gray_features=gray_features,
    use_undersampling=True  # 使用欠采样以便公平比较
)

print("\n" + "=" * 50)
print("方法3: 运行Sex-Agnostic模型（不使用欠采样）")
print("=" * 50)

# 可选：运行不使用欠采样的sex-agnostic模型进行比较
sex_agnostic_no_undersample = sex_agnostic_model_inference(
    train_df=train_df,
    test_df=test_df,
    features=features,
    label_col="VT/VF/SCD",
    survival_df=survival_df,
    seed=seed,
    gray_features=gray_features,
    use_undersampling=False  # 不使用欠采样
)

print("\n" + "=" * 80)
print("模型训练和分析完成!")
print("=" * 80)

print("\n总结:")
print("1. Sex-Specific模型: 为男性和女性分别训练独立的模型")
print("2. Sex-Agnostic模型(欠采样): 在所有数据上训练单一模型，使用欠采样平衡数据")
print("3. Sex-Agnostic模型(无欠采样): 在所有数据上训练单一模型，不进行欠采样")
print("\n每个模型都会输出:")
print("- 特征重要性图（自定义颜色：灰色vs蓝色）")
print("- 按性别和风险组的生存分析（4组）")
print("- Kaplan-Meier生存曲线")
print("- Log-rank检验结果")
print("- 发病率分析")

# 保存结果数据框（可选）
print(f"\nSex-specific模型结果数据: {len(sex_specific_result)} 样本")
print(f"Sex-agnostic模型结果数据: {len(sex_agnostic_result)} 样本")
print(f"Sex-agnostic(无欠采样)结果数据: {len(sex_agnostic_no_undersample)} 样本")