# 快速开始：运行sex-specific和sex-agnostic模型
# 简化版本，适合快速测试

# 加载a.py中的所有函数和数据
exec(open('/workspace/a.py').read())

# 快速设置
features = ["Female", "Age by decade", "BMI", "AF", "Beta Blocker", "CrCl>45", 
           "LVEF", "QTc", "NYHA>2", "CRT", "AAD", "Significant LGE"]
gray_features = ["Female", "Age by decade", "BMI"]  # 人口统计学特征标为灰色

print("🚀 快速开始：比较Sex-Specific vs Sex-Agnostic模型")
print(f"📊 使用 {len(features)} 个特征")

# 1. Sex-Specific模型（推荐用于主要分析）
print("\n1️⃣ 运行Sex-Specific模型...")
sex_specific_result = inference_with_features(
    train_df, test_df, features, "VT/VF/SCD", survival_df, 42, gray_features
)

# 2. Sex-Agnostic模型（用于比较）
print("\n2️⃣ 运行Sex-Agnostic模型...")
sex_agnostic_result = sex_agnostic_model_inference(
    train_df, test_df, features, "VT/VF/SCD", survival_df, 42, gray_features, True
)

print("\n✅ 完成！两个模型都已训练并生成了生存分析图表。")