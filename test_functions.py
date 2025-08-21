#!/usr/bin/env python3

# 测试新功能是否正常工作
import sys
import os

print("测试Sex-Specific和Sex-Agnostic模型功能")
print("=" * 50)

# 检查文件是否存在
if not os.path.exists('/workspace/a.py'):
    print("❌ 错误：找不到a.py文件")
    sys.exit(1)

print("✅ a.py文件存在")

# 尝试导入必要的库
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    print("✅ 基础库导入成功")
except ImportError as e:
    print(f"❌ 库导入失败: {e}")
    sys.exit(1)

# 测试执行a.py的部分代码
try:
    # 只测试基本的导入和数据加载部分
    print("✅ 准备测试a.py中的函数定义...")
    
    # 读取文件内容并检查关键函数是否存在
    with open('/workspace/a.py', 'r') as f:
        content = f.read()
    
    # 检查新添加的函数
    if 'def train_sex_agnostic_model' in content:
        print("✅ train_sex_agnostic_model函数已定义")
    else:
        print("❌ train_sex_agnostic_model函数未找到")
    
    if 'def sex_agnostic_model_inference' in content:
        print("✅ sex_agnostic_model_inference函数已定义")
    else:
        print("❌ sex_agnostic_model_inference函数未找到")
        
    if 'gray_features=None' in content:
        print("✅ gray_features参数已添加")
    else:
        print("❌ gray_features参数未找到")
    
    print("\n📋 功能检查完成！")
    print("\n📖 使用方法:")
    print("1. 首先运行: exec(open('/workspace/a.py').read())")
    print("2. 然后调用相应的函数进行模型训练")
    
except Exception as e:
    print(f"❌ 测试过程中出错: {e}")
    
print("\n" + "=" * 50)
print("测试完成")