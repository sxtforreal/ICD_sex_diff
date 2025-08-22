#!/usr/bin/env python3
"""
PyTorch Lightning 重复日志问题调查脚本
直接运行此脚本来分析你的model.py文件中的重复logging问题
"""

import os
import re
import sys
from pathlib import Path

def find_model_files():
    """查找可能的model.py文件"""
    print("🔍 搜索model.py文件...")
    
    possible_paths = [
        "/home/sunx/data/aiiih/projects/sunx/projects/SeqSetVAE/main/model.py",
        "/data/aiiih/projects/sunx/projects/SeqSetVAE/main/model.py", 
    ]
    
    # 也搜索当前用户目录
    home_dir = os.path.expanduser("~")
    for root, dirs, files in os.walk(home_dir):
        if "model.py" in files and "SeqSetVAE" in root:
            possible_paths.append(os.path.join(root, "model.py"))
        # 限制搜索深度避免太慢
        if root.count(os.sep) - home_dir.count(os.sep) > 5:
            dirs.clear()
    
    found_files = []
    for path in possible_paths:
        if os.path.exists(path):
            found_files.append(path)
            print(f"✅ 找到: {path}")
    
    if not found_files:
        print("❌ 未找到model.py文件")
        print("请手动指定文件路径:")
        manual_path = input("输入model.py的完整路径: ").strip()
        if os.path.exists(manual_path):
            found_files.append(manual_path)
        else:
            print("❌ 指定的文件不存在")
            return []
    
    return found_files

def analyze_logging_calls(file_path):
    """分析文件中的所有logging调用"""
    print(f"\n📊 分析文件: {file_path}")
    print("=" * 80)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ 无法读取文件: {e}")
        return
    
    # 1. 查找所有self.log调用
    log_calls = find_all_log_calls(content)
    
    # 2. 分析focal_loss相关的调用
    focal_loss_calls = [call for call in log_calls if 'focal_loss' in call['content'].lower()]
    
    print(f"📈 总共找到 {len(log_calls)} 个logging调用")
    print(f"🎯 其中 {len(focal_loss_calls)} 个与focal_loss相关")
    
    if focal_loss_calls:
        print("\n🚨 focal_loss相关的logging调用:")
        print("-" * 50)
        for call in focal_loss_calls:
            print(f"第 {call['line']} 行: {call['type']}")
            print(f"   内容: {call['content'][:100]}{'...' if len(call['content']) > 100 else ''}")
            print()
    
    # 3. 分析validation_step方法
    analyze_validation_step(content)
    
    # 4. 检查可能的重复模式
    check_duplicate_patterns(log_calls)
    
    # 5. 检查最近可能的问题代码
    check_problematic_patterns(content)

def find_all_log_calls(content):
    """查找所有的logging调用"""
    log_calls = []
    
    # 匹配self.log和self.log_dict调用
    patterns = [
        (r'self\.log\s*\([^)]*\)', 'self.log'),
        (r'self\.log_dict\s*\([^)]*\)', 'self.log_dict')
    ]
    
    for pattern, call_type in patterns:
        matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
        for match in matches:
            line_num = content[:match.start()].count('\n') + 1
            log_calls.append({
                'line': line_num,
                'type': call_type,
                'content': match.group(0),
                'start_pos': match.start(),
                'end_pos': match.end()
            })
    
    # 按行号排序
    log_calls.sort(key=lambda x: x['line'])
    return log_calls

def analyze_validation_step(content):
    """分析validation_step方法"""
    print("\n🔬 分析validation_step方法:")
    print("-" * 50)
    
    # 查找validation_step方法
    validation_pattern = r'def\s+validation_step\s*\([^)]*\):(.*?)(?=def\s+\w+|class\s+\w+|\Z)'
    match = re.search(validation_pattern, content, re.DOTALL)
    
    if match:
        method_content = match.group(1)
        method_start = match.start()
        
        # 在这个方法中查找logging调用
        log_calls_in_method = []
        log_patterns = [
            r'self\.log\s*\([^)]*\)',
            r'self\.log_dict\s*\([^)]*\)'
        ]
        
        for pattern in log_patterns:
            for log_match in re.finditer(pattern, method_content, re.MULTILINE | re.DOTALL):
                line_in_method = method_content[:log_match.start()].count('\n') + 1
                total_line = content[:method_start].count('\n') + line_in_method + 1
                log_calls_in_method.append({
                    'line': total_line,
                    'content': log_match.group(0)
                })
        
        if log_calls_in_method:
            print(f"在validation_step方法中找到 {len(log_calls_in_method)} 个logging调用:")
            for call in log_calls_in_method:
                print(f"  第 {call['line']} 行: {call['content'][:80]}{'...' if len(call['content']) > 80 else ''}")
        else:
            print("validation_step方法中没有直接的logging调用")
            
        # 检查是否调用了_step方法
        if '_step(' in method_content:
            print("⚠️  validation_step调用了_step方法，需要检查_step中的logging")
    else:
        print("❌ 未找到validation_step方法")

def check_duplicate_patterns(log_calls):
    """检查可能导致重复的模式"""
    print("\n🔍 检查重复模式:")
    print("-" * 50)
    
    # 检查是否有多个log_dict调用
    log_dict_calls = [call for call in log_calls if 'log_dict' in call['type']]
    if len(log_dict_calls) > 1:
        print(f"⚠️  发现 {len(log_dict_calls)} 个log_dict调用，可能包含重复键:")
        for call in log_dict_calls:
            print(f"  第 {call['line']} 行")
    
    # 检查是否混合使用log和log_dict
    log_calls_simple = [call for call in log_calls if call['type'] == 'self.log']
    if log_calls_simple and log_dict_calls:
        print(f"⚠️  同时使用了self.log ({len(log_calls_simple)}次) 和 self.log_dict ({len(log_dict_calls)}次)")
        print("   这可能导致相同键被记录多次")
    
    # 检查行号接近的调用
    for i in range(len(log_calls) - 1):
        current = log_calls[i]
        next_call = log_calls[i + 1]
        if next_call['line'] - current['line'] <= 5:  # 5行内的调用
            print(f"⚠️  第 {current['line']} 行和第 {next_call['line']} 行的logging调用很接近")

def check_problematic_patterns(content):
    """检查可能有问题的代码模式"""
    print("\n🚨 检查问题模式:")
    print("-" * 50)
    
    problematic_patterns = [
        (r'self\.log.*focal_loss.*\n.*self\.log.*focal_loss', "同一区域多次记录focal_loss"),
        (r'val_dataloader.*\[.*\]', "多个验证数据加载器"),
        (r'for.*fold.*in', "可能的交叉验证循环"),
        (r'super\(\)\.validation_step', "调用父类validation_step"),
        (r'splitdata|split_data', "数据分割相关代码"),
    ]
    
    found_issues = []
    for pattern, description in problematic_patterns:
        matches = list(re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE))
        if matches:
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                found_issues.append((line_num, description, match.group(0)[:50]))
    
    if found_issues:
        print("发现可能的问题模式:")
        for line, desc, match_text in found_issues:
            print(f"  第 {line} 行: {desc}")
            print(f"    代码片段: {match_text}...")
    else:
        print("✅ 未发现明显的问题模式")

def provide_specific_fix_suggestions(file_path):
    """提供具体的修复建议"""
    print(f"\n💡 针对 {file_path} 的修复建议:")
    print("=" * 80)
    
    print("""
1. 立即修复步骤:
   - 搜索所有包含 'val/focal_loss' 的行
   - 确保只在一个地方记录这个指标
   - 检查validation_step和_step方法中的所有logging调用

2. 推荐的修复模式:
   ```python
   def validation_step(self, batch, batch_idx):
       # 计算所有指标
       logits, recon_loss, kl_loss = self(batch)
       focal_loss = self.compute_focal_loss(logits, batch)
       
       # 收集所有指标到一个字典
       val_metrics = {
           'val/focal_loss': focal_loss,
           'val/recon_loss': recon_loss,
           'val/kl_loss': kl_loss,
       }
       
       # 一次性记录所有指标
       self.log_dict(val_metrics, prog_bar=True, sync_dist=True)
       
       return focal_loss
   ```

3. 调试命令:
   ```bash
   grep -n "val/focal_loss" """ + file_path + """
   grep -n "self.log" """ + file_path + """ | head -20
   ```
""")

def main():
    print("🔧 PyTorch Lightning 重复日志问题调查工具")
    print("=" * 80)
    
    # 查找模型文件
    model_files = find_model_files()
    
    if not model_files:
        print("❌ 无法找到模型文件，调查终止")
        return
    
    # 分析每个找到的文件
    for file_path in model_files:
        analyze_logging_calls(file_path)
        provide_specific_fix_suggestions(file_path)
        print("\n" + "="*80 + "\n")
    
    print("🎯 调查完成!")
    print("\n关键修复步骤:")
    print("1. 找到所有记录 'val/focal_loss' 的位置")
    print("2. 合并为单个log_dict调用")  
    print("3. 确保参数一致")
    print("4. 测试修复结果")

if __name__ == "__main__":
    main()