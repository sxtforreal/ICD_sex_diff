#!/usr/bin/env python3
"""
PyTorch Lightning 重复日志问题诊断脚本
用于分析和解决 "You called self.log(val/focal_loss, ...) twice in validation_step with different arguments" 错误
"""

import ast
import re
from typing import List, Dict, Tuple, Set
from pathlib import Path

class LightningLoggingDiagnostic:
    def __init__(self, model_file_path: str):
        self.model_file_path = model_file_path
        self.logging_calls = []
        self.duplicate_keys = set()
        
    def analyze_file(self):
        """分析Python文件中的所有logging调用"""
        print(f"=== 分析文件: {self.model_file_path} ===\n")
        
        try:
            with open(self.model_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"错误: 找不到文件 {self.model_file_path}")
            return
            
        # 1. 查找所有 self.log 和 self.log_dict 调用
        self._find_logging_calls(content)
        
        # 2. 分析重复的键
        self._analyze_duplicate_keys()
        
        # 3. 检查方法中的多次调用
        self._check_method_logging_patterns(content)
        
        # 4. 检查可能的数据分割相关问题
        self._check_data_splitting_issues(content)
        
        # 5. 提供解决方案
        self._provide_solutions()
        
    def _find_logging_calls(self, content: str):
        """查找所有的logging调用"""
        print("1. 查找所有 logging 调用:")
        print("-" * 40)
        
        # 匹配 self.log 和 self.log_dict 调用
        log_pattern = r'self\.log(_dict)?\s*\((.*?)\)'
        matches = re.finditer(log_pattern, content, re.MULTILINE | re.DOTALL)
        
        line_number = 1
        for match in matches:
            # 计算行号
            lines_before = content[:match.start()].count('\n')
            line_num = lines_before + 1
            
            call_type = "log_dict" if match.group(1) else "log"
            args = match.group(2).strip()
            
            self.logging_calls.append({
                'line': line_num,
                'type': call_type,
                'args': args,
                'full_match': match.group(0)
            })
            
            print(f"第 {line_num} 行: self.{call_type}({args[:100]}{'...' if len(args) > 100 else ''})")
        
        print(f"\n总共找到 {len(self.logging_calls)} 个 logging 调用\n")
        
    def _analyze_duplicate_keys(self):
        """分析重复的日志键"""
        print("2. 分析重复的日志键:")
        print("-" * 40)
        
        key_locations = {}
        
        for call in self.logging_calls:
            keys = self._extract_keys_from_call(call)
            for key in keys:
                if key not in key_locations:
                    key_locations[key] = []
                key_locations[key].append(call['line'])
        
        # 找出重复的键
        for key, locations in key_locations.items():
            if len(locations) > 1:
                self.duplicate_keys.add(key)
                print(f"🚨 重复键 '{key}' 出现在第 {locations} 行")
        
        if not self.duplicate_keys:
            print("✅ 未发现明显的重复键")
        print()
        
    def _extract_keys_from_call(self, call: Dict) -> List[str]:
        """从logging调用中提取键"""
        keys = []
        args = call['args']
        
        if call['type'] == 'log':
            # self.log("key", value, ...)
            # 提取第一个参数作为键
            match = re.match(r'["\']([^"\']+)["\']', args.strip())
            if match:
                keys.append(match.group(1))
        else:  # log_dict
            # self.log_dict({...}, ...)
            # 尝试提取字典中的键
            dict_match = re.search(r'\{([^}]+)\}', args)
            if dict_match:
                dict_content = dict_match.group(1)
                # 简单的键提取 - 查找 "key": 或 'key': 模式
                key_matches = re.findall(r'["\']([^"\']+)["\']\s*:', dict_content)
                keys.extend(key_matches)
        
        return keys
        
    def _check_method_logging_patterns(self, content: str):
        """检查方法中的logging模式"""
        print("3. 检查方法中的 logging 模式:")
        print("-" * 40)
        
        # 查找validation_step和_step方法
        method_patterns = [
            r'def\s+validation_step\s*\([^)]*\):(.*?)(?=def|\Z)',
            r'def\s+_step\s*\([^)]*\):(.*?)(?=def|\Z)',
            r'def\s+training_step\s*\([^)]*\):(.*?)(?=def|\Z)'
        ]
        
        for pattern in method_patterns:
            matches = re.finditer(pattern, content, re.DOTALL)
            for match in matches:
                method_content = match.group(1)
                method_name = re.search(r'def\s+(\w+)', match.group(0)).group(1)
                
                # 在这个方法中查找logging调用
                method_log_calls = []
                for call in self.logging_calls:
                    call_pos = content.find(call['full_match'])
                    method_start = match.start()
                    method_end = match.end()
                    
                    if method_start <= call_pos <= method_end:
                        method_log_calls.append(call)
                
                if method_log_calls:
                    print(f"方法 '{method_name}' 中的 logging 调用:")
                    for call in method_log_calls:
                        print(f"  - 第 {call['line']} 行: {call['type']}")
                    print()
        
    def _check_data_splitting_issues(self, content: str):
        """检查数据分割相关的问题"""
        print("4. 检查数据分割相关问题:")
        print("-" * 40)
        
        # 查找可能导致多次validation的模式
        patterns_to_check = [
            (r'split.*data', "数据分割相关代码"),
            (r'multiple.*dataloader', "多个数据加载器"),
            (r'val_dataloader.*return', "validation dataloader返回"),
            (r'StratifiedKFold|KFold', "交叉验证"),
            (r'for.*fold', "fold循环"),
            (r'self\.current_epoch', "epoch相关"),
        ]
        
        found_issues = []
        for pattern, description in patterns_to_check:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                found_issues.append((line_num, description, match.group(0)))
        
        if found_issues:
            print("🔍 发现可能相关的模式:")
            for line, desc, match_text in found_issues:
                print(f"  第 {line} 行: {desc} - '{match_text}'")
        else:
            print("✅ 未发现明显的数据分割相关问题")
        print()
        
    def _provide_solutions(self):
        """提供解决方案"""
        print("5. 解决方案建议:")
        print("=" * 50)
        
        if self.duplicate_keys:
            print("🚨 发现重复键，建议修复方案:")
            print()
            for key in self.duplicate_keys:
                print(f"对于重复键 '{key}':")
                print("  方案1: 合并所有相同键的logging到单个log_dict调用")
                print("  方案2: 确保每个键只在一个地方被记录")
                print("  方案3: 使用不同的键名区分不同的用途")
                print()
        
        print("通用解决方案:")
        print("1. 检查validation_step方法中的所有self.log和self.log_dict调用")
        print("2. 确保每个metric键只被记录一次")
        print("3. 使用一致的logging参数 (sync_dist, prog_bar等)")
        print("4. 将所有metrics收集到单个字典中，然后使用一次log_dict")
        print()
        
        print("推荐的代码模式:")
        print("""
def validation_step(self, batch, batch_idx):
    # 模型前向传播
    logits, recon_loss, kl_loss = self(batch)
    
    # 计算所有指标
    focal_loss = self.focal_loss(logits, labels)
    
    # 收集所有要记录的指标
    metrics = {
        "val/focal_loss": focal_loss,
        "val/recon_loss": recon_loss, 
        "val/kl_loss": kl_loss,
        # 其他指标...
    }
    
    # 单次记录所有指标
    self.log_dict(metrics, prog_bar=True, sync_dist=True, on_epoch=True)
    
    return focal_loss
        """)

def main():
    """主函数 - 使用示例"""
    print("PyTorch Lightning 重复日志问题诊断工具")
    print("=" * 60)
    print()
    
    # 这里需要替换为实际的model.py路径
    model_file_path = "/home/sunx/data/aiiih/projects/sunx/projects/SeqSetVAE/main/model.py"
    
    diagnostic = LightningLoggingDiagnostic(model_file_path)
    diagnostic.analyze_file()
    
    print("\n" + "=" * 60)
    print("诊断完成!")
    print("如果问题仍然存在，请检查:")
    print("1. 是否有多个进程/GPU在并行训练")
    print("2. 是否有自定义的logging回调函数")
    print("3. 是否在父类中也有相同的logging调用")

if __name__ == "__main__":
    main()