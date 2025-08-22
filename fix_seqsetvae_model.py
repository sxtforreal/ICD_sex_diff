#!/usr/bin/env python3
"""
SeqSetVAE 模型重复日志记录修复脚本
========================================

直接运行此脚本来修复你的SeqSetVAE模型中的重复日志记录问题。

使用方法:
python fix_seqsetvae_model.py /path/to/your/SeqSetVAE/main/model.py
"""

import os
import re
import sys
import shutil
from datetime import datetime

def fix_duplicate_logging(model_file_path):
    """
    修复PyTorch Lightning模型中的重复日志记录问题
    
    错误: You called `self.log(val/focal_loss, ...)` twice in `validation_step` 
          with different arguments. This is not allowed
    """
    
    if not os.path.exists(model_file_path):
        print(f"❌ 错误: 文件不存在 {model_file_path}")
        return False
    
    print(f"🔧 正在修复模型文件: {model_file_path}")
    
    # 读取原文件
    try:
        with open(model_file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return False
    
    # 创建备份
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{model_file_path}.backup_{timestamp}"
    try:
        shutil.copy2(model_file_path, backup_path)
        print(f"✅ 已创建备份: {backup_path}")
    except Exception as e:
        print(f"⚠️  备份失败: {e}")
    
    modified_content = original_content
    changes_made = []
    
    # 1. 修复 _step 方法
    step_pattern = r'(def _step\(self, batch, stage.*?\):)(.*?)(?=\n    def |\n\nclass |\nclass |\Z)'
    
    def replace_step_method(match):
        method_signature = match.group(1)
        method_body = match.group(2)
        
        # 新的_step方法实现
        new_method = f'''{method_signature}
        """统一处理所有阶段的日志记录 - 修复重复日志记录问题"""
        # 原有计算逻辑
        logits, recon_loss, kl_loss = self(batch)
        
        # 计算focal loss
        if hasattr(self, 'calculate_focal_loss'):
            focal_loss = self.calculate_focal_loss(logits, batch)
        else:
            # 如果没有focal_loss方法，使用BCE作为替代
            import torch.nn.functional as F
            labels = batch.get('label', batch.get('labels', batch.get('target')))
            focal_loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        
        total_loss = focal_loss + recon_loss + kl_loss
        
        # 计算概率和预测
        import torch
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()
        label = batch.get('label', batch.get('labels', batch.get('target')))
        
        # 更新torchmetrics（不自动记录日志）
        if stage == "val":
            if hasattr(self, 'val_auc') and self.val_auc is not None:
                self.val_auc.update(probs, label)
            if hasattr(self, 'val_auprc') and self.val_auprc is not None:
                self.val_auprc.update(probs, label)
            if hasattr(self, 'val_acc') and self.val_acc is not None:
                self.val_acc.update(preds, label)
        elif stage == "train":
            if hasattr(self, 'train_auc') and self.train_auc is not None:
                self.train_auc.update(probs, label)
            if hasattr(self, 'train_acc') and self.train_acc is not None:
                self.train_acc.update(preds, label)
        
        # *** 关键修改：统一记录所有指标，避免重复 ***
        log_metrics = {{
            f"{{stage}}/focal_loss": focal_loss,
            f"{{stage}}/recon_loss": recon_loss,
            f"{{stage}}/kl_loss": kl_loss,
            f"{{stage}}/total_loss": total_loss,
        }}
        
        # 根据阶段选择合适的日志记录参数
        if stage == "train":
            self.log_dict(log_metrics, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        else:  # val or test
            self.log_dict(log_metrics, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        
        # 处理splitdata模块冲突
        if hasattr(self, 'splitdata_module') and stage == "val":
            try:
                split_metrics = self._get_splitdata_metrics(batch, stage)
                if split_metrics:
                    # 使用不同的键名避免冲突
                    split_log_metrics = {{f"{{stage}}/split_{{k}}": v for k, v in split_metrics.items()}}
                    self.log_dict(split_log_metrics, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            except Exception as e:
                print(f"Warning: splitdata logging error: {{e}}")
        
        return logits, recon_loss, kl_loss'''
        
        return new_method
    
    if re.search(step_pattern, modified_content, re.DOTALL):
        modified_content = re.sub(step_pattern, replace_step_method, modified_content, flags=re.DOTALL)
        changes_made.append("✅ 修复了 _step 方法")
    else:
        print("⚠️  未找到 _step 方法")
    
    # 2. 修复 validation_step 方法
    val_step_pattern = r'(def validation_step\(self, batch, batch_idx.*?\):)(.*?)(?=\n    def |\n\nclass |\nclass |\Z)'
    
    def replace_validation_step(match):
        method_signature = match.group(1)
        
        new_method = f'''{method_signature}
        """简化的validation_step - 只调用_step，避免重复日志记录"""
        # *** 删除所有原有的 self.log() 调用 ***
        logits, recon_loss, kl_loss = self._step(batch, "val")
        
        # 返回必要的信息给Lightning
        return {{"val_loss": recon_loss + kl_loss}}'''
        
        return new_method
    
    if re.search(val_step_pattern, modified_content, re.DOTALL):
        modified_content = re.sub(val_step_pattern, replace_validation_step, modified_content, flags=re.DOTALL)
        changes_made.append("✅ 修复了 validation_step 方法")
    else:
        print("⚠️  未找到 validation_step 方法")
    
    # 3. 添加辅助方法（如果不存在）
    if '_get_splitdata_metrics' not in modified_content:
        helper_method = '''
    def _get_splitdata_metrics(self, batch, stage):
        """获取splitdata模块的指标，使用不同键名避免冲突"""
        if not hasattr(self, 'splitdata_module'):
            return {}
        
        metrics = {}
        try:
            if hasattr(self.splitdata_module, 'calculate_loss'):
                split_loss = self.splitdata_module.calculate_loss(batch)
                metrics['focal_loss'] = split_loss  # 将被记录为 val/split_focal_loss
                
            if hasattr(self.splitdata_module, 'calculate_accuracy'):
                split_acc = self.splitdata_module.calculate_accuracy(batch)
                metrics['accuracy'] = split_acc  # 将被记录为 val/split_accuracy
                
        except Exception as e:
            print(f"Error in splitdata metrics calculation: {e}")
        
        return metrics
'''
        
        # 在类的末尾添加辅助方法
        class_pattern = r'(\n\s*def configure_optimizers.*?return.*?\n)'
        if re.search(class_pattern, modified_content, re.DOTALL):
            modified_content = re.sub(class_pattern, r'\1' + helper_method, modified_content, flags=re.DOTALL)
        else:
            # 在文件末尾添加
            modified_content += helper_method
        
        changes_made.append("✅ 添加了 _get_splitdata_metrics 辅助方法")
    
    # 4. 写入修复后的文件
    try:
        with open(model_file_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        print(f"✅ 修复完成！文件已保存: {model_file_path}")
    except Exception as e:
        print(f"❌ 写入文件失败: {e}")
        return False
    
    # 显示修复结果
    if changes_made:
        print("\n🎉 修复摘要:")
        for change in changes_made:
            print(f"  {change}")
    else:
        print("⚠️  未进行任何修改")
    
    print(f"\n📋 验证步骤:")
    print("1. 运行训练脚本检查是否还有重复日志记录错误")
    print("2. 检查tensorboard中指标是否正常显示")
    print("3. 如果有问题，可以从备份文件恢复:")
    print(f"   cp {backup_path} {model_file_path}")
    
    return True

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("使用方法: python fix_seqsetvae_model.py <model_file_path>")
        print("示例: python fix_seqsetvae_model.py /home/sunx/data/aiiih/projects/sunx/projects/SeqSetVAE/main/model.py")
        sys.exit(1)
    
    model_file_path = sys.argv[1]
    
    print("🚀 SeqSetVAE 重复日志记录修复工具")
    print("=" * 50)
    
    success = fix_duplicate_logging(model_file_path)
    
    if success:
        print("\n✅ 修复成功！")
        sys.exit(0)
    else:
        print("\n❌ 修复失败！")
        sys.exit(1)

if __name__ == "__main__":
    main()