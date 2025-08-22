"""
统一 Step 方法解决方案
=====================

这个方案通过统一的 _step 方法处理所有阶段的日志记录，
避免在 validation_step 中重复记录指标。
"""

import torch
import pytorch_lightning as pl
from typing import Dict, Any, Tuple, Optional

class UnifiedStepSeqSetVAE(pl.LightningModule):
    """
    使用统一step方法的SeqSetVAE模型，避免重复日志记录
    """
    
    def __init__(self):
        super().__init__()
        # 你的模型组件
        self.encoder = torch.nn.Linear(100, 50)
        self.decoder = torch.nn.Linear(50, 100)
        
        # 初始化指标
        from torchmetrics import AUROC, AveragePrecision, Accuracy
        self.val_auc = AUROC(task="binary")
        self.val_auprc = AveragePrecision(task="binary")
        self.val_acc = Accuracy(task="binary")
        
        self.train_auc = AUROC(task="binary")
        self.train_acc = Accuracy(task="binary")
    
    def _step(self, batch, stage: str, batch_idx: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        统一的step方法，处理train/val/test所有阶段
        
        Args:
            batch: 输入批次数据
            stage: 阶段标识 ("train", "val", "test")
            batch_idx: 批次索引（可选）
            
        Returns:
            包含损失值的字典，供Lightning使用
        """
        # 1. 模型前向传播
        logits, recon_loss, kl_loss = self(batch)
        
        # 2. 计算focal loss
        focal_loss = self.calculate_focal_loss(logits, batch)
        total_loss = focal_loss + recon_loss + kl_loss
        
        # 3. 计算概率和预测
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()
        label = batch.get('label', batch.get('labels', batch.get('target')))
        
        # 4. 准备日志指标字典
        log_metrics = {
            f"{stage}/focal_loss": focal_loss,
            f"{stage}/recon_loss": recon_loss,
            f"{stage}/kl_loss": kl_loss,
            f"{stage}/total_loss": total_loss,
        }
        
        # 5. 根据不同阶段处理指标和日志记录
        if stage == "train":
            # 训练阶段：更新指标并记录
            self.train_auc.update(probs, label)
            self.train_acc.update(preds, label)
            
            # 训练阶段可以每步记录
            self.log_dict(
                log_metrics,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
                sync_dist=True
            )
            
        elif stage == "val":
            # 验证阶段：更新指标，只在epoch结束时记录
            self.val_auc.update(probs, label)
            self.val_auprc.update(probs, label)
            self.val_acc.update(preds, label)
            
            # 验证阶段只在epoch结束时记录，避免重复
            self.log_dict(
                log_metrics,
                prog_bar=True,
                logger=True,
                on_step=False,  # 不在每步记录
                on_epoch=True,  # 只在epoch结束记录
                sync_dist=True
            )
            
        elif stage == "test":
            # 测试阶段：类似验证阶段
            self.log_dict(
                log_metrics,
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True
            )
        
        # 6. 处理splitdata模块（如果存在）
        if hasattr(self, 'splitdata_module') and stage == "val":
            splitdata_metrics = self._get_splitdata_metrics(batch, stage)
            if splitdata_metrics:
                # 使用不同的键名避免冲突
                split_log_metrics = {f"{stage}/split_{k}": v for k, v in splitdata_metrics.items()}
                self.log_dict(
                    split_log_metrics,
                    prog_bar=False,
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True
                )
        
        # 7. 返回主要损失给Lightning
        return {"loss": total_loss}
    
    def training_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """训练步骤：直接调用统一的_step方法"""
        return self._step(batch, "train", batch_idx)
    
    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """
        验证步骤：只调用_step，不再额外记录任何指标
        这样就避免了重复记录的问题
        """
        return self._step(batch, "val", batch_idx)
    
    def test_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """测试步骤：直接调用统一的_step方法"""
        return self._step(batch, "test", batch_idx)
    
    def on_train_epoch_end(self) -> None:
        """训练epoch结束时记录计算的指标"""
        if self.train_auc._update_called:
            train_auc = self.train_auc.compute()
            self.log("train/auc", train_auc, prog_bar=True, logger=True, sync_dist=True)
            self.train_auc.reset()
            
        if self.train_acc._update_called:
            train_acc = self.train_acc.compute()
            self.log("train/acc", train_acc, prog_bar=True, logger=True, sync_dist=True)
            self.train_acc.reset()
    
    def on_validation_epoch_end(self) -> None:
        """验证epoch结束时记录计算的指标"""
        computed_metrics = {}
        
        if self.val_auc._update_called:
            computed_metrics["val/auc"] = self.val_auc.compute()
            self.val_auc.reset()
            
        if self.val_auprc._update_called:
            computed_metrics["val/auprc"] = self.val_auprc.compute()
            self.val_auprc.reset()
            
        if self.val_acc._update_called:
            computed_metrics["val/acc"] = self.val_acc.compute()
            self.val_acc.reset()
        
        if computed_metrics:
            self.log_dict(computed_metrics, prog_bar=True, logger=True, sync_dist=True)
    
    def _get_splitdata_metrics(self, batch, stage: str) -> Dict[str, torch.Tensor]:
        """
        获取splitdata模块的指标（如果存在）
        使用不同的键名避免与主要指标冲突
        """
        if not hasattr(self, 'splitdata_module'):
            return {}
        
        # 让splitdata模块计算指标但不记录日志
        metrics = {}
        try:
            # 假设splitdata模块有这些方法
            if hasattr(self.splitdata_module, 'calculate_loss'):
                split_loss = self.splitdata_module.calculate_loss(batch)
                metrics['focal_loss'] = split_loss  # 注意：这里用的是不同的键名
                
            if hasattr(self.splitdata_module, 'calculate_accuracy'):
                split_acc = self.splitdata_module.calculate_accuracy(batch)
                metrics['accuracy'] = split_acc
                
        except Exception as e:
            # 如果splitdata模块有问题，不影响主要训练
            print(f"Warning: splitdata module error: {e}")
        
        return metrics
    
    def calculate_focal_loss(self, logits, batch):
        """计算focal loss"""
        # 你的focal loss实现
        labels = batch.get('label', batch.get('labels', batch.get('target')))
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels.float()
        )
    
    def forward(self, batch):
        """前向传播"""
        # 你的前向传播实现
        x = batch.get('input', batch)
        if isinstance(x, dict):
            x = x['input']
            
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        # 示例输出
        logits = torch.randn(x.size(0), 1)
        recon_loss = torch.nn.functional.mse_loss(decoded, x)
        kl_loss = torch.tensor(0.1)
        
        return logits, recon_loss, kl_loss


# 针对现有代码的快速修改方案
class QuickFixTemplate:
    """
    如果你不想大幅修改现有代码，这里是快速修复模板
    """
    
    def validation_step(self, batch, batch_idx):
        """
        修改后的validation_step：只调用_step，不额外记录
        """
        # 原来的代码：
        # logits, recon_loss, kl_loss = self._step(batch, "val")
        # self.log("val/focal_loss", some_loss)  # 删除这行！
        
        # 修改后的代码：
        result = self._step(batch, "val")
        
        # 如果你需要返回特定的loss给Lightning
        return result  # _step已经处理了所有日志记录
    
    def _step(self, batch, stage):
        """
        修改后的_step：处理所有日志记录
        """
        # 你的原有计算逻辑
        logits, recon_loss, kl_loss = self(batch)
        focal_loss = self.calculate_focal_loss(logits, batch)
        
        # 计算指标
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int()
        label = batch.get('label', batch.get('labels'))
        
        # 更新torchmetrics
        if stage == "val":
            self.val_auc.update(probs, label)
            self.val_auprc.update(probs, label)
            self.val_acc.update(preds, label)
        
        # 统一记录所有指标 - 只在这里记录一次！
        log_metrics = {
            f"{stage}/focal_loss": focal_loss,
            f"{stage}/recon_loss": recon_loss,
            f"{stage}/kl_loss": kl_loss,
            f"{stage}/total_loss": focal_loss + recon_loss + kl_loss,
        }
        
        # 根据阶段选择日志记录参数
        if stage == "train":
            self.log_dict(log_metrics, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        else:  # val or test
            self.log_dict(log_metrics, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        return logits, recon_loss, kl_loss


if __name__ == "__main__":
    print("统一Step方法解决方案")
    print("=" * 40)
    print("主要改动：")
    print("1. 在_step方法中统一处理所有日志记录")
    print("2. validation_step只调用_step，不再额外记录")
    print("3. 根据stage参数调整日志记录行为")
    print("4. splitdata模块使用不同键名避免冲突")
    print()
    print("这样就彻底避免了重复记录val/focal_loss的问题！")