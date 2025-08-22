"""
SeqSetVAE Model.py 修改示例
===========================

这个文件展示了需要在你的 main/model.py 文件中进行的具体修改。
"""

# ============================================================================
# 修改1: 重构 _step 方法
# ============================================================================

def _step(self, batch, stage):
    """
    修改后的_step方法 - 统一处理所有日志记录
    """
    # 原有的计算逻辑保持不变
    logits, recon_loss, kl_loss = self(batch)
    
    # 计算focal loss
    focal_loss = self.calculate_focal_loss(logits, batch)
    total_loss = focal_loss + recon_loss + kl_loss
    
    # 计算概率和预测（用于指标更新）
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).int()
    label = batch.get('label', batch.get('labels', batch.get('target')))
    
    # 更新torchmetrics（不自动记录日志）
    if stage == "val":
        if hasattr(self, 'val_auc'):
            self.val_auc.update(probs, label)
        if hasattr(self, 'val_auprc'):
            self.val_auprc.update(probs, label)
        if hasattr(self, 'val_acc'):
            self.val_acc.update(preds, label)
    elif stage == "train":
        if hasattr(self, 'train_auc'):
            self.train_auc.update(probs, label)
        if hasattr(self, 'train_acc'):
            self.train_acc.update(preds, label)
    
    # *** 关键修改：统一记录所有损失指标 ***
    log_metrics = {
        f"{stage}/focal_loss": focal_loss,
        f"{stage}/recon_loss": recon_loss,
        f"{stage}/kl_loss": kl_loss,
        f"{stage}/total_loss": total_loss,
    }
    
    # 根据阶段选择合适的日志记录参数
    if stage == "train":
        # 训练阶段：每步和每epoch都记录
        self.log_dict(
            log_metrics,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )
    else:  # val or test
        # 验证/测试阶段：只在epoch结束时记录
        self.log_dict(
            log_metrics,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )
    
    # *** 处理splitdata模块冲突 ***
    if hasattr(self, 'splitdata_module') and stage == "val":
        try:
            # 获取splitdata指标但使用不同的键名
            split_metrics = self._get_splitdata_metrics(batch, stage)
            if split_metrics:
                # 使用 val/split_* 前缀避免冲突
                split_log_metrics = {f"{stage}/split_{k}": v for k, v in split_metrics.items()}
                self.log_dict(
                    split_log_metrics,
                    prog_bar=False,
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True
                )
        except Exception as e:
            print(f"Warning: splitdata logging error: {e}")
    
    return logits, recon_loss, kl_loss

# ============================================================================
# 修改2: 简化 validation_step 方法
# ============================================================================

def validation_step(self, batch, batch_idx):
    """
    修改后的validation_step - 只调用_step，不再额外记录
    """
    # *** 删除所有原有的 self.log() 调用 ***
    # 例如删除这些行：
    # self.log("val/focal_loss", focal_loss, prog_bar=True)
    # self.log("val/recon_loss", recon_loss, logger=True) 
    # self.log_dict({...})
    
    # 只调用_step方法
    logits, recon_loss, kl_loss = self._step(batch, "val")
    
    # 如果需要，可以返回损失信息给Lightning
    return {"val_loss": recon_loss + kl_loss}

# ============================================================================
# 修改3: 添加辅助方法处理splitdata
# ============================================================================

def _get_splitdata_metrics(self, batch, stage):
    """
    获取splitdata模块的指标，使用不同键名避免冲突
    """
    if not hasattr(self, 'splitdata_module'):
        return {}
    
    metrics = {}
    try:
        # 假设splitdata模块有这些方法
        if hasattr(self.splitdata_module, 'calculate_loss'):
            split_loss = self.splitdata_module.calculate_loss(batch)
            metrics['focal_loss'] = split_loss  # 将被记录为 val/split_focal_loss
            
        if hasattr(self.splitdata_module, 'calculate_accuracy'):
            split_acc = self.splitdata_module.calculate_accuracy(batch)
            metrics['accuracy'] = split_acc  # 将被记录为 val/split_accuracy
            
        # 添加其他splitdata指标...
        
    except Exception as e:
        print(f"Error in splitdata metrics calculation: {e}")
    
    return metrics

# ============================================================================
# 修改4: 确保epoch结束时的指标记录（可选优化）
# ============================================================================

def on_validation_epoch_end(self):
    """
    验证epoch结束时记录计算的指标
    """
    computed_metrics = {}
    
    # 计算并记录torchmetrics
    if hasattr(self, 'val_auc') and self.val_auc._update_called:
        computed_metrics["val/auc"] = self.val_auc.compute()
        self.val_auc.reset()
        
    if hasattr(self, 'val_auprc') and self.val_auprc._update_called:
        computed_metrics["val/auprc"] = self.val_auprc.compute()
        self.val_auprc.reset()
        
    if hasattr(self, 'val_acc') and self.val_acc._update_called:
        computed_metrics["val/acc"] = self.val_acc.compute()
        self.val_acc.reset()
    
    # 记录计算的指标
    if computed_metrics:
        self.log_dict(
            computed_metrics,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )

# ============================================================================
# 修改总结
# ============================================================================

"""
主要修改点：

1. **_step方法**：
   - 添加统一的log_metrics字典
   - 根据stage参数选择不同的日志记录参数
   - 处理splitdata模块冲突

2. **validation_step方法**：
   - 删除所有self.log()调用
   - 只调用_step()方法

3. **新增方法**：
   - _get_splitdata_metrics(): 处理splitdata模块指标

4. **参数统一**：
   - 训练阶段: on_step=True, on_epoch=True
   - 验证阶段: on_step=False, on_epoch=True
   - 所有阶段: prog_bar=True, logger=True, sync_dist=True

这样修改后，val/focal_loss只会在_step方法中记录一次，
彻底解决重复日志记录的错误。
"""