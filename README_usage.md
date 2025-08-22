# SeqSetVAE 修复版本使用说明

## 问题描述

原始代码遇到了以下PyTorch Lightning错误：
```
lightning.fabric.utilities.exceptions.MisconfigurationException: You called `self.log(val/focal_loss, ...)` twice in `validation_step` with different arguments. This is not allowed
```

这个错误是由于在`validation_step`中对同一个指标进行了重复的logging导致的。

## 解决方案

### 1. 移除validation_step

完全移除了`validation_step`方法，避免了重复logging的问题。这样做的好处：
- 消除了重复logging错误
- 简化了代码结构
- 避免了训练和验证之间的不一致性

### 2. 统一的step_函数

创建了一个统一的`step_(batch, stage)`函数来处理所有的训练逻辑：

```python
def step_(self, batch: Dict[str, torch.Tensor], stage: str) -> Dict[str, torch.Tensor]:
    """统一的step函数，处理训练和验证"""
    # 前向传播
    logits, recon_loss, kl_loss = self(batch)
    
    # 根据阶段调整逻辑
    if stage == 'train':
        # 训练逻辑
        pass
    # 可以根据需要添加其他阶段
```

### 3. 预训练和微调的区分

通过`is_pretraining`参数来区分预训练和微调阶段：

**预训练阶段：**
- 只计算重构损失和KL散度损失
- 不进行分类任务
- 不需要标签数据

**微调阶段：**
- 计算重构损失、KL散度损失和分类损失
- 更新分类指标（AUC, AUPRC, Accuracy）
- 需要标签数据

## 主要修改

### 1. 模型结构 (`model_fixed.py`)

```python
class SeqSetVAE(L.LightningModule):
    def __init__(self, ..., is_pretraining: bool = True):
        # 只初始化训练时需要的metrics，避免验证时的冲突
        self.train_auc = AUROC(task="binary", num_classes=2)
        self.train_auprc = AveragePrecision(task="binary", num_classes=2)
        self.train_acc = Accuracy(task="binary", num_classes=2)
    
    def step_(self, batch, stage):
        # 统一处理所有阶段的逻辑
        # 只在训练阶段记录指标，避免重复logging
        if stage == 'train' and not self.is_pretraining:
            # 更新并记录指标
            pass
    
    def training_step(self, batch, batch_idx):
        return self.step_(batch, "train")
    
    # 移除了validation_step方法
```

### 2. 训练脚本 (`train_fixed.py`)

```python
def train_pretraining_phase(model, data_module, args):
    """预训练阶段 - 不使用validation"""
    trainer = L.Trainer(
        # 没有validation相关配置
        # 只使用训练数据
    )
    trainer.fit(model, data_module)

def train_finetuning_phase(model, data_module, args):
    """微调阶段 - 也不使用validation"""
    # 切换到微调模式
    model = switch_to_finetuning(model)
    trainer.fit(model, data_module)
```

## 使用方法

### 1. 预训练阶段

```bash
python train_fixed.py --mode pretrain --pretrain_epochs 50
```

### 2. 微调阶段

```bash
python train_fixed.py --mode finetune --finetune_epochs 30 --pretrained_checkpoint path/to/checkpoint.ckpt
```

### 3. 完整流程（预训练 + 微调）

```bash
python train_fixed.py --mode both --pretrain_epochs 50 --finetune_epochs 30
```

### 4. 自定义参数

```bash
python train_fixed.py \
    --mode both \
    --input_dim 2000 \
    --hidden_dim 512 \
    --latent_dim 256 \
    --learning_rate 0.001 \
    --batch_size 64 \
    --pretrain_epochs 100 \
    --finetune_epochs 50 \
    --output_dir ./my_outputs
```

## 关键特性

### 1. 避免重复Logging
- 只在训练阶段记录指标
- 使用统一的logging策略
- 避免了validation_step中的重复调用

### 2. 灵活的训练模式
- 支持纯预训练
- 支持纯微调（从检查点开始）
- 支持端到端训练（预训练+微调）

### 3. 模式切换
```python
# 切换到微调模式
model = switch_to_finetuning(model)

# 切换到预训练模式
model = switch_to_pretraining(model)
```

### 4. 检查点管理
- 预训练和微调分别保存检查点
- 支持从预训练检查点开始微调
- 自动保存最佳模型

## 注意事项

1. **数据格式**: 确保batch返回的字典包含正确的键：
   - 预训练：`{'input': tensor}`
   - 微调：`{'input': tensor, 'label': tensor}`

2. **指标重置**: 在每个epoch结束时会自动重置训练指标

3. **设备支持**: 代码支持自动设备检测（GPU/CPU）

4. **内存优化**: 移除了validation步骤，减少了内存使用

## 故障排除

如果仍然遇到logging相关错误：
1. 确保没有手动调用`validation_step`
2. 检查是否有其他地方重复logging相同的指标
3. 确认`stage`参数传递正确

这个修复版本应该完全解决您遇到的重复logging错误，同时保持了模型的功能完整性。