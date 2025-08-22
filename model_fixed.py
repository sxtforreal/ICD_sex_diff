import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import Accuracy, AUROC, AveragePrecision
from typing import Any, Dict, Optional, Tuple


class SeqSetVAE(L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_classes: int = 2,
        learning_rate: float = 1e-3,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        kl_weight: float = 1.0,
        is_pretraining: bool = True,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Classification head (for fine-tuning)
        self.classifier = nn.Linear(latent_dim, num_classes)
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.kl_weight = kl_weight
        self.is_pretraining = is_pretraining
        
        # Metrics - 只初始化训练时需要的metrics
        self.train_auc = AUROC(task="binary", num_classes=2)
        self.train_auprc = AveragePrecision(task="binary", num_classes=2)
        self.train_acc = Accuracy(task="binary", num_classes=2)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播"""
        # Encoding
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        
        # Reparameterization
        z = self.reparameterize(mu, logvar)
        
        # Decoding (reconstruction)
        reconstructed = self.decoder(z)
        
        # Classification (if fine-tuning)
        if not self.is_pretraining:
            logits = self.classifier(z)
        else:
            logits = None
        
        # Compute losses
        recon_loss = F.mse_loss(reconstructed, x, reduction='mean')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        return logits, recon_loss, kl_loss
    
    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal Loss计算"""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def step_(self, batch: Dict[str, torch.Tensor], stage: str) -> Dict[str, torch.Tensor]:
        """统一的step函数，处理训练和验证"""
        x = batch['input']
        
        # 前向传播
        logits, recon_loss, kl_loss = self(x)
        
        # 计算总损失
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        # 准备logging字典
        log_dict = {
            f'{stage}/recon_loss': recon_loss,
            f'{stage}/kl_loss': kl_loss,
            f'{stage}/total_loss': total_loss,
        }
        
        # 如果是微调阶段，添加分类相关的损失和指标
        if not self.is_pretraining and logits is not None:
            label = batch['label']
            
            # 分类损失
            focal_loss = self.focal_loss(logits, label)
            total_loss = total_loss + focal_loss
            
            # 更新logging字典
            log_dict.update({
                f'{stage}/focal_loss': focal_loss,
                f'{stage}/total_loss': total_loss,  # 更新总损失
            })
            
            # 只在训练阶段计算和记录指标，避免validation阶段的重复logging问题
            if stage == 'train':
                # 计算预测概率和预测类别
                probs = F.softmax(logits, dim=1)[:, 1]  # 获取正类概率
                preds = torch.argmax(logits, dim=1)
                
                # 更新metrics
                self.train_auc.update(probs, label)
                self.train_auprc.update(probs, label)
                self.train_acc.update(preds, label)
                
                # 添加metrics到logging字典
                log_dict.update({
                    f'{stage}/auc': self.train_auc,
                    f'{stage}/auprc': self.train_auprc,
                    f'{stage}/acc': self.train_acc,
                })
        
        # 记录所有指标
        self.log_dict(log_dict, on_step=(stage=='train'), on_epoch=True, prog_bar=True, sync_dist=True)
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'logits': logits,
        }
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """训练步骤"""
        return self.step_(batch, "train")
    
    def on_train_epoch_end(self) -> None:
        """训练epoch结束时的处理"""
        if not self.is_pretraining:
            # 重置metrics
            self.train_auc.reset()
            self.train_auprc.reset()
            self.train_acc.reset()
    
    def configure_optimizers(self):
        """配置优化器"""
        if self.is_pretraining:
            # 预训练阶段：只优化encoder和decoder
            params = list(self.encoder.parameters()) + \
                    list(self.mu_layer.parameters()) + \
                    list(self.logvar_layer.parameters()) + \
                    list(self.decoder.parameters())
        else:
            # 微调阶段：优化所有参数
            params = self.parameters()
        
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        
        # 可选：添加学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/total_loss",
            },
        }


# 辅助函数：切换模式
def switch_to_finetuning(model: SeqSetVAE) -> SeqSetVAE:
    """将预训练模型切换到微调模式"""
    model.is_pretraining = False
    model.hparams.is_pretraining = False
    
    # 可选：冻结encoder参数
    # for param in model.encoder.parameters():
    #     param.requires_grad = False
    # for param in model.mu_layer.parameters():
    #     param.requires_grad = False
    # for param in model.logvar_layer.parameters():
    #     param.requires_grad = False
    
    return model


def switch_to_pretraining(model: SeqSetVAE) -> SeqSetVAE:
    """将模型切换到预训练模式"""
    model.is_pretraining = True
    model.hparams.is_pretraining = True
    
    # 解冻所有参数
    for param in model.parameters():
        param.requires_grad = True
    
    return model