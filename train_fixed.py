import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
from model_fixed import SeqSetVAE, switch_to_finetuning, switch_to_pretraining


class SeqSetDataModule(L.LightningDataModule):
    def __init__(self, 
                 train_data: torch.Tensor,
                 train_labels: torch.Tensor = None,
                 val_data: torch.Tensor = None,
                 val_labels: torch.Tensor = None,
                 batch_size: int = 32,
                 num_workers: int = 4):
        super().__init__()
        self.train_data = train_data
        self.train_labels = train_labels
        self.val_data = val_data
        self.val_labels = val_labels
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage: str):
        if stage == "fit":
            # 训练数据
            if self.train_labels is not None:
                self.train_dataset = TensorDataset(self.train_data, self.train_labels)
            else:
                # 预训练阶段，不需要标签
                self.train_dataset = TensorDataset(self.train_data)
            
            # 验证数据（如果有的话）
            if self.val_data is not None:
                if self.val_labels is not None:
                    self.val_dataset = TensorDataset(self.val_data, self.val_labels)
                else:
                    self.val_dataset = TensorDataset(self.val_data)
            else:
                self.val_dataset = None
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        if self.val_dataset is not None:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                persistent_workers=True if self.num_workers > 0 else False
            )
        return None


def collate_fn_pretraining(batch):
    """预训练阶段的collate函数"""
    if len(batch[0]) == 1:  # 只有输入数据，没有标签
        inputs = torch.stack([item[0] for item in batch])
        return {'input': inputs}
    else:  # 有输入和标签
        inputs = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        return {'input': inputs, 'label': labels}


def collate_fn_finetuning(batch):
    """微调阶段的collate函数"""
    inputs = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return {'input': inputs, 'label': labels}


def train_pretraining_phase(model, data_module, args):
    """预训练阶段"""
    print("开始预训练阶段...")
    
    # 确保模型处于预训练模式
    model = switch_to_pretraining(model)
    
    # 设置回调函数
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "pretraining_checkpoints"),
            filename="pretrain-{epoch:02d}-{train/total_loss:.4f}",
            monitor="train/total_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="train/total_loss",
            mode="min",
            patience=args.patience,
            verbose=True,
        ),
    ]
    
    # 设置logger
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name="pretraining_logs",
        version=None,
    )
    
    # 创建trainer（注意：没有validation）
    trainer = L.Trainer(
        max_epochs=args.pretrain_epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=args.log_every_n_steps,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # 开始训练
    trainer.fit(model, data_module)
    
    return trainer.checkpoint_callback.best_model_path


def train_finetuning_phase(model, data_module, args, pretrained_checkpoint_path=None):
    """微调阶段"""
    print("开始微调阶段...")
    
    # 如果有预训练检查点，加载它
    if pretrained_checkpoint_path and os.path.exists(pretrained_checkpoint_path):
        print(f"从预训练检查点加载模型: {pretrained_checkpoint_path}")
        model = SeqSetVAE.load_from_checkpoint(
            pretrained_checkpoint_path,
            is_pretraining=False  # 切换到微调模式
        )
    else:
        # 直接切换模式
        model = switch_to_finetuning(model)
    
    # 设置回调函数
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "finetuning_checkpoints"),
            filename="finetune-{epoch:02d}-{train/total_loss:.4f}",
            monitor="train/total_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="train/total_loss",
            mode="min",
            patience=args.patience,
            verbose=True,
        ),
    ]
    
    # 设置logger
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name="finetuning_logs",
        version=None,
    )
    
    # 创建trainer（注意：没有validation）
    trainer = L.Trainer(
        max_epochs=args.finetune_epochs,
        accelerator="auto",
        devices="auto",
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=args.log_every_n_steps,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # 开始训练
    trainer.fit(model, data_module)
    
    return trainer.checkpoint_callback.best_model_path


def main():
    parser = argparse.ArgumentParser(description="SeqSetVAE Training Script")
    
    # 数据参数
    parser.add_argument("--input_dim", type=int, default=1000, help="Input dimension")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    
    # 模型参数
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--focal_alpha", type=float, default=1.0, help="Focal loss alpha")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument("--kl_weight", type=float, default=1.0, help="KL divergence weight")
    
    # 训练参数
    parser.add_argument("--pretrain_epochs", type=int, default=50, help="Pretraining epochs")
    parser.add_argument("--finetune_epochs", type=int, default=30, help="Fine-tuning epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--log_every_n_steps", type=int, default=50, help="Log every n steps")
    
    # 训练模式
    parser.add_argument("--mode", type=str, choices=["pretrain", "finetune", "both"], 
                       default="both", help="Training mode")
    parser.add_argument("--pretrained_checkpoint", type=str, default=None,
                       help="Path to pretrained checkpoint for fine-tuning")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建示例数据（实际使用时请替换为您的数据）
    print("创建示例数据...")
    train_data = torch.randn(1000, args.input_dim)  # 1000个样本
    train_labels = torch.randint(0, args.num_classes, (1000,))  # 分类标签
    
    # 初始化模型
    model = SeqSetVAE(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        kl_weight=args.kl_weight,
        is_pretraining=True,
    )
    
    pretrained_checkpoint_path = None
    
    # 预训练阶段
    if args.mode in ["pretrain", "both"]:
        # 预训练数据模块（不需要标签）
        pretrain_data_module = SeqSetDataModule(
            train_data=train_data,
            train_labels=None,  # 预训练不需要标签
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        
        # 设置collate函数
        pretrain_data_module.train_dataloader().collate_fn = collate_fn_pretraining
        
        pretrained_checkpoint_path = train_pretraining_phase(
            model, pretrain_data_module, args
        )
    
    # 微调阶段
    if args.mode in ["finetune", "both"]:
        # 微调数据模块（需要标签）
        finetune_data_module = SeqSetDataModule(
            train_data=train_data,
            train_labels=train_labels,  # 微调需要标签
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        
        # 设置collate函数
        finetune_data_module.train_dataloader().collate_fn = collate_fn_finetuning
        
        # 使用预训练检查点或者用户指定的检查点
        checkpoint_path = pretrained_checkpoint_path or args.pretrained_checkpoint
        
        finetuned_checkpoint_path = train_finetuning_phase(
            model, finetune_data_module, args, checkpoint_path
        )
        
        print(f"微调完成！最佳模型保存在: {finetuned_checkpoint_path}")
    
    print("训练完成！")


if __name__ == "__main__":
    main()