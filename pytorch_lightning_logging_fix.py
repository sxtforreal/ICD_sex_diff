"""
PyTorch Lightning Duplicate Logging Fix
=======================================

This file demonstrates how to fix the recurring error:
"You called `self.log(val/focal_loss, ...)` twice in `validation_step` with different arguments. This is not allowed"

The issue occurs when the same metric key is logged multiple times with different parameters
in the same step method.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Optional

class SeqSetVAEModel(pl.LightningModule):
    """
    Example of how to fix duplicate logging issues in PyTorch Lightning
    """
    
    def __init__(self):
        super().__init__()
        # Your model components here
        self.encoder = nn.Linear(100, 50)
        self.decoder = nn.Linear(50, 100)
        
        # Initialize metrics
        from torchmetrics import AUROC, AveragePrecision, Accuracy
        self.val_auc = AUROC(task="binary")
        self.val_auprc = AveragePrecision(task="binary")
        self.val_acc = Accuracy(task="binary")
    
    def _step(self, batch, stage: str):
        """
        Common step function for train/val/test
        
        PROBLEM: If this method logs metrics and is called from validation_step,
        and validation_step also logs the same metrics, you get duplicate logging.
        """
        # Your model forward pass
        logits, recon_loss, kl_loss = self(batch)
        
        # Calculate focal loss (example)
        focal_loss = self.calculate_focal_loss(logits, batch['labels'])
        
        # SOLUTION 1: Use a dictionary to collect metrics instead of logging immediately
        metrics = {}
        
        if stage == "val":
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            label = batch['labels']
            
            # Update metrics but don't log yet
            self.val_auc.update(probs, label)
            self.val_auprc.update(probs, label)
            self.val_acc.update(preds, label)
            
            # Collect metrics in dictionary
            metrics.update({
                f"{stage}/focal_loss": focal_loss,
                f"{stage}/recon_loss": recon_loss,
                f"{stage}/kl_loss": kl_loss,
                f"{stage}/total_loss": focal_loss + recon_loss + kl_loss,
            })
        
        return logits, recon_loss, kl_loss, metrics
    
    def validation_step(self, batch, batch_idx):
        """
        SOLUTION: Collect all metrics in _step and log them once here
        """
        logits, recon_loss, kl_loss, metrics = self._step(batch, "val")
        
        # Log all metrics at once - this prevents duplicate logging
        if metrics:
            self.log_dict(metrics, prog_bar=True, logger=True, sync_dist=True)
        
        return {"val_loss": metrics.get("val/total_loss", 0)}
    
    def on_validation_epoch_end(self):
        """
        Log computed metrics at epoch end
        """
        # SOLUTION 2: Log computed metrics here instead of in validation_step
        val_auc = self.val_auc.compute()
        val_auprc = self.val_auprc.compute()
        val_acc = self.val_acc.compute()
        
        # Log with consistent parameters
        self.log_dict({
            "val/auc": val_auc,
            "val/auprc": val_auprc, 
            "val/acc": val_acc,
        }, prog_bar=True, logger=True, sync_dist=True)
        
        # Reset metrics
        self.val_auc.reset()
        self.val_auprc.reset()
        self.val_acc.reset()
    
    def calculate_focal_loss(self, logits, labels):
        """Example focal loss calculation"""
        # Your focal loss implementation
        return nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
    
    def forward(self, batch):
        """Forward pass"""
        # Your forward implementation
        x = batch['input']
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        logits = torch.randn(x.size(0), 1)  # Example
        recon_loss = nn.functional.mse_loss(decoded, x)
        kl_loss = torch.tensor(0.1)  # Example
        
        return logits, recon_loss, kl_loss


# ADDITIONAL SOLUTIONS FOR COMMON SCENARIOS

class FixedLoggingMixin:
    """
    Mixin class to help prevent duplicate logging issues
    """
    
    def __init__(self):
        super().__init__()
        self._logged_metrics = {}
    
    def safe_log(self, key: str, value: Any, **kwargs):
        """
        Safely log a metric, preventing duplicates in the same step
        """
        step_key = f"{self.current_epoch}_{self.global_step}_{key}"
        
        if step_key not in self._logged_metrics:
            self.log(key, value, **kwargs)
            self._logged_metrics[step_key] = True
    
    def clear_logged_metrics(self):
        """Clear the logged metrics cache"""
        self._logged_metrics.clear()
    
    def on_validation_epoch_start(self):
        """Clear cache at start of validation"""
        super().on_validation_epoch_start()
        self.clear_logged_metrics()


# DEBUGGING TIPS

def debug_logging_calls():
    """
    Tips for debugging duplicate logging issues:
    
    1. Search for all self.log() calls in your validation_step and _step methods
    2. Check if the same metric key is used multiple times
    3. Look for conditional branches that might log the same metric
    4. Check if splitdata or other modules are adding extra logging
    5. Use different metric keys for different contexts (e.g., val/focal_loss_1, val/focal_loss_2)
    """
    
    # Example of problematic code:
    """
    def validation_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch)
        self.log("val/focal_loss", loss, prog_bar=True)  # First call
        
        if some_condition:
            adjusted_loss = loss * 0.5
            self.log("val/focal_loss", adjusted_loss, logger=True)  # ERROR: Same key, different args!
    """
    
    # Fixed version:
    """
    def validation_step(self, batch, batch_idx):
        loss = self.calculate_loss(batch)
        
        if some_condition:
            adjusted_loss = loss * 0.5
            self.log("val/focal_loss", adjusted_loss, prog_bar=True, logger=True)  # Same args
        else:
            self.log("val/focal_loss", loss, prog_bar=True, logger=True)  # Same args
    """


if __name__ == "__main__":
    print("This file contains solutions for PyTorch Lightning duplicate logging errors.")
    print("Apply the patterns shown here to fix your SeqSetVAE model.")