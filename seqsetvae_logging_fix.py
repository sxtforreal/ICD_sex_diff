"""
SeqSetVAE Model Logging Fix
==========================

Specific fix for the recurring validation_step logging error.
Apply these patterns to your model.py file.
"""

import torch
import pytorch_lightning as pl
from typing import Dict, Any, Tuple, Optional

class SeqSetVAELoggingFix:
    """
    Template for fixing the SeqSetVAE logging issues
    """
    
    def _step(self, batch, stage: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Modified _step method that collects metrics instead of logging immediately
        
        Returns:
            logits, recon_loss, kl_loss, metrics_dict
        """
        # Your existing model forward pass
        logits, recon_loss, kl_loss = self(batch)
        
        # Calculate focal loss
        focal_loss = self.calculate_focal_loss(logits, batch)  # Your existing method
        
        # Collect ALL metrics in a dictionary - DO NOT LOG HERE
        metrics = {}
        
        if stage == "val":
            # Calculate probabilities and predictions
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            label = batch.get('label', batch.get('labels', batch.get('target')))
            
            # Update torchmetrics (these don't log automatically)
            if hasattr(self, 'val_auc'):
                self.val_auc.update(probs, label)
            if hasattr(self, 'val_auprc'):
                self.val_auprc.update(probs, label)
            if hasattr(self, 'val_acc'):
                self.val_acc.update(preds, label)
            
            # Collect all loss metrics
            metrics.update({
                f"{stage}/focal_loss": focal_loss,
                f"{stage}/recon_loss": recon_loss,
                f"{stage}/kl_loss": kl_loss,
                f"{stage}/total_loss": focal_loss + recon_loss + kl_loss,
            })
            
            # If splitdata module adds metrics, collect them here too
            if hasattr(self, 'splitdata_metrics'):
                splitdata_metrics = self.get_splitdata_metrics(batch, stage)
                # Use different key names to avoid conflicts
                for key, value in splitdata_metrics.items():
                    metrics[f"{stage}/split_{key}"] = value
        
        return logits, recon_loss, kl_loss, metrics
    
    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """
        Fixed validation_step that logs metrics only once
        """
        # Get metrics from _step
        logits, recon_loss, kl_loss, metrics = self._step(batch, "val")
        
        # CRITICAL: Log all metrics at once with consistent parameters
        if metrics:
            # Use the same logging parameters for all metrics
            self.log_dict(
                metrics,
                prog_bar=True,
                logger=True,
                on_step=False,  # Only log at epoch end for validation
                on_epoch=True,
                sync_dist=True,
                batch_size=batch.get('input', batch).size(0) if hasattr(batch, 'size') else len(batch)
            )
        
        # Return the main loss for Lightning
        return {
            "val_loss": metrics.get("val/total_loss", metrics.get("val/focal_loss", torch.tensor(0.0)))
        }
    
    def on_validation_epoch_end(self) -> None:
        """
        Log computed metrics at the end of validation epoch
        """
        # Compute and log torchmetrics
        computed_metrics = {}
        
        if hasattr(self, 'val_auc') and self.val_auc._update_called:
            computed_metrics["val/auc"] = self.val_auc.compute()
            self.val_auc.reset()
            
        if hasattr(self, 'val_auprc') and self.val_auprc._update_called:
            computed_metrics["val/auprc"] = self.val_auprc.compute()
            self.val_auprc.reset()
            
        if hasattr(self, 'val_acc') and self.val_acc._update_called:
            computed_metrics["val/acc"] = self.val_acc.compute()
            self.val_acc.reset()
        
        # Log computed metrics if any
        if computed_metrics:
            self.log_dict(
                computed_metrics,
                prog_bar=True,
                logger=True,
                sync_dist=True
            )
    
    def get_splitdata_metrics(self, batch, stage: str) -> Dict[str, torch.Tensor]:
        """
        Method to get metrics from splitdata module if it exists
        
        This prevents the splitdata module from logging directly,
        which could cause conflicts.
        """
        metrics = {}
        
        # Example: if your splitdata module calculates additional losses
        if hasattr(self, 'splitdata_module'):
            # Get metrics but don't let it log directly
            split_loss = self.splitdata_module.calculate_loss(batch)
            metrics['loss'] = split_loss
            
            # Add any other splitdata metrics
            # metrics['accuracy'] = self.splitdata_module.calculate_accuracy(batch)
        
        return metrics


# EMERGENCY FIX: If you need a quick fix right now
class EmergencyLoggingFix:
    """
    Quick and dirty fix to prevent duplicate logging
    """
    
    def __init__(self):
        super().__init__()
        self._validation_logged_keys = set()
    
    def safe_validation_log(self, key: str, value: torch.Tensor, **kwargs):
        """
        Only log if the key hasn't been logged in this validation step
        """
        step_key = f"{self.current_epoch}_{self.global_step}_{key}"
        
        if step_key not in self._validation_logged_keys:
            self.log(key, value, **kwargs)
            self._validation_logged_keys.add(step_key)
    
    def on_validation_epoch_start(self):
        """Clear logged keys at start of each validation epoch"""
        if hasattr(self, '_validation_logged_keys'):
            self._validation_logged_keys.clear()
        super().on_validation_epoch_start()


# SPECIFIC PATTERNS TO LOOK FOR AND FIX

def common_problematic_patterns():
    """
    Common patterns that cause duplicate logging in SeqSetVAE models
    """
    
    # PATTERN 1: Logging in both _step and validation_step
    """
    # PROBLEMATIC:
    def _step(self, batch, stage):
        focal_loss = calculate_focal_loss(...)
        self.log(f"{stage}/focal_loss", focal_loss)  # First log
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, "val")
        self.log("val/focal_loss", loss)  # Second log - ERROR!
    """
    
    # PATTERN 2: Conditional logging with different parameters
    """
    # PROBLEMATIC:
    def validation_step(self, batch, batch_idx):
        loss = calculate_loss(...)
        
        if self.current_epoch > 10:
            self.log("val/focal_loss", loss, prog_bar=True)
        else:
            self.log("val/focal_loss", loss, logger=True)  # Different params - ERROR!
    """
    
    # PATTERN 3: Splitdata module logging conflicts
    """
    # PROBLEMATIC:
    def validation_step(self, batch, batch_idx):
        loss = calculate_loss(...)
        self.log("val/focal_loss", loss)
        
        # If splitdata module also logs val/focal_loss internally
        self.splitdata_module.validate(batch)  # This might log val/focal_loss too!
    """


if __name__ == "__main__":
    print("Apply these fixes to your SeqSetVAE model.py file:")
    print("1. Modify _step to return metrics instead of logging")
    print("2. Log all metrics once in validation_step")
    print("3. Use consistent logging parameters")
    print("4. Check splitdata module for conflicting logs")