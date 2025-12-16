import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from torch.utils.data import DataLoader
from einops import rearrange
from loss import RMSELoss

class BandSCNetTrainer(nn.Module):
    def __init__(
        self, 
        model,
        optimizer,
        train_ds,
        valid_ds,
        device='cuda',
        autocast_enabled=False,
        autocast_device_type='cuda',
        autocast_dtype=torch.float16,
        batch_size=4,
        num_workers=4,
        save_path = 'output/'
    ):
        super().__init__()

        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

        self.train_ds = train_ds
        self.valid_ds = valid_ds

        self.train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        self.valid_dl = DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        self.criterion = RMSELoss()
        self.best_val_loss = float('inf')

        self.autocast_config = {
            'device_type':autocast_device_type,
            'dtype':autocast_dtype,
            'enabled':autocast_enabled
        }

        self.save_path = save_path
        if os.path.exists(self.save_path)==False:
            os.makedirs(self.save_path)

    def train_one_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        x = batch['mixture'].to(self.device)
        y = batch['stems'].to(self.device)

        with torch.autocast(**self.autocast_config):
            x_hat, y, x_recon, y_orig = self.model(x, y)
            loss = self.criterion(x_hat, y)

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.detach(),
        }

    def valid_one_step(self, batch):
        self.model.eval()
        
        with torch.no_grad():
            x = batch['mixture'].to(self.device)
            y = batch['stems'].to(self.device)

            with torch.autocast(**self.autocast_config):
                x_hat, y, x_recon, y_orig = self.model(x, y)
                loss = self.criterion(x_hat, y)

        return {
            "loss": loss.detach(),
        }
    
    def train(self, num_epochs, log_interval=50, val_interval=1):
        for epoch in range(num_epochs):
            self.model.train()
            train_loss_sum = 0.0
            for step, batch in enumerate(self.train_dl):
                log = self.train_one_step(batch)
                train_loss_sum += log["loss"].item()

                if (step + 1) % log_interval == 0:
                    avg = train_loss_sum / (step + 1)
                    print(f"[Epoch {epoch+1} | Step {step+1}] "
                          f"train_loss={avg:.4f}")

            if (epoch + 1) % val_interval == 0:
                self.model.eval()
                val_loss_sum = 0.0
                with torch.no_grad():
                    for batch in self.valid_dl:
                        log = self.valid_one_step(batch)
                        val_loss_sum += log["loss"].item()
                val_avg = val_loss_sum / max(1, len(self.valid_dl))
                print(f"[Epoch {epoch+1}] val_loss={val_avg:.4f}")

                if val_avg < self.best_val_loss:
                    print(f"Validation loss improved from {self.best_val_loss:.4f} to {val_avg:.4f}. Saving best model...")
                    self.best_val_loss = val_avg
                    
                    save_path = os.path.join(self.save_path, "best_model.pth")
                    torch.save(self.model.state_dict(), save_path)
                
                last_path = os.path.join(self.save_path, "last_model.pth")
                torch.save(self.model.state_dict(), last_path)