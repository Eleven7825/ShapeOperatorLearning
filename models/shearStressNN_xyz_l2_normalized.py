import torch
import torch.nn as nn
import time
import os
import csv
import pdb


class ShearStressNN(nn.Module):
    def __init__(self, branch_dims=[6, 32, 64], trunk_dims=[3, 16, 32], final_dim=64):
        super(ShearStressNN, self).__init__()
        
        # Branch network for A matrix features
        modules = []
        in_channels = branch_dims[0]
        for h_dim in branch_dims[1:]:
            modules.append(nn.Sequential(
                nn.Linear(in_channels, h_dim),
                nn.LayerNorm(h_dim),
                nn.Tanh()
            ))
            in_channels = h_dim
        self._branch = nn.Sequential(*modules)
        
        # Trunk network for spherical coordinates
        modules = []
        in_channels = trunk_dims[0]
        for h_dim in trunk_dims[1:]:
            modules.append(nn.Sequential(
                nn.Linear(in_channels, h_dim),
                nn.LayerNorm(h_dim),
                nn.Tanh()
            ))
            in_channels = h_dim
        self._trunk = nn.Sequential(*modules)
        
        self.branch_to_final = nn.Linear(branch_dims[-1], final_dim)
        self.trunk_to_final = nn.Linear(trunk_dims[-1], final_dim)
        self._out_layer = nn.Linear(final_dim, 3)
        
        # # Additional layers after element-wise multiplication
        # self._final_layers = nn.Sequential(
        #     nn.Linear(final_dim, final_dim),
        #     nn.LayerNorm(final_dim),
        #     nn.Tanh(),
        #     nn.Linear(final_dim, final_dim // 2),
        #     nn.LayerNorm(final_dim // 2),
        #     nn.Tanh(),
        #     nn.Linear(final_dim // 2, 3)
        # )
    
    def forward(self, A, xyz):
        y_br = self._branch(A)
        y_br = self.branch_to_final(y_br)
        
        y_tr = self._trunk(xyz)
        y_tr = self.trunk_to_final(y_tr)
        
        y_combined = y_br * y_tr
        
        return self._out_layer(y_combined)
    
    def loss(self, A, xyz, values, reg_lambda=1):
        y_pred = self.forward(A, xyz)
        
        # Compute the difference
        diff = y_pred - values
        
        # Compute the norm of the values tensor
        values_norm = torch.norm(values, p=2, dim=1)
        
        # Ensure we don't divide by zero
        eps = 1e-8
        values_norm = torch.clamp(values_norm, min=eps)
        
        # Normalize the L2 loss by the norm of values
        normalized_l2_loss = torch.norm(diff, p=2, dim=1) / values_norm
        #breakpoint()
        return normalized_l2_loss.mean()
    
    def loss_0(self, A, xyz, values, reg_lambda=1):
        y_pred = self.forward(A, xyz)
        #l2_loss =  torch.mean(torch.sqrt(torch.sum((y_pred - values) ** 2, dim=1)))
        l2_loss = torch.mean(torch.norm(y_pred - values, p=2, dim=1))
        return l2_loss

def train_model(model, train_loader, val_loader, n_epochs=100, lr=1e-3, results_dir='results'):
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Set up CSV file for immediate logging
    csv_path = os.path.join(results_dir, 'training_losses.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'time_elapsed_min', 'epoch_time_sec'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Use mixed precision training for efficiency
    scaler = torch.cuda.amp.GradScaler()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    start_time = time.time()
    
    print(f"Training on {device} for {n_epochs} epochs")
    print(f"Results will be saved to {csv_path}")
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        epoch_start_time = time.time()
        
        # Process each batch without progress bar
        for A_batch, xyz_batch, values_batch in train_loader:
            A_batch = A_batch.to(device, non_blocking=True)
            xyz_batch = xyz_batch.to(device, non_blocking=True)
            values_batch = values_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                if epoch< 2:
                    loss = model.module.loss(A_batch, xyz_batch, values_batch)
                else:
                    loss = model.module.loss(A_batch, xyz_batch, values_batch)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_train_loss += loss.item() * len(A_batch)
        
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0.0
        
        with torch.no_grad():
            for A_batch, xyz_batch, values_batch in val_loader:
                A_batch = A_batch.to(device, non_blocking=True)
                xyz_batch = xyz_batch.to(device, non_blocking=True)
                values_batch = values_batch.to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    if epoch< 2:
                        loss = model.module.loss(A_batch, xyz_batch, values_batch)
                    else:
                        loss = model.module.loss(A_batch, xyz_batch, values_batch)
                
                total_val_loss += loss.item() * len(A_batch)
        
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        
        # Calculate timing information
        epoch_time = time.time() - epoch_start_time
        total_time_elapsed = (time.time() - start_time) / 60
        
        # Log to CSV immediately after each epoch
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, total_time_elapsed, epoch_time])
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{n_epochs} - Time: {total_time_elapsed:.2f}m (epoch: {epoch_time:.2f}s) - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(results_dir, 'shear_stress_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'scaler_state_dict': scaler.state_dict(),
            }, model_path)
            print(f"  → New best model saved at {model_path} (val_loss: {best_val_loss:.6f})")
    
    total_time = (time.time() - start_time) / 60
    print(f"\nTraining completed in {total_time:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Results saved to {csv_path}")
    print(f"Best model saved to {os.path.join(results_dir, 'shear_stress_model.pt')}")
    
    return train_losses, val_losses
