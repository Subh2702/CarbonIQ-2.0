import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import wandb  # For experiment tracking

class GNNTrainer:
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10
        )
        
    def train_epoch(self, train_loader):
        """Ek epoch ka training"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch.x, batch.edge_index)
            
            # Loss calculation
            loss = self._calculate_loss(outputs, batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Logging
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        return total_loss / len(train_loader)
    
    def _calculate_loss(self, outputs, batch):
        """Multi-task loss function"""
        
        # Carbon flow prediction loss (MSE)
        if hasattr(batch, 'carbon_flow_targets'):
            flow_loss = F.mse_loss(
                outputs['carbon_flows'].squeeze(), 
                batch.carbon_flow_targets
            )
        else:
            flow_loss = 0
        
        # Supplier classification loss (CrossEntropy)
        if hasattr(batch, 'supplier_labels'):
            class_loss = F.cross_entropy(
                outputs['supplier_classes'], 
                batch.supplier_labels
            )
        else:
            class_loss = 0
        
        # Graph structure preservation loss
        structure_loss = self._graph_contrastive_loss(outputs['node_embeddings'])
        
        # Total weighted loss
        total_loss = (
            0.5 * flow_loss + 
            0.3 * class_loss + 
            0.2 * structure_loss
        )
        
        return total_loss
    
    def _graph_contrastive_loss(self, embeddings):
        """
        Similar suppliers should have similar embeddings
        """
        # Implement contrastive learning for graph structure
        # This encourages similar suppliers to have similar embeddings
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        # Add contrastive loss logic here...
        return torch.tensor(0.0)  # Placeholder
    
    def validate(self, val_loader):
        """Validation loop"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch.x, batch.edge_index)
                loss = self._calculate_loss(outputs, batch)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs):
        """Complete training pipeline"""
        
        # Initialize experiment tracking
        wandb.init(project="carbon-gnn", config=self.config.__dict__)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Logging
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_gnn_model.pth')
                print("Model saved!")
        
        wandb.finish()