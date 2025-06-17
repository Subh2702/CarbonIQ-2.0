import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
import wandb

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
            self.optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-6
        )
        
    def train_epoch(self, train_loader):
        """Ek epoch ka training"""
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch.x, batch.edge_index)
            
            # Loss and accuracy calculation
            loss, accuracy = self._calculate_loss(outputs, batch, is_training=True)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_accuracy += accuracy
            
            # Logging
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
        
        return total_loss / len(train_loader), total_accuracy / len(train_loader)
    
    def _calculate_loss(self, outputs, batch, is_training=True):
        """Multi-task loss function"""
        # Select appropriate masks
        node_mask = batch.train_mask if is_training and hasattr(batch, 'train_mask') else \
                    batch.val_mask if not is_training and hasattr(batch, 'val_mask') else \
                    torch.ones(batch.x.size(0), dtype=torch.bool, device=self.device)
        edge_mask = batch.train_edge_mask if is_training and hasattr(batch, 'train_edge_mask') else \
                    batch.val_edge_mask if not is_training and hasattr(batch, 'val_edge_mask') else \
                    torch.ones(batch.edge_index.size(1), dtype=torch.bool, device=self.device)
        
        # Carbon flow prediction loss (MSE)
        flow_loss = torch.tensor(0.0, device=self.device)
        if 'carbon_flows' in outputs and 'carbon_flow_targets' in batch:
            flow_pred = outputs['carbon_flows'][edge_mask].squeeze()
            flow_target = batch.carbon_flow_targets[edge_mask]
            if flow_pred.numel() > 0:
                flow_loss = F.mse_loss(flow_pred, flow_target)
        
        # Supplier classification loss (CrossEntropy)
        class_loss = torch.tensor(0.0, device=self.device)
        accuracy = 0.0
        if 'supplier_classes' in outputs and 'supplier_labels' in batch:
            class_pred = outputs['supplier_classes'][node_mask]
            class_target = batch.supplier_labels[node_mask]
            if class_pred.numel() > 0:
                class_loss = F.cross_entropy(class_pred, class_target)
                preds = class_pred.argmax(dim=1)
                accuracy = (preds == class_target).float().mean().item()
        
        # Graph structure preservation loss
        structure_loss = self._graph_contrastive_loss(outputs['node_embeddings'], batch)
        
        # Total weighted loss
        total_loss = 0.5 * flow_loss + 0.3 * class_loss + 0.2 * structure_loss
        
        return total_loss, accuracy
    
    def _graph_contrastive_loss(self, embeddings, batch):
        """Contrastive loss for similar suppliers"""
        if not hasattr(batch, 'supplier_labels'):
            return torch.tensor(0.0, device=self.device)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        temperature = 0.5
        num_nodes = embeddings.size(0)
        
        labels = batch.supplier_labels
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        pos_mask.fill_diagonal_(0)  # Exclude self
        neg_mask = 1 - pos_mask - torch.eye(num_nodes, device=self.device)
        
        pos_sim = torch.exp(similarity_matrix / temperature) * pos_mask
        neg_sim = torch.exp(similarity_matrix / temperature) * neg_mask
        pos_sum = pos_sim.sum(dim=1)
        neg_sum = neg_sim.sum(dim=1)
        
        loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-6)).mean()
        return loss
    
    def validate(self, val_loader):
        """Validation loop"""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch.x, batch.edge_index)
                loss, accuracy = self._calculate_loss(outputs, batch, is_training=False)
                total_loss += loss.item()
                total_accuracy += accuracy
        
        return total_loss / len(val_loader), total_accuracy / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs):
        """Complete training pipeline"""
        wandb.init(project="carbon-gnn", config=self.config.__dict__)
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training
            train_loss, train_accuracy = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_accuracy = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Logging
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_gnn_model.pth')
                print("Model saved!")
        
        wandb.finish()