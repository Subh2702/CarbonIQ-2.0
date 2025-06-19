import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

class EnhancedGNNTrainer:
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # automatic lr adjust karne ke liye
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=20,
            T_mult=2,
            eta_min=1e-6
        )
        
        self.class_weights = None
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience = 15
        self.patience_counter = 0
        
        self.accumulation_steps = 4
    
    # imbalence data set ko jhelne ke liye
    def compute_class_weights(self, train_loader):
        all_labels = []
        for batch in train_loader:
            if hasattr(batch, 'supplier_labels'):
                all_labels.extend(batch.supplier_labels.cpu().numpy())
        
        if len(all_labels) > 0:
            unique_classes = np.unique(all_labels)
            weights = compute_class_weight('balanced', classes=unique_classes, y=all_labels)
            self.class_weights = torch.FloatTensor(weights).to(self.device)
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            
            outputs = self.model(batch.x, batch.edge_index, 
                               edge_attr=getattr(batch, 'edge_attr', None))
            
            loss, metrics = self._calculate_enhanced_loss(outputs, batch, is_training=True)
            
            loss = loss / self.accumulation_steps
            
            loss.backward()
            
            # Gradient se sath chedkhad
            if (batch_idx + 1) % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps
            total_accuracy += metrics['accuracy']
            num_batches += 1
            
            #metrics printing
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item() * self.accumulation_steps:.4f}, '
                      f'Accuracy: {metrics["accuracy"]:.4f}')
        
        return total_loss / num_batches, total_accuracy / num_batches
    
    def _calculate_enhanced_loss(self, outputs, batch, is_training=True):
        device = self.device
        
        #nodes and edges chunna for loss calculation
        node_mask = batch.train_mask if is_training and hasattr(batch, 'train_mask') else \
                    batch.val_mask if not is_training and hasattr(batch, 'val_mask') else \
                    torch.ones(batch.x.size(0), dtype=torch.bool, device=device)
        
        edge_mask = batch.train_edge_mask if is_training and hasattr(batch, 'train_edge_mask') else \
                    batch.val_edge_mask if not is_training and hasattr(batch, 'val_edge_mask') else \
                    torch.ones(batch.edge_index.size(1), dtype=torch.bool, device=device)
        
        total_loss = torch.tensor(0.0, device=device)
        metrics = {'accuracy': 0.0, 'flow_mse': 0.0}
        
        #sare loss calculate karra hu aur jf raha hu

        if 'carbon_flows' in outputs and hasattr(batch, 'carbon_flow_targets'):
            flow_pred = outputs['carbon_flows'][edge_mask].squeeze()
            flow_target = batch.carbon_flow_targets[edge_mask]
            if flow_pred.numel() > 0:
                flow_loss = F.smooth_l1_loss(flow_pred, flow_target)  # Huber loss
                total_loss += 0.4 * flow_loss
                metrics['flow_mse'] = F.mse_loss(flow_pred, flow_target).item()
        
        if 'supplier_classes' in outputs and hasattr(batch, 'supplier_labels'):
            class_pred = outputs['supplier_classes'][node_mask]
            class_target = batch.supplier_labels[node_mask]
            if class_pred.numel() > 0:
                class_loss = F.cross_entropy(class_pred, class_target, weight=self.class_weights)
                total_loss += 0.3 * class_loss
                
                preds = class_pred.argmax(dim=1)
                metrics['accuracy'] = (preds == class_target).float().mean().item()
        
        if 'location_pred' in outputs and hasattr(batch, 'x'):
            location_features = batch.x[node_mask, 2:4]  # lat, lon
            location_target = self._extract_location_labels(location_features)
            if location_target is not None:
                location_pred = outputs['location_pred'][node_mask]
                location_loss = F.cross_entropy(location_pred, location_target)
                total_loss += 0.1 * location_loss
        
        if 'performance_pred' in outputs and hasattr(batch, 'x'):
            perf_target = batch.x[node_mask, 1]
            perf_pred = outputs['performance_pred'][node_mask].squeeze()
            if perf_pred.numel() > 0:
                perf_loss = F.mse_loss(perf_pred, perf_target)
                total_loss += 0.1 * perf_loss
        
        if 'node_embeddings' in outputs:
            contrastive_loss = self._enhanced_contrastive_loss(
                outputs['node_embeddings'], batch, node_mask
            )
            total_loss += 0.1 * contrastive_loss
        
        return total_loss, metrics
    
    def _extract_location_labels(self, location_features):

        mumbai = torch.tensor([19.0760, 72.8777], device=self.device)
        delhi = torch.tensor([28.7041, 77.1025], device=self.device)
        bangalore = torch.tensor([12.9716, 77.5946], device=self.device)
        
        locations = torch.stack([mumbai, delhi, bangalore])
        
        #closest dist find karna
        distances = torch.cdist(location_features, locations)
        location_labels = distances.argmin(dim=1)
        
        return location_labels
    
    def _enhanced_contrastive_loss(self, embeddings, batch, node_mask):
        if not hasattr(batch, 'supplier_labels'):
            return torch.tensor(0.0, device=self.device)
        
        embeddings = F.normalize(embeddings[node_mask], p=2, dim=1)
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        
        labels = batch.supplier_labels[node_mask]
        batch_size = embeddings.size(0)
        
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        pos_mask.fill_diagonal_(0)
        neg_mask = 1 - pos_mask - torch.eye(batch_size, device=self.device)
        
        temperature = 0.1
        similarity_matrix = similarity_matrix / temperature
        
        neg_similarities = similarity_matrix * neg_mask
        hard_negatives, _ = neg_similarities.topk(k=min(5, neg_mask.sum(dim=1).max().int()), dim=1)
        
        pos_similarities = similarity_matrix * pos_mask
        pos_exp = torch.exp(pos_similarities).sum(dim=1)
        neg_exp = torch.exp(hard_negatives).sum(dim=1)
        
        loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-8))
        return loss.mean()
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_metrics = {'accuracy': 0.0, 'flow_mse': 0.0}
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch.x, batch.edge_index,
                                   edge_attr=getattr(batch, 'edge_attr', None))
                loss, metrics = self._calculate_enhanced_loss(outputs, batch, is_training=False)
                
                total_loss += loss.item()
                for key in all_metrics:
                    all_metrics[key] += metrics[key]
        
        num_batches = len(val_loader)
        avg_loss = total_loss / num_batches
        for key in all_metrics:
            all_metrics[key] /= num_batches
        
        return avg_loss, all_metrics
    
    def train(self, train_loader, val_loader, epochs):
        self.compute_class_weights(train_loader)
        
        # WandB
        wandb.init(project="carbon-gnn-enhanced", config={
            **self.config.__dict__,
            'model_type': 'enhanced_gnn',
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingWarmRestarts'
        })
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training
            train_loss, train_accuracy = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_metrics = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Logging
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_metrics['accuracy'],
                'val_flow_mse': val_metrics['flow_mse'],
                'learning_rate': current_lr
            })
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"LR: {current_lr:.6f}")
            
            # Early stopping and model saving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'config': self.config
                }, 'best_enhanced_gnn_model.pth')
                print(";) ;) Best model saved! ;) ;)")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                print(f";) ;) Early stopping at epoch {epoch+1} ;) ;)")
                break
        
        wandb.finish()
        print(f";) ;) Training completed! Best validation loss: {self.best_val_loss:.4f} ;) ;)")