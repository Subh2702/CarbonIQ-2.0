import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, BatchNorm

class ImprovedCarbonGNN(nn.Module):
    def __init__(self, config):
        super(ImprovedCarbonGNN, self).__init__()
        self.config = config
        
        #sari layers ko container mai daaldiya
        self.input_transform = nn.Sequential(
            nn.Linear(config.INPUT_DIM, config.HIDDEN_DIM),
            nn.BatchNorm1d(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        #do list banayi
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        #attention layer aur bachnorm joda
        self.convs.append(GATConv(config.HIDDEN_DIM, config.HIDDEN_DIM // 8, heads=8, dropout=0.1))
        self.batch_norms.append(BatchNorm(config.HIDDEN_DIM))
        
        #2 sage layer and bachnorm joda
        for i in range(config.NUM_LAYERS - 2):
            self.convs.append(SAGEConv(config.HIDDEN_DIM, config.HIDDEN_DIM))
            self.batch_norms.append(BatchNorm(config.HIDDEN_DIM))
        
        #akhri mai output layer joda
        self.convs.append(SAGEConv(config.HIDDEN_DIM, config.OUTPUT_DIM))
        self.batch_norms.append(BatchNorm(config.OUTPUT_DIM))
        
        # Residual connections
        self.use_residual = True
        
        self.dropout1 = nn.Dropout(0.1)  # Light dropout for early layers
        self.dropout2 = nn.Dropout(config.DROPOUT)  # Standard dropout
        
        #carbon flow predict karne ke liye
        self.carbon_flow_predictor = nn.Sequential(
            nn.Linear(config.OUTPUT_DIM * 2 + 3, config.HIDDEN_DIM),  # +3 for edge features
            nn.BatchNorm1d(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.HIDDEN_DIM // 2, 1)
        )
        
        #supplier class predict karne ke liye
        self.supplier_classifier = nn.Sequential(
            nn.Linear(config.OUTPUT_DIM, config.HIDDEN_DIM),
            nn.BatchNorm1d(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.HIDDEN_DIM // 2, 4)  # 4 supplier categories
        )
        
        #simple location aur performance predict karne ke liye
        self.location_predictor = nn.Linear(config.OUTPUT_DIM, 3)  # 3 locations
        self.performance_predictor = nn.Linear(config.OUTPUT_DIM, 1)  # Performance score
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        h = self.input_transform(x)
        
        #h ko layers se pass karna convs aur bach alt
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            h_prev = h
            
            h = conv(h, edge_index)
            
            # Batch normalization
            h = bn(h)
            
            # Activation and dropout (except last layer)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                
                #purane batch ki info bhi add kardo
                if self.use_residual and h.shape == h_prev.shape:
                    h = h + h_prev
                
                # Adaptive dropout
                if i == 0:
                    h = self.dropout1(h)
                else:
                    h = self.dropout2(h)
        
        # Final node embeddings
        node_embeddings = h
        
        # Edge-level predictions with edge features
        edge_embeddings = self._get_enhanced_edge_embeddings(node_embeddings, edge_index, edge_attr)
        carbon_flows = self.carbon_flow_predictor(edge_embeddings)
        
        # Node-level predictions
        supplier_classes = self.supplier_classifier(node_embeddings)
        
        # Auxiliary predictions for multi-task learning
        location_pred = self.location_predictor(node_embeddings)
        performance_pred = self.performance_predictor(node_embeddings)
        
        return {
            'node_embeddings': node_embeddings,
            'carbon_flows': carbon_flows,
            'supplier_classes': supplier_classes,
            'location_pred': location_pred,
            'performance_pred': performance_pred
        }
    
    def _get_enhanced_edge_embeddings(self, node_embeddings, edge_index, edge_attr=None):
        source_embeddings = node_embeddings[edge_index[0]]
        target_embeddings = node_embeddings[edge_index[1]]
        
        # Concatenate node embeddings
        edge_embeddings = torch.cat([source_embeddings, target_embeddings], dim=1)
        
        # Add edge attributes if available
        if edge_attr is not None:
            edge_embeddings = torch.cat([edge_embeddings, edge_attr], dim=1)
        
        return edge_embeddings