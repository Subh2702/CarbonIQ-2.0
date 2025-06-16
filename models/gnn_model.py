import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, global_mean_pool

class CarbonGNN(nn.Module):
    def __init__(self, config):
        super(CarbonGNN, self).__init__()
        self.config = config
        
        # GraphSAGE layers for supplier embedding
        self.convs = nn.ModuleList([
            SAGEConv(config.INPUT_DIM, config.HIDDEN_DIM),
            SAGEConv(config.HIDDEN_DIM, config.HIDDEN_DIM),
            SAGEConv(config.HIDDEN_DIM, config.OUTPUT_DIM)
        ])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.DROPOUT)
        
        # Output heads for different tasks
        self.carbon_flow_predictor = nn.Sequential(
            nn.Linear(config.OUTPUT_DIM * 2, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM, 1)  # Carbon flow between suppliers
        )
        
        self.supplier_classifier = nn.Linear(config.OUTPUT_DIM, 4)  # Supplier categories
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass:
        Input: Node features, Edge connections
        Output: Node embeddings, Carbon flow predictions
        """
        
        # Multi-layer message passing
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:  # No activation on last layer
                h = F.relu(h)
                h = self.dropout(h)
        
        # h is now node embeddings [num_nodes, output_dim]
        node_embeddings = h
        
        # Edge-level predictions (carbon flow between suppliers)
        edge_embeddings = self._get_edge_embeddings(node_embeddings, edge_index)
        carbon_flows = self.carbon_flow_predictor(edge_embeddings)
        
        # Node-level predictions (supplier classification)
        supplier_classes = self.supplier_classifier(node_embeddings)
        
        return {
            'node_embeddings': node_embeddings,
            'carbon_flows': carbon_flows,
            'supplier_classes': supplier_classes
        }
    
    def _get_edge_embeddings(self, node_embeddings, edge_index):
        """
        Edge ke dono ends ke node embeddings ko concatenate karo
        """
        source_embeddings = node_embeddings[edge_index[0]]  # Source nodes
        target_embeddings = node_embeddings[edge_index[1]]  # Target nodes
        
        # Concatenate source and target embeddings
        edge_embeddings = torch.cat([source_embeddings, target_embeddings], dim=1)
        
        return edge_embeddings
    
    def get_supplier_similarity(self, supplier_id1, supplier_id2):
        """
        Do suppliers ke beech similarity calculate karo
        """
        emb1 = self.node_embeddings[supplier_id1]
        emb2 = self.node_embeddings[supplier_id2]
        
        # Cosine similarity
        similarity = F.cosine_similarity(emb1, emb2, dim=0)
        return similarity.item()