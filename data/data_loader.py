import torch
import numpy as np
from torch_geometric.utils import dropout_adj, add_random_edge
from torch_geometric.data import Data
import pandas as pd

class GraphDataAugmentation:
    def __init__(self, config):
        self.config = config
        self.node_noise_std = config.NODE_FEATURE_NOISE
        self.edge_dropout_rate = config.EDGE_DROPOUT_RATE
    
    def augment_graph_data(self, data, training=True):
        """Apply data augmentation to graph data"""
        if not training:
            return data
        
        # Create augmented copy
        augmented_data = data.clone()
        
        # 1. Add noise to node features
        augmented_data.x = self._add_node_noise(augmented_data.x)
        
        # 2. Random edge dropout
        augmented_data.edge_index, augmented_data.edge_attr = self._edge_dropout(
            augmented_data.edge_index, augmented_data.edge_attr
        )
        
        # 3. Feature masking (randomly mask some features)
        augmented_data.x = self._feature_masking(augmented_data.x)
        
        return augmented_data
    
    def _add_node_noise(self, node_features):
        """Add Gaussian noise to node features"""
        noise = torch.randn_like(node_features) * self.node_noise_std
        return node_features + noise
    
    def _edge_dropout(self, edge_index, edge_attr):
        """Randomly drop edges"""
        edge_index, edge_attr = dropout_adj(
            edge_index, edge_attr, 
            p=self.edge_dropout_rate,
            training=True
        )
        return edge_index, edge_attr
    
    def _feature_masking(self, node_features, mask_prob=0.1):
        """Randomly mask some node features"""
        mask = torch.rand_like(node_features) < mask_prob
        masked_features = node_features.clone()
        masked_features[mask] = 0
        return masked_features
    
    def create_multiple_views(self, data):
        """Create multiple augmented views for contrastive learning"""
        view1 = self.augment_graph_data(data, training=True)
        view2 = self.augment_graph_data(data, training=True)
        return view1, view2

class ImprovedDataLoader:
    def __init__(self, config):
        self.config = config
        self.augmentation = GraphDataAugmentation(config)
    
    def load_enhanced_data(self):
        """Load data with better feature engineering"""
        # Original data loading
        suppliers_df, relationships_df = self._load_from_databases()
        
        # Enhanced feature engineering
        suppliers_df = self._enhance_supplier_features(suppliers_df)
        relationships_df = self._enhance_relationship_features(relationships_df)
        
        return suppliers_df, relationships_df
    
    def _load_from_databases(self):
        """Enhanced data loading with better distributions"""
        num_suppliers = 500  # More suppliers for better training
        
        # Better feature distributions
        suppliers_df = pd.DataFrame({
            'supplier_id': [f'SUP_{i:03d}' for i in range(num_suppliers)],
            'carbon_intensity': np.random.beta(2, 5, num_suppliers),  # More realistic distribution
            'location': np.random.choice(['Mumbai', 'Delhi', 'Bangalore'], num_suppliers, p=[0.4, 0.3, 0.3]),
            'category': np.random.choice(['Electronics', 'Textiles', 'Automotive', 'Chemical'], 
                                       num_suppliers, p=[0.3, 0.25, 0.25, 0.2]),
            'performance_score': np.random.beta(3, 2, num_suppliers),  # Higher performance bias
            'renewable_percentage': np.random.beta(2, 3, num_suppliers),
            'production_volume': np.random.lognormal(0, 1, num_suppliers),  # Log-normal for volume
            'cost_efficiency': np.random.beta(2, 2, num_suppliers),
        })
        
        # More realistic relationships
        num_relationships = 1500  # More edges for better connectivity
        relationships_df = pd.DataFrame({
            'supplier_from_id': np.random.choice(suppliers_df['supplier_id'], num_relationships),
            'supplier_to_id': np.random.choice(suppliers_df['supplier_id'], num_relationships),
            'carbon_flow': np.random.beta(2, 5, num_relationships),
            'volume': np.random.lognormal(0, 1, num_relationships),
            'transportation_emissions': np.random.gamma(2, 0.5, num_relationships),
        })
        
        # Remove self-loops
        relationships_df = relationships_df[
            relationships_df['supplier_from_id'] != relationships_df['supplier_to_id']
        ]
        
        return suppliers_df, relationships_df
    
    def _enhance_supplier_features(self, suppliers_df):
        """Add derived features"""
        # Carbon efficiency score
        suppliers_df['carbon_efficiency'] = (
            suppliers_df['performance_score'] / (suppliers_df['carbon_intensity'] + 0.1)
        )
        
        # Sustainability score
        suppliers_df['sustainability_score'] = (
            0.4 * suppliers_df['renewable_percentage'] + 
            0.3 * (1 - suppliers_df['carbon_intensity']) +
            0.3 * suppliers_df['performance_score']
        )
        
        # Geographic clustering
        location_coords = {
            'Mumbai': [19.0760, 72.8777],
            'Delhi': [28.7041, 77.1025], 
            'Bangalore': [12.9716, 77.5946]
        }
        suppliers_df['latitude'] = suppliers_df['location'].map(lambda x: location_coords[x][0])
        suppliers_df['longitude'] = suppliers_df['location'].map(lambda x: location_coords[x][1])
        
        return suppliers_df
    
    def _enhance_relationship_features(self, relationships_df):
        """Add derived relationship features"""
        # Transport efficiency
        relationships_df['transport_efficiency'] = (
            relationships_df['volume'] / (relationships_df['transportation_emissions'] + 0.1)
        )
        
        # Carbon per unit volume
        relationships_df['carbon_per_volume'] = (
            relationships_df['carbon_flow'] / (relationships_df['volume'] + 0.1)
        )
        
        return relationships_df

# Usage in your main training script
def get_enhanced_data_pipeline():
    """Get complete enhanced data pipeline"""
    config = EnhancedGNNConfig()
    data_loader = ImprovedDataLoader(config)
    return data_loader, config