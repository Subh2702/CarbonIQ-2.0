import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np

class CarbonGraphBuilder:
    def __init__(self, config):
        self.config = config
        
    def build_supplier_graph(self, suppliers_df, relationships_df):
        # Map supplier IDs to integer indices (assuming supplier_id is in suppliers_df)
        supplier_id_to_index = {sid: idx for idx, sid in enumerate(suppliers_df['supplier_id'])}
    
        # Prepare node features (already implemented)
        node_features = self._prepare_node_features(suppliers_df)
    
        # Build edges and get carbon flow targets
        edge_index, edge_weights, carbon_flow_targets = self._build_edges(relationships_df, supplier_id_to_index)
    
        # Prepare supplier_labels by mapping categories to integers
        category_map = {'Electronics': 0, 'Textiles': 1, 'Automotive': 2, 'Chemical': 3}  # Adjust based on actual categories
        supplier_labels = [category_map[supplier['category']] for _, supplier in suppliers_df.iterrows()]
    
        # Create the Data object with all required attributes
        graph_data = Data(
            x=torch.FloatTensor(node_features),
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.FloatTensor(edge_weights),
            supplier_labels=torch.LongTensor(supplier_labels),
            carbon_flow_targets=torch.FloatTensor(carbon_flow_targets)
        )
        return graph_data
    
    def _prepare_node_features(self, suppliers_df):
        features = []
        for _, supplier in suppliers_df.iterrows():
            feature_vector = [
                supplier['carbon_intensity'],
                supplier['performance_score'],
            ] + self._encode_location(supplier['location']) + self._encode_category(supplier['category']) + [
                supplier['renewable_percentage'],
                supplier['production_volume'],
                supplier['cost_efficiency'],
            ]
            features.append(feature_vector)
        return np.array(features)
    
    def _build_edges(self, relationships_df, supplier_id_to_index):
        edge_index = [[], []]  # [source_indices, target_indices]
        edge_weights = []      # Edge features [carbon_flow, volume, transportation_emissions]
        carbon_flow_targets = []  # Target carbon flow values for each edge
    
        for _, relation in relationships_df.iterrows():
            source_idx = supplier_id_to_index[relation['supplier_from_id']]
            target_idx = supplier_id_to_index[relation['supplier_to_id']]
            carbon_flow = relation['carbon_flow']
        
            # Apply edge threshold (if present in your config)
            if carbon_flow > self.config.EDGE_THRESHOLD:
                edge_index[0].append(source_idx)
                edge_index[1].append(target_idx)
                edge_weights.append([
                    carbon_flow,
                    relation['volume'],
                    relation['transportation_emissions']
                ])
                carbon_flow_targets.append(carbon_flow)
    
        return edge_index, edge_weights, carbon_flow_targets
    
    def _encode_location(self, location):
        location_map = {'Mumbai': [19.0760, 72.8777], 'Delhi': [28.7041, 77.1025], 'Bangalore': [12.9716, 77.5946]}
        return location_map.get(location, [0.0, 0.0])
    
    def _encode_category(self, category):
        categories = ['Electronics', 'Textiles', 'Automotive', 'Chemical']
        return [1 if cat == category else 0 for cat in categories]