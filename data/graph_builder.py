import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np

class CarbonGraphBuilder:
    def __init__(self, config):
        self.config = config
        
    def build_supplier_graph(self, suppliers_df, relationships_df):
        """
        Main function: Raw data se graph banata hai
        """
        # Step 1: Node features prepare karo
        node_features = self._prepare_node_features(suppliers_df)
        
        # Step 2: Edge connections banao
        edge_index, edge_weights = self._build_edges(relationships_df)
        
        # Step 3: Graph data object create karo
        graph_data = Data(
            x=torch.FloatTensor(node_features),
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.FloatTensor(edge_weights)
        )
        
        return graph_data
    
    def _prepare_node_features(self, suppliers_df):
        """
        Supplier features ko numerical format mein convert karo
        """
        features = []
        
        for _, supplier in suppliers_df.iterrows():
            feature_vector = [
                supplier['carbon_intensity'],           # Raw carbon intensity
                supplier['performance_score'],          # Historical performance
                self._encode_location(supplier['location']),  # Location encoding
                self._encode_category(supplier['category']),  # Category encoding
                supplier['renewable_percentage'],       # Green energy %
                supplier['production_volume'],          # Scale
                supplier['cost_efficiency'],            # Cost factor
                # Add more features as needed...
            ]
            features.append(feature_vector)
            
        return np.array(features)
    
    def _build_edges(self, relationships_df):
        """
        Supplier relationships se edges banao
        """
        edge_index = [[], []]  # [source_nodes, target_nodes]
        edge_weights = []
        
        for _, relation in relationships_df.iterrows():
            source_id = relation['supplier_from_id']
            target_id = relation['supplier_to_id']
            carbon_flow = relation['carbon_flow']
            
            # Only significant carbon flows ko edges banao
            if carbon_flow > self.config.EDGE_THRESHOLD:
                edge_index[0].append(source_id)
                edge_index[1].append(target_id)
                edge_weights.append([
                    carbon_flow,
                    relation['volume'],
                    relation['transportation_emissions']
                ])
        
        return edge_index, edge_weights
    
    def _encode_location(self, location):
        """Location ko lat/long coordinates mein convert"""
        # Implement location encoding logic
        location_map = {
            'Mumbai': [19.0760, 72.8777],
            'Delhi': [28.7041, 77.1025],
            'Bangalore': [12.9716, 77.5946],
            # Add more cities...
        }
        return location_map.get(location, [0.0, 0.0])
    
    def _encode_category(self, category):
        """Category ko one-hot encoding"""
        categories = ['Electronics', 'Textiles', 'Automotive', 'Chemical']
        return [1 if cat == category else 0 for cat in categories]