import torch
from torch_geometric.data import Data
import pandas as pd
import numpy as np

class CarbonGraphBuilder:
    def __init__(self, config):
        self.config = config
        
    def build_supplier_graph(self, suppliers_df, relationships_df):
        #supplier ko map kia index ke sath kuch ansa dikhega {'SUP_001': 0, 'SUP_002': 1, 'SUP_003': 2}
        supplier_id_to_index = {sid: idx for idx, sid in enumerate(suppliers_df['supplier_id'])}
    
        #har node ke liye maine feature vector banaya
        node_features = self._prepare_node_features(suppliers_df)
    
        # ye sab ki value rakhi maine build edge fucn ke sath
        edge_index, edge_weights, carbon_flow_targets = self._build_edges(relationships_df, supplier_id_to_index)
    
        category_map = {'Electronics': 0, 'Textiles': 1, 'Automotive': 2, 'Chemical': 3}
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
        edge_index = [[], []]  # [source se, target tak]
        edge_weights = []      # Edge ke features [carbon_flow, volume, transportation_emissions]
        carbon_flow_targets = []  # Target carbon flow values for each edge
    
        for _, relation in relationships_df.iterrows():
            source_idx = supplier_id_to_index[relation['supplier_from_id']]
            target_idx = supplier_id_to_index[relation['supplier_to_id']]
            carbon_flow = relation['carbon_flow']
        
            #hum sirf bade carbon flow wali edge consider karenge
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
    
    #code for location and category encoding

    def _encode_location(self, location):
        location_map = {'Mumbai': [19.0760, 72.8777], 'Delhi': [28.7041, 77.1025], 'Bangalore': [12.9716, 77.5946]}
        return location_map.get(location, [0.0, 0.0])
    
    def _encode_category(self, category):
        categories = ['Electronics', 'Textiles', 'Automotive', 'Chemical']
        return [1 if cat == category else 0 for cat in categories]