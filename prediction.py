import torch
import numpy as np
import pandas as pd
from typing import List, Dict
import os

# Import your existing modules
from config.model_config import EnhancedGNNConfig
from models.gnn_model import ImprovedCarbonGNN
from data.graph_builder import CarbonGraphBuilder
from data.data_loader import ImprovedDataLoader

class CarbonPredictor:
    """
    Main prediction class for carbon impact analysis and supplier recommendations
    """
    
    def __init__(self, model_path: str = "best_enhanced_gnn_model.pth", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config = EnhancedGNNConfig()
        self.model_path = model_path
        
        # Initialize components
        self.model = None
        self.data_loader = ImprovedDataLoader(self.config)
        self.graph_builder = CarbonGraphBuilder(self.config)
        
        # Data storage
        self.suppliers_df = None
        self.relationships_df = None
        self.graph_data = None
        self.supplier_embeddings = None
        
        # Load model and data
        self._load_model()
        self._load_data()
        self._precompute_embeddings()
    
    def _load_model(self):
        """Load trained GNN model"""
        try:
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model = ImprovedCarbonGNN(self.config).to(self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                print(f"✅ Model loaded from {self.model_path}")
            else:
                print(f"⚠️  Model file not found at {self.model_path}")
                print("Training a new model...")
                self._train_new_model()
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self._train_new_model()
    
    def _train_new_model(self):
        """Train a new model if no saved model exists"""
        print("🔄 Training new model...")
        from training.trainer import EnhancedGNNTrainer
        from torch_geometric.loader import DataLoader
        
        # Initialize model
        self.model = ImprovedCarbonGNN(self.config).to(self.device)
        
        # Load data for training
        suppliers_df, relationships_df = self.data_loader.load_enhanced_data()
        graph_data = self.graph_builder.build_supplier_graph(suppliers_df, relationships_df)
        
        # Create data loaders
        train_loader = DataLoader([graph_data], batch_size=1, shuffle=True)
        val_loader = DataLoader([graph_data], batch_size=1, shuffle=False)
        
        # Train model
        trainer = EnhancedGNNTrainer(self.model, self.config, device=self.device)
        trainer.train(train_loader, val_loader, epochs=50)  # Quick training
        
        print("✅ New model trained!")
    
    def _load_data(self):
        """Load and process supplier data"""
        self.suppliers_df, self.relationships_df = self.data_loader.load_enhanced_data()
        self.graph_data = self.graph_builder.build_supplier_graph(
            self.suppliers_df, self.relationships_df
        )
        print(f"✅ Data loaded: {len(self.suppliers_df)} suppliers, {len(self.relationships_df)} relationships")
    
    def _precompute_embeddings(self):
        """Precompute supplier embeddings for fast retrieval"""
        if self.model is None or self.graph_data is None:
            return
        
        with torch.no_grad():
            graph_data = self.graph_data.to(self.device)
            outputs = self.model(
                graph_data.x, 
                graph_data.edge_index, 
                edge_attr=getattr(graph_data, 'edge_attr', None)
            )
            self.supplier_embeddings = outputs['node_embeddings'].cpu().numpy()
        
        print("✅ Supplier embeddings precomputed")
    
    def predict_carbon_impact(self, supplier_id: str) -> Dict:
        """
        Predict carbon impact for a specific supplier
        
        Args:
            supplier_id: Supplier identifier
            
        Returns:
            Dictionary with carbon impact predictions
        """
        try:
            # Find supplier in dataframe
            supplier_row = self.suppliers_df[
                self.suppliers_df['supplier_id'] == supplier_id
            ]
            
            if supplier_row.empty:
                return {
                    'error': f'Supplier {supplier_id} not found',
                    'available_suppliers': self.suppliers_df['supplier_id'].head(10).tolist()
                }
            
            supplier_idx = supplier_row.index[0]
            supplier_data = supplier_row.iloc[0]
            
            # Get supplier embedding
            if self.supplier_embeddings is not None:
                embedding = self.supplier_embeddings[supplier_idx]
            else:
                embedding = None
            
            # Get relationships for this supplier
            supplier_relationships = self.relationships_df[
                (self.relationships_df['supplier_from_id'] == supplier_id) |
                (self.relationships_df['supplier_to_id'] == supplier_id)
            ]
            
            # Calculate carbon impact metrics
            direct_emissions = supplier_data['carbon_intensity']
            indirect_emissions = supplier_relationships['carbon_flow'].sum()
            total_carbon_impact = direct_emissions + indirect_emissions
            
            # Get performance metrics
            performance_score = supplier_data['performance_score']
            carbon_efficiency = supplier_data.get('carbon_efficiency', 0)
            sustainability_score = supplier_data.get('sustainability_score', 0)
            
            # Predict future carbon flow using model
            predicted_flows = []
            if self.model is not None:
                with torch.no_grad():
                    graph_data = self.graph_data.to(self.device)
                    outputs = self.model(
                        graph_data.x, 
                        graph_data.edge_index,
                        edge_attr=getattr(graph_data, 'edge_attr', None)
                    )
                    
                    # Get predictions for edges involving this supplier
                    edge_predictions = outputs['carbon_flows'].cpu().numpy()
                    
                    # Find edges involving this supplier
                    edge_index = graph_data.edge_index.cpu().numpy()
                    supplier_edges = np.where(
                        (edge_index[0] == supplier_idx) | (edge_index[1] == supplier_idx)
                    )[0]
                    
                    predicted_flows = edge_predictions[supplier_edges].flatten()
            
            result = {
                'supplier_id': supplier_id,
                'location': supplier_data['location'],
                'category': supplier_data['category'],
                'carbon_impact': {
                    'direct_emissions': float(direct_emissions),
                    'indirect_emissions': float(indirect_emissions),
                    'total_impact': float(total_carbon_impact),
                    'predicted_future_flows': predicted_flows.tolist() if len(predicted_flows) > 0 else []
                },
                'performance_metrics': {
                    'performance_score': float(performance_score),
                    'carbon_efficiency': float(carbon_efficiency),
                    'sustainability_score': float(sustainability_score),
                    'renewable_percentage': float(supplier_data.get('renewable_percentage', 0))
                },
                'relationships': {
                    'total_connections': len(supplier_relationships),
                    'average_flow': float(supplier_relationships['carbon_flow'].mean()) if len(supplier_relationships) > 0 else 0
                }
            }
            
            return result
            
        except Exception as e:
            return {'error': f'Error predicting carbon impact: {str(e)}'}
    
    def recommend_green_suppliers(self, location: str = "Mumbai", top_k: int = 5) -> List[Dict]:
        """
        Recommend top green suppliers based on carbon efficiency
        
        Args:
            location: Filter by location
            top_k: Number of top suppliers to return
            
        Returns:
            List of recommended suppliers
        """
        try:
            # Filter suppliers by location
            if location:
                filtered_suppliers = self.suppliers_df[
                    self.suppliers_df['location'] == location
                ].copy()
            else:
                filtered_suppliers = self.suppliers_df.copy()
            
            if filtered_suppliers.empty:
                available_locations = self.suppliers_df['location'].unique().tolist()
                return [{
                    'error': f'No suppliers found in {location}',
                    'available_locations': available_locations
                }]
            
            # Calculate comprehensive green score
            filtered_suppliers['green_score'] = (
                0.3 * filtered_suppliers['carbon_efficiency'] +
                0.25 * (1 - filtered_suppliers['carbon_intensity']) +
                0.25 * filtered_suppliers['sustainability_score'] +
                0.2 * filtered_suppliers['renewable_percentage']
            )
            
            # Sort by green score
            top_suppliers = filtered_suppliers.nlargest(top_k, 'green_score')
            
            recommendations = []
            for _, supplier in top_suppliers.iterrows():
                # Get additional insights
                supplier_relationships = self.relationships_df[
                    (self.relationships_df['supplier_from_id'] == supplier['supplier_id']) |
                    (self.relationships_df['supplier_to_id'] == supplier['supplier_id'])
                ]
                
                recommendation = {
                    'supplier_id': supplier['supplier_id'],
                    'location': supplier['location'],
                    'category': supplier['category'],
                    'green_score': float(supplier['green_score']),
                    'metrics': {
                        'carbon_intensity': float(supplier['carbon_intensity']),
                        'carbon_efficiency': float(supplier['carbon_efficiency']),
                        'sustainability_score': float(supplier['sustainability_score']),
                        'renewable_percentage': float(supplier['renewable_percentage']),
                        'performance_score': float(supplier['performance_score'])
                    },
                    'network_info': {
                        'total_connections': len(supplier_relationships),
                        'avg_carbon_flow': float(supplier_relationships['carbon_flow'].mean()) if len(supplier_relationships) > 0 else 0
                    },
                    'recommendation_reason': self._get_recommendation_reason(supplier)
                }
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            return [{'error': f'Error getting recommendations: {str(e)}'}]
    
    def _get_recommendation_reason(self, supplier: pd.Series) -> str:
        """Generate recommendation reason"""
        reasons = []
        
        if supplier['carbon_efficiency'] > 0.7:
            reasons.append("High carbon efficiency")
        if supplier['renewable_percentage'] > 0.6:
            reasons.append("High renewable energy usage")
        if supplier['sustainability_score'] > 0.7:
            reasons.append("Strong sustainability practices")
        if supplier['carbon_intensity'] < 0.3:
            reasons.append("Low carbon emissions")
        
        return ", ".join(reasons) if reasons else "Balanced environmental performance"
    
    def get_supplier_similarity(self, supplier_id: str, top_k: int = 5) -> List[Dict]:
        """Find similar suppliers based on embeddings"""
        try:
            if self.supplier_embeddings is None:
                return [{'error': 'Supplier embeddings not available'}]
            
            # Find supplier index
            supplier_row = self.suppliers_df[
                self.suppliers_df['supplier_id'] == supplier_id
            ]
            
            if supplier_row.empty:
                return [{'error': f'Supplier {supplier_id} not found'}]
            
            supplier_idx = supplier_row.index[0]
            query_embedding = self.supplier_embeddings[supplier_idx]
            
            # Calculate similarities
            similarities = np.dot(self.supplier_embeddings, query_embedding)
            
            # Get top similar suppliers (excluding the query supplier itself)
            similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
            
            similar_suppliers = []
            for idx in similar_indices:
                similar_supplier = self.suppliers_df.iloc[idx]
                similar_suppliers.append({
                    'supplier_id': similar_supplier['supplier_id'],
                    'location': similar_supplier['location'],
                    'category': similar_supplier['category'],
                    'similarity_score': float(similarities[idx]),
                    'carbon_efficiency': float(similar_supplier['carbon_efficiency'])
                })
            
            return similar_suppliers
            
        except Exception as e:
            return [{'error': f'Error finding similar suppliers: {str(e)}'}]
    
    def batch_predict(self, supplier_ids: List[str]) -> Dict:
        """Batch prediction for multiple suppliers"""
        results = {}
        for supplier_id in supplier_ids:
            results[supplier_id] = self.predict_carbon_impact(supplier_id)
        return results
    
    def get_available_suppliers(self, location: str = None, category: str = None) -> List[str]:
        """Get list of available suppliers with optional filtering"""
        df = self.suppliers_df
        
        if location:
            df = df[df['location'] == location]
        if category:
            df = df[df['category'] == category]
        
        return df['supplier_id'].tolist()


# Simple utility functions for easy usage
def predict_carbon_impact(supplier_id: str, model_path: str = "best_enhanced_gnn_model.pth") -> str:
    """
    Simple prediction function - your original request
    """
    predictor = CarbonPredictor(model_path)
    result = predictor.predict_carbon_impact(supplier_id)
    
    if 'error' in result:
        return f"Error: {result['error']}"
    
    carbon_flow = result['carbon_impact']['total_impact']
    return f"Supplier {supplier_id} carbon impact: {carbon_flow:.2f}"


def recommend_green_suppliers(location: str = "Mumbai", top_k: int = 5) -> List[Dict]:
    """
    Simple recommendation function - your original request
    """
    predictor = CarbonPredictor()
    suppliers = predictor.recommend_green_suppliers(location, top_k)
    
    # Filter for your original format
    green_suppliers = []
    for supplier in suppliers:
        if 'error' not in supplier:
            green_suppliers.append({
                'supplier_id': supplier['supplier_id'],
                'carbon_efficiency': supplier['metrics']['carbon_efficiency'],
                'location': supplier['location'],
                'green_score': supplier['green_score']
            })
    
    return sorted(green_suppliers, key=lambda x: x['carbon_efficiency'], reverse=True)[:top_k]


# Example usage and testing
if __name__ == "__main__":
    # Initialize predictor
    predictor = CarbonPredictor()
    
    # Test 1: Simple prediction
    print("=== Testing Carbon Impact Prediction ===")
    result = predict_carbon_impact("SUP_001")
    print(result)
    
    # Test 2: Detailed prediction
    print("\n=== Testing Detailed Prediction ===")
    detailed_result = predictor.predict_carbon_impact("SUP_001")
    print(f"Detailed result: {detailed_result}")
    
    # Test 3: Green supplier recommendations
    print("\n=== Testing Green Supplier Recommendations ===")
    green_suppliers = recommend_green_suppliers("Mumbai", 3)
    print("Top green suppliers:")
    for supplier in green_suppliers:
        print(f"- {supplier['supplier_id']}: efficiency={supplier['carbon_efficiency']:.3f}")
    
    # Test 4: Similar suppliers
    print("\n=== Testing Similar Suppliers ===")
    similar = predictor.get_supplier_similarity("SUP_001", 3)
    print("Similar suppliers:")
    for supplier in similar:
        if 'error' not in supplier:
            print(f"- {supplier['supplier_id']}: similarity={supplier['similarity_score']:.3f}")
    
    # Test 5: Available suppliers
    print("\n=== Available Suppliers ===")
    mumbai_suppliers = predictor.get_available_suppliers(location="Mumbai")
    print(f"Mumbai suppliers: {mumbai_suppliers[:5]}...")  # First 5