# """
# Machine Learning Models Module
# Contains GNN, Bandit, and Transformer implementations
# """

# from .gnn_model import CarbonGNN, GraphSAGEModel, GCNModel
# from .bandit_model import MultiArmedBandit, EpsilonGreedyBandit, UCBBandit
# from .transformer_model import CarbonTransformer, TimeSeriesTransformer
# from .model_utils import ModelUtils, EmbeddingUtils, PredictionUtils

# # Import base classes
# from .gnn_model import BaseGNN
# from .bandit_model import BaseBandit
# from .transformer_model import BaseTransformer

# __version__ = "1.0.0"

# # All models available for import
# __all__ = [
#     # GNN Models
#     'CarbonGNN',
#     'GraphSAGEModel', 
#     'GCNModel',
#     'BaseGNN',
    
#     # Bandit Models
#     'MultiArmedBandit',
#     'EpsilonGreedyBandit',
#     'UCBBandit',
#     'BaseBandit',
    
#     # Transformer Models
#     'CarbonTransformer',
#     'TimeSeriesTransformer',
#     'BaseTransformer',
    
#     # Utilities
#     'ModelUtils',
#     'EmbeddingUtils',
#     'PredictionUtils'
# ]

# # Model factory functions
# def create_gnn_model(config, model_type='sage'):
#     """Factory function to create GNN models"""
#     if model_type == 'sage':
#         return CarbonGNN(config)
#     elif model_type == 'gcn':
#         return GCNModel(config)
#     else:
#         raise ValueError(f"Unknown GNN model type: {model_type}")

# def create_bandit_model(config, strategy='epsilon_greedy'):
#     """Factory function to create Bandit models"""
#     if strategy == 'epsilon_greedy':
#         return EpsilonGreedyBandit(config)
#     elif strategy == 'ucb':
#         return UCBBandit(config)
#     else:
#         return MultiArmedBandit(config)

# def create_transformer_model(config):
#     """Factory function to create Transformer model"""
#     return CarbonTransformer(config)

# # Add factory functions to exports
# __all__.extend(['create_gnn_model', 'create_bandit_model', 'create_transformer_model'])