# """
# Training Module for Carbon GNN Project
# Handles model training, validation, and evaluation
# """

# from .trainer import GNNTrainer, BanditTrainer, TransformerTrainer
# from .validator import ModelValidator, CrossValidator
# from .loss_functions import CarbonLoss, GraphLoss, BanditLoss, TransformerLoss

# # Import training utilities
# from .trainer import train_model, save_checkpoint, load_checkpoint
# from .validator import evaluate_model, compute_metrics
# from .loss_functions import combined_loss, weighted_loss

# __version__ = "1.0.0"

# __all__ = [
#     # Main trainer classes
#     'GNNTrainer',
#     'BanditTrainer', 
#     'TransformerTrainer',
    
#     # Validation classes
#     'ModelValidator',
#     'CrossValidator',
    
#     # Loss functions
#     'CarbonLoss',
#     'GraphLoss',
#     'BanditLoss',
#     'TransformerLoss',
    
#     # Utility functions
#     'train_model',
#     'save_checkpoint',
#     'load_checkpoint',
#     'evaluate_model',
#     'compute_metrics',
#     'combined_loss',
#     'weighted_loss'
# ]

# # Training pipeline factory
# def create_training_pipeline(model_type, config):
#     """Creates complete training pipeline for given model type"""
#     if model_type == 'gnn':
#         return GNNTrainer(config)
#     elif model_type == 'bandit':
#         return BanditTrainer(config)
#     elif model_type == 'transformer':
#         return TransformerTrainer(config)
#     else:
#         raise ValueError(f"Unknown model type: {model_type}")

# __all__.append('create_training_pipeline')