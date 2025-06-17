# """
# Inference Module for Carbon GNN Project
# Real-time predictions and model serving
# """

# from .predictor import CarbonPredictor, GNNPredictor, BanditPredictor, TransformerPredictor
# from .model_server import ModelServer, FastAPIServer

# # Import utility functions
# from .predictor import predict_carbon_flow, predict_supplier_selection, predict_emissions
# from .model_server import start_server, health_check

# __version__ = "1.0.0"

# __all__ = [
#     # Predictor classes
#     'CarbonPredictor',
#     'GNNPredictor',
#     'BanditPredictor', 
#     'TransformerPredictor',
    
#     # Server classes
#     'ModelServer',
#     'FastAPIServer',
    
#     # Utility functions
#     'predict_carbon_flow',
#     'predict_supplier_selection',
#     'predict_emissions',
#     'start_server',
#     'health_check'
# ]

# # Inference pipeline
# def create_inference_pipeline(model_paths):
#     """Creates complete inference pipeline with all models"""
#     return CarbonPredictor(model_paths)

# __all__.append('create_inference_pipeline')