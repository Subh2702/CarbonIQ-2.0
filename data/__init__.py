# """
# Data processing module for Carbon GNN
# Handles data loading, preprocessing, and graph construction
# """

# from .data_loader import DataLoader, DatabaseConnector
# from .graph_builder import CarbonGraphBuilder, GraphProcessor
# from .preprocessor import DataPreprocessor, FeatureScaler
# from .feature_engineering import FeatureEngineer, CarbonFeatureExtractor

# # Import utility functions
# from .data_loader import load_supplier_data, load_relationships
# from .graph_builder import build_graph_from_dataframes
# from .preprocessor import normalize_features, handle_missing_values

# # Version and metadata
# __version__ = "1.0.0"

# # Main classes for easy import
# __all__ = [
#     # Main classes
#     'DataLoader',
#     'DatabaseConnector',
#     'CarbonGraphBuilder',
#     'GraphProcessor',
#     'DataPreprocessor',
#     'FeatureScaler',
#     'FeatureEngineer',
#     'CarbonFeatureExtractor',
    
#     # Utility functions
#     'load_supplier_data',
#     'load_relationships',
#     'build_graph_from_dataframes',
#     'normalize_features',
#     'handle_missing_values'
# ]

# # Quick access functions
# def get_data_pipeline():
#     """Returns complete data processing pipeline"""
#     return {
#         'loader': DataLoader(),
#         'graph_builder': CarbonGraphBuilder(),
#         'preprocessor': DataPreprocessor(),
#         'feature_engineer': FeatureEngineer()
#     }