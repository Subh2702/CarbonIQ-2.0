# """
# Configuration module for Carbon GNN Project
# Centralizes all configuration classes and settings
# """

# from .model_config import GNNConfig, BanditConfig, TransformerConfig
# from .database_config import DatabaseConfig, Neo4jConfig, InfluxDBConfig, PostgreSQLConfig
# from .training_config import TrainingConfig, ValidationConfig

# # Version info
# __version__ = "1.0.0"
# __author__ = "Carbon GNN Team"

# # Default configurations
# DEFAULT_GNN_CONFIG = GNNConfig()
# DEFAULT_BANDIT_CONFIG = BanditConfig()
# DEFAULT_TRANSFORMER_CONFIG = TransformerConfig()

# # Export all configs
# __all__ = [
#     'GNNConfig',
#     'BanditConfig', 
#     'TransformerConfig',
#     'DatabaseConfig',
#     'Neo4jConfig',
#     'InfluxDBConfig',
#     'PostgreSQLConfig',
#     'TrainingConfig',
#     'ValidationConfig',
#     'DEFAULT_GNN_CONFIG',
#     'DEFAULT_BANDIT_CONFIG',
#     'DEFAULT_TRANSFORMER_CONFIG'
# ]