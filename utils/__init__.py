# """
# Utility Module for Carbon GNN Project
# Common utilities, database connections, logging, visualization
# """

# from .database import DatabaseManager, Neo4jConnector, InfluxConnector, PostgreSQLConnector
# from .visualization import GraphVisualizer, MetricsPlotter, CarbonFlowVisualizer
# from .logging_utils import setup_logging, get_logger, log_metrics

# # Import utility functions
# from .database import connect_all_databases, test_connections
# from .visualization import plot_graph, plot_training_curves, plot_carbon_flows
# from .logging_utils import log_experiment, save_metrics

# __version__ = "1.0.0"

# __all__ = [
#     # Database utilities
#     'DatabaseManager',
#     'Neo4jConnector',
#     'InfluxConnector', 
#     'PostgreSQLConnector',
    
#     # Visualization utilities
#     'GraphVisualizer',
#     'MetricsPlotter',
#     'CarbonFlowVisualizer',
    
#     # Logging utilities
#     'setup_logging',
#     'get_logger',
#     'log_metrics',
    
#     # Utility functions
#     'connect_all_databases',
#     'test_connections',
#     'plot_graph',
#     'plot_training_curves',
#     'plot_carbon_flows',
#     'log_experiment',
#     'save_metrics'
# ]

# # Global utilities
# def setup_project_environment():
#     """Sets up complete project environment"""
#     # Setup logging
#     setup_logging()
    
#     # Test database connections
#     test_connections()
    
#     # Return status
#     return {"status": "ready", "message": "Project environment initialized"}

# __all__.append('setup_project_environment')