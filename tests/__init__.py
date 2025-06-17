# """
# Test Module for Carbon GNN Project
# Unit tests and integration tests
# """

# from .test_models import TestGNN, TestBandit, TestTransformer
# from .test_data import TestDataLoader, TestGraphBuilder
# from .test_training import TestTrainer, TestValidator
# from .test_inference import TestPredictor, TestServer

# # Import test utilities
# from .test_models import run_model_tests
# from .test_data import run_data_tests
# from .test_training import run_training_tests
# from .test_inference import run_inference_tests

# __version__ = "1.0.0"

# __all__ = [
#     # Test classes
#     'TestGNN',
#     'TestBandit',
#     'TestTransformer',
#     'TestDataLoader',
#     'TestGraphBuilder',
#     'TestTrainer',
#     'TestValidator',
#     'TestPredictor',
#     'TestServer',
    
#     # Test runners
#     'run_model_tests',
#     'run_data_tests', 
#     'run_training_tests',
#     'run_inference_tests'
# ]

# # Master test runner
# def run_all_tests():
#     """Runs complete test suite"""
#     results = {
#         'models': run_model_tests(),
#         'data': run_data_tests(),
#         'training': run_training_tests(),
#         'inference': run_inference_tests()
#     }
    
#     return results

# __all__.append('run_all_tests')