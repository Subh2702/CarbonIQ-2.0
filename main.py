from config.model_config import EnhancedGNNConfig
from data.graph_builder import CarbonGraphBuilder
from data.data_loader import ImprovedDataLoader
from models.gnn_model import ImprovedCarbonGNN
from training.trainer import EnhancedGNNTrainer
from tests.real_word import TestingForRealWorld
from training.demonstrate_bandit_integration import DemonstrateBandit

import torch
from torch_geometric.loader import DataLoader as PyGDataLoader
import numpy as np
import pandas as pd
import logging

def create_data_loaders(graph_data, config, train_ratio=0.8):
    # multiple copies with different agumentation
    data_list = [graph_data] * config.BATCH_SIZE
    total_size = len(data_list)
    train_size = int(total_size * train_ratio)
    train_data = data_list[:train_size] if train_size > 0 else [graph_data]
    val_data = data_list[train_size:] if train_size < total_size else [graph_data]
    # Create data loaders
    train_loader = PyGDataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = PyGDataLoader(val_data, batch_size=1, shuffle=False)
    return train_loader, val_loader

def add_train_val_masks(graph_data):
    num_nodes = graph_data.x.size(0)
    num_edges = graph_data.edge_index.size(1)
    
    node_indices = torch.randperm(num_nodes)
    train_size = int(0.8 * num_nodes)
    
    graph_data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    graph_data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    graph_data.train_mask[node_indices[:train_size]] = True
    graph_data.val_mask[node_indices[train_size:]] = True
    
    edge_indices = torch.randperm(num_edges)
    edge_train_size = int(0.8 * num_edges)
    
    graph_data.train_edge_mask = torch.zeros(num_edges, dtype=torch.bool)
    graph_data.val_edge_mask = torch.zeros(num_edges, dtype=torch.bool)
    
    graph_data.train_edge_mask[edge_indices[:edge_train_size]] = True
    graph_data.val_edge_mask[edge_indices[edge_train_size:]] = True
    
    return graph_data

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    config = EnhancedGNNConfig()
    
    logger.info(f"Using config: INPUT_DIM={config.INPUT_DIM}, HIDDEN_DIM={config.HIDDEN_DIM}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    logger.info("Loading data")
    data_loader = ImprovedDataLoader(config)
    
    suppliers_df, relationships_df = data_loader.load_enhanced_data()
    logger.info(f"Loaded {len(suppliers_df)} suppliers and {len(relationships_df)} relationships")
    logger.info("Building supplier graph")
    
    graph_builder = CarbonGraphBuilder(config)
    graph_data = graph_builder.build_supplier_graph(suppliers_df, relationships_df)
    
    # Robust shape checks
    num_nodes = getattr(graph_data.x, 'shape', [0])[0] if graph_data.x is not None else 0
    num_edges = getattr(graph_data.edge_index, 'shape', [[0],[0]])[1] if graph_data.edge_index is not None else 0
    logger.info(f"Graph created: {num_nodes} nodes, {num_edges} edges")
    logger.info(f"Node features shape: {getattr(graph_data.x, 'shape', None)}")
    logger.info(f"Edge features shape: {getattr(graph_data.edge_attr, 'shape', None) if graph_data.edge_attr is not None else 'None'}")
    
    graph_data = add_train_val_masks(graph_data)
    logger.info("Initializing enhanced model")
    
    model = ImprovedCarbonGNN(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Use input() for interactive, but fallback to default for non-interactive
    try:
        train_model = input("\n Train the model? (y/n): ").lower().strip() == 'y'
    except Exception:
        train_model = True  # Default to training in non-interactive
    if train_model:
        # EnhancedGNNTrainer expects device as str or torch.device, both are supported
        trainer = EnhancedGNNTrainer(model, config, device=str(device))
        train_loader, val_loader = create_data_loaders(graph_data, config)
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        logger.info("training started...")
        try:
            trainer.train(train_loader, val_loader, epochs=config.EPOCHS)
            logger.info("Training ended...")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
    try:
        checkpoint = torch.load('best_enhanced_gnn_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
    
        graph_data = graph_data.to(str(device))
        demonstrator = DemonstrateBandit(config)
        demo_results = demonstrator.demonstrate_bandit_integration(model, graph_data)
        gnn_bandit = demo_results['gnn_bandit']
        real_world_tester = TestingForRealWorld(gnn_bandit, graph_data)
        results = real_world_tester.run_scenarios()
        logger.info(f"Real-world scenario results: {results}")
        try:
            save_bandit = input("\nSave bandit learning state? (y/n): ").lower().strip() == 'y'
        except Exception:
            save_bandit = True
        if save_bandit:
            try:
                gnn_bandit.bandit_agent.save_state('trained_bandit_state.pth')
                logger.info("Bandit state saved to 'trained_bandit_state.pth'")
            except Exception as e:
                logger.error(f"Failed to save bandit state: {e}")
    except Exception as e:
        logger.error(f"System initialization failed: {e}")

if __name__ == "__main__":
    main()