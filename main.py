from config.model_config import EnhancedGNNConfig
from data.graph_builder import CarbonGraphBuilder
from data.data_loader import ImprovedDataLoader
from models.gnn_model import ImprovedCarbonGNN
from training.trainer import EnhancedGNNTrainer
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.model_selection import train_test_split

def create_data_loaders(graph_data, config, train_ratio=0.8):
    """Create train/validation data loaders"""
    
    # Simple approach: duplicate the graph data for batching
    # In practice, you'd have multiple graphs or use different splitting strategies
    
    # For now, create multiple copies with different augmentations
    data_list = [graph_data] * config.BATCH_SIZE
    
    # Split indices for train/val
    total_size = len(data_list)
    train_size = int(total_size * train_ratio)
    
    train_data = data_list[:train_size] if train_size > 0 else [graph_data]
    val_data = data_list[train_size:] if train_size < total_size else [graph_data]
    
    # Create data loaders
    train_loader = PyGDataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = PyGDataLoader(val_data, batch_size=1, shuffle=False)
    
    return train_loader, val_loader

def add_train_val_masks(graph_data):
    """Add training/validation masks to graph data"""
    num_nodes = graph_data.x.size(0)
    num_edges = graph_data.edge_index.size(1)
    
    # Node masks (80% train, 20% val)
    node_indices = torch.randperm(num_nodes)
    train_size = int(0.8 * num_nodes)
    
    graph_data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    graph_data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    graph_data.train_mask[node_indices[:train_size]] = True
    graph_data.val_mask[node_indices[train_size:]] = True
    
    # Edge masks (80% train, 20% val)
    edge_indices = torch.randperm(num_edges)
    edge_train_size = int(0.8 * num_edges)
    
    graph_data.train_edge_mask = torch.zeros(num_edges, dtype=torch.bool)
    graph_data.val_edge_mask = torch.zeros(num_edges, dtype=torch.bool)
    
    graph_data.train_edge_mask[edge_indices[:edge_train_size]] = True
    graph_data.val_edge_mask[edge_indices[edge_train_size:]] = True
    
    return graph_data

def main():
    # Configuration
    config = EnhancedGNNConfig()
    print(f"Using config: INPUT_DIM={config.INPUT_DIM}, HIDDEN_DIM={config.HIDDEN_DIM}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loading and preprocessing
    print("Loading enhanced data...")
    data_loader = ImprovedDataLoader(config)
    suppliers_df, relationships_df = data_loader.load_enhanced_data()
    
    print(f"Loaded {len(suppliers_df)} suppliers and {len(relationships_df)} relationships")
    
    # Graph construction
    print("Building supplier graph...")
    graph_builder = CarbonGraphBuilder(config)
    graph_data = graph_builder.build_supplier_graph(suppliers_df, relationships_df)
    
    print(f"Graph created: {graph_data.x.shape[0]} nodes, {graph_data.edge_index.shape[1]} edges")
    print(f"Node features shape: {graph_data.x.shape}")
    print(f"Edge features shape: {graph_data.edge_attr.shape if graph_data.edge_attr is not None else 'None'}")
    
    # Add train/validation masks
    graph_data = add_train_val_masks(graph_data)
    
    # Model initialization
    print("Initializing enhanced model...")
    model = ImprovedCarbonGNN(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    trainer = EnhancedGNNTrainer(model, config, device=device)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(graph_data, config)
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Training
    print("Starting enhanced training...")
    try:
        trainer.train(train_loader, val_loader, epochs=config.EPOCHS)
        print("Training completed successfully! ✅")
    except Exception as e:
        print(f"Training failed with error: {e}")
        print("Check your CUDA installation and memory usage")
    
    # Load best model for inference
    try:
        checkpoint = torch.load('best_enhanced_gnn_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Best model loaded for inference! ✅")
        
        # Quick inference test
        model.eval()
        with torch.no_grad():
            test_batch = next(iter(train_loader)).to(device)
            outputs = model(test_batch.x, test_batch.edge_index, 
                          edge_attr=getattr(test_batch, 'edge_attr', None))
            print(f"Inference test successful!")
            print(f"Node embeddings shape: {outputs['node_embeddings'].shape}")
            print(f"Carbon flows predictions: {outputs['carbon_flows'].shape}")
            
    except Exception as e:
        print(f"Model loading failed: {e}")

if __name__ == "__main__":
    main()