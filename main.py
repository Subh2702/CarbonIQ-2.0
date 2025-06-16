from config.model_config import GNNConfig
from data.graph_builder import CarbonGraphBuilder
from data.data_loader import DataLoader
from models.gnn_model import CarbonGNN
from training.trainer import GNNTrainer

def main():
    # Configuration
    config = GNNConfig()
    
    # Data loading and preprocessing
    print("Loading data...")
    data_loader = DataLoader(config)
    suppliers_df, relationships_df = data_loader.load_from_databases()
    
    # Graph construction
    print("Building graph...")
    graph_builder = CarbonGraphBuilder(config)
    graph_data = graph_builder.build_supplier_graph(suppliers_df, relationships_df)
    
    # Model initialization
    print("Initializing model...")
    model = CarbonGNN(config)
    
    # Training setup
    trainer = GNNTrainer(model, config)
    
    # Create data loaders
    train_loader, val_loader = data_loader.create_loaders(graph_data)
    
    # Training
    print("Starting training...")
    trainer.train(train_loader, val_loader, epochs=config.EPOCHS)
    
    print("Training completed!")

if __name__ == "__main__":
    main()