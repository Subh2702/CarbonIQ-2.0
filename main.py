from config.model_config import EnhancedGNNConfig
from data.graph_builder import CarbonGraphBuilder
from data.data_loader import ImprovedDataLoader
from models.gnn_model import ImprovedCarbonGNN
from training.trainer import EnhancedGNNTrainer
from models.bandit_model import SupplierBanditAgent, GNNBanditIntegration  # Import our new bandit model
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader
import numpy as np
import pandas as pd

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

def create_supplier_dataframe(graph_data):
    """Create supplier DataFrame from graph data for bandit reward calculation"""
    num_suppliers = graph_data.x.shape[0]
    
    suppliers_df = pd.DataFrame({
        'supplier_id': [f'SUP_{i:03d}' for i in range(num_suppliers)],
        'carbon_intensity': graph_data.x[:, 0].cpu().numpy(),
        'performance_score': graph_data.x[:, 1].cpu().numpy(),
        'renewable_percentage': graph_data.x[:, 5].cpu().numpy() if graph_data.x.shape[1] > 5 else np.random.random(num_suppliers),
        'cost_efficiency': graph_data.x[:, 7].cpu().numpy() if graph_data.x.shape[1] > 7 else np.random.random(num_suppliers),
        'delivery_reliability': np.random.beta(3, 1, num_suppliers)  # Simulated delivery reliability
    })
    
    return suppliers_df

def demonstrate_bandit_integration(model, graph_data, config):
    """Demonstrate the GNN-Bandit integration for dynamic supplier selection"""
    print("\n" + "-"*60)
    print("DEMONSTRATING GNN-BANDIT INTEGRATION")
    print("-"*60)
    
    # Initialize bandit agent
    num_suppliers = graph_data.x.shape[0]
    bandit_agent = SupplierBanditAgent(config, num_suppliers=num_suppliers)
    
    # Create integration
    gnn_bandit = GNNBanditIntegration(model, bandit_agent)
    
    # Create supplier data for reward calculation
    suppliers_df = create_supplier_dataframe(graph_data)
    
    print(f"Initialized bandit for {num_suppliers} suppliers")
    print(f"Bandit strategy: {bandit_agent.strategy}")
    
    # Demonstrate single selection cycle
    print("\nSINGLE SUPPLIER SELECTION CYCLE")
    print("-" * 40)
    
    demand_forecast = 1500  # Example demand
    selection_result = gnn_bandit.dynamic_supplier_selection(
        graph_data, demand_forecast, num_suppliers=5
    )
    
    print(f"Demand forecast: {demand_forecast}")
    print(f"Selected suppliers: {selection_result['selected_suppliers']}")
    print(f"Selection probabilities: {selection_result['selection_probabilities'][selection_result['selected_suppliers']]}")
    
    # Show detailed reasoning
    print("\nSELECTION REASONING:")
    for reason in selection_result['selection_reasoning']:
        print(f"  {reason['supplier_id']}: Reward={reason['average_reward']:.3f}, "
              f"Count={reason['selection_count']}, Confidence={reason['confidence']}")
    
    # Simulate procurement cycles
    print("\nSIMULATING PROCUREMENT CYCLES")
    print("-" * 40)
    
    simulation_results = gnn_bandit.simulate_procurement_cycle(
        graph_data, num_cycles=1000, base_demand=1200
    )
    
    # Analyze learning progression
    print("\nLEARNING PROGRESSION ANALYSIS:")
    print("-" * 40)
    
    cycle_rewards = [result['average_reward'] for result in simulation_results]
    print(f"Initial average reward: {cycle_rewards[0]:.3f}")
    print(f"Final average reward: {cycle_rewards[-1]:.3f}")
    print(f"Improvement: {((cycle_rewards[-1] - cycle_rewards[0]) / cycle_rewards[0] * 100):.1f}%")
    
    # Show top suppliers after learning
    rankings = bandit_agent.get_supplier_rankings()
    print("\nTOP 10 SUPPLIERS AFTER LEARNING:")
    print("-" * 40)
    for i, ranking in enumerate(rankings[:10]):
        print(f"  {i+1}. {ranking['supplier_id']}: "
              f"Reward={ranking['avg_reward']:.3f}, "
              f"Selections={ranking['selection_count']}")
    
    # Strategy adaptation analysis
    print(f"\n STRATEGY ADAPTATION:")
    print(f"  Final epsilon: {bandit_agent.epsilon:.3f}")
    print(f"  Strategy: {bandit_agent.strategy}")
    
    return gnn_bandit, simulation_results

def real_world_scenario_demo(gnn_bandit, graph_data):
    """Demonstrate real-world scenario like the laptop supply chain example"""
    print("\n" + "-"*60)
    print(" REAL-WORLD SCENARIO: LAPTOP SUPPLY CHAIN")
    print("-"*60)
    
    # Simulate different market conditions
    scenarios = [
        {"name": "Normal Operations", "demand": 1000, "market_stress": 1.0},
        {"name": "High Demand Season", "demand": 2000, "market_stress": 1.2},
        {"name": "Supply Chain Crisis", "demand": 800, "market_stress": 0.7},
        {"name": "ESG Compliance Push", "demand": 1200, "market_stress": 1.1},
        {"name": "Cost Optimization", "demand": 1500, "market_stress": 0.9}
    ]
    
    print(" Testing different market scenarios:")
    
    for scenario in scenarios:
        print(f"\n Scenario: {scenario['name']}")
        print("-" * 30)
        
        # Adjust bandit parameters based on scenario
        if scenario['name'] == "ESG Compliance Push":
            # Increase exploration to find sustainable suppliers
            gnn_bandit.bandit_agent.epsilon = 0.2
            gnn_bandit.bandit_agent.strategy = 'contextual_ucb'
        elif scenario['name'] == "Cost Optimization":
            # Reduce exploration, exploit known good suppliers
            gnn_bandit.bandit_agent.epsilon = 0.05
            gnn_bandit.bandit_agent.strategy = 'ucb'
        
        # Select suppliers for this scenario
        selection_result = gnn_bandit.dynamic_supplier_selection(
            graph_data, scenario['demand'], num_suppliers=3
        )
        
        print(f"  Demand: {scenario['demand']}")
        print(f"  Selected: {[f'SUP_{i:03d}' for i in selection_result['selected_suppliers']]}")
        print(f"  Strategy: {gnn_bandit.bandit_agent.strategy}")
        
        # Simulate rewards based on scenario
        rewards = []
        for supplier_idx in selection_result['selected_suppliers']:
            base_reward = gnn_bandit.bandit_agent.supplier_rewards[supplier_idx]
            scenario_adjusted = base_reward * scenario['market_stress']
            rewards.append(scenario_adjusted)
        
        print(f"  Avg Reward: {np.mean(rewards):.3f}")
        
        # Update bandit with scenario results
        gnn_bandit.bandit_agent.update_rewards(
            selection_result['selected_suppliers'], rewards
        )

def model_inference_demo(model, graph_data, device):
    """Demonstrate model inference capabilities"""
    print("\n" + "-"*60)
    print("MODEL INFERENCE DEMONSTRATION...")
    print("-"*60)
    
    model.eval()
    graph_data = graph_data.to(device)
    
    try:
        with torch.no_grad():
            outputs = model(graph_data.x, graph_data.edge_index, 
                          edge_attr=getattr(graph_data, 'edge_attr', None))

            print(f"Model Outputs Summary:")
            print(f"  Node embeddings shape: {outputs['node_embeddings'].shape}")
            print(f"  Carbon flows predictions: {outputs['carbon_flows'].shape}")
            print(f"  Supplier classifications: {outputs['supplier_classes'].shape}")
            print(f"  Location predictions: {outputs['location_pred'].shape}")
            print(f"  Performance predictions: {outputs['performance_pred'].shape}")
            
            # Show some sample predictions
            print(f"\nSample Predictions:")
            print(f"  Carbon flow range: [{outputs['carbon_flows'].min():.3f}, {outputs['carbon_flows'].max():.3f}]")
            print(f"  Performance range: [{outputs['performance_pred'].min():.3f}, {outputs['performance_pred'].max():.3f}]")
            
            # Classification distribution
            supplier_probs = torch.softmax(outputs['supplier_classes'], dim=1)
            class_distribution = supplier_probs.mean(dim=0)
            print(f"  Supplier class distribution: {class_distribution.cpu().numpy()}")
            
            return outputs
            
    except Exception as e:
        print(f"Model inference failed: {e}")
        return None

def main():
    config = EnhancedGNNConfig()
    print(f"Using config: INPUT_DIM={config.INPUT_DIM}, HIDDEN_DIM={config.HIDDEN_DIM}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading data")
    data_loader = ImprovedDataLoader(config)
    suppliers_df, relationships_df = data_loader.load_enhanced_data()
    
    print(f"Loaded {len(suppliers_df)} suppliers and {len(relationships_df)} relationships")
    
    print("Building supplier graph")
    graph_builder = CarbonGraphBuilder(config)
    graph_data = graph_builder.build_supplier_graph(suppliers_df, relationships_df)
    
    print(f"Graph created: {graph_data.x.shape[0]} nodes, {graph_data.edge_index.shape[1]} edges")
    print(f"Node features shape: {graph_data.x.shape}")
    print(f"Edge features shape: {graph_data.edge_attr.shape if graph_data.edge_attr is not None else 'None'}")
    
    graph_data = add_train_val_masks(graph_data)
    
    print("Initializing enhanced model")
    model = ImprovedCarbonGNN(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Option 1: Train the model first
    train_model = input("\n Do you want to train the GNN model first? (y/n): ").lower().strip() == 'y'
    
    if train_model:
        trainer = EnhancedGNNTrainer(model, config, device=device)
        
        print("Creating data loaders...")
        train_loader, val_loader = create_data_loaders(graph_data, config)
        
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        print("training started...")
        try:
            trainer.train(train_loader, val_loader, epochs=config.EPOCHS)
            print("Training completed...")
        except Exception as e:
            print(f"Training failed with error: {e}")
    
    # Option 2: Load pre-trained model or use current model
    try:
        if train_model:
            checkpoint = torch.load('best_enhanced_gnn_model.pth', map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("model loaded -")
        else:
            print("Using randomly initialized model for demonstration")
        
        model.to(device)
        graph_data = graph_data.to(device)
        
        # Demonstrate Model Inference
        model_outputs = model_inference_demo(model, graph_data, device)
        
        # Demonstrate GNN-Bandit Integration
        gnn_bandit, simulation_results = demonstrate_bandit_integration(model, graph_data, config)
        
        # Demonstrate Real-world Scenarios
        real_world_scenario_demo(gnn_bandit, graph_data)
        
        # Option to save the trained bandit state
        save_bandit = input("\nSave bandit learning state? (y/n): ").lower().strip() == 'y'
        if save_bandit:
            try:
                gnn_bandit.bandit_agent.save_state('trained_bandit_state.pth')
                print("Bandit state saved to 'trained_bandit_state.pth'")
            except Exception as e:
                print(f"Failed to save bandit state: {e}")
        
    except Exception as e:
        print(f"System initialization failed: {e}")

if __name__ == "__main__":
    main()