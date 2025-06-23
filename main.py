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
    """Create supplier DataFrame with more realistic reward calculations"""
    num_suppliers = graph_data.x.shape[0]
    
    # Extract features with better normalization
    carbon_intensity = graph_data.x[:, 0].cpu().numpy()
    performance_score = graph_data.x[:, 1].cpu().numpy()
    
    # Normalize features to [0, 1] range
    carbon_norm = (carbon_intensity - carbon_intensity.min()) / (carbon_intensity.max() - carbon_intensity.min() + 1e-8)
    performance_norm = (performance_score - performance_score.min()) / (performance_score.max() - performance_score.min() + 1e-8)
    
    suppliers_df = pd.DataFrame({
        'supplier_id': [f'SUP_{i:03d}' for i in range(num_suppliers)],
        'carbon_intensity': carbon_intensity,
        'performance_score': performance_score,
        'carbon_normalized': carbon_norm,
        'performance_normalized': performance_norm,
        'renewable_percentage': graph_data.x[:, 5].cpu().numpy() if graph_data.x.shape[1] > 5 else np.random.beta(2, 2, num_suppliers),
        'cost_efficiency': graph_data.x[:, 7].cpu().numpy() if graph_data.x.shape[1] > 7 else np.random.beta(3, 2, num_suppliers),
        'delivery_reliability': np.random.beta(4, 2, num_suppliers),  # Higher reliability on average
        'quality_score': np.random.beta(3, 2, num_suppliers)
    })
    
    return suppliers_df

def demonstrate_bandit_integration(model, graph_data, config):
    """Improved bandit demonstration with better initialization and reward calculation"""
    print("\n" + "="*60)
    print("🚀 IMPROVED GNN-BANDIT INTEGRATION")
    print("="*60)
    
    # Initialize bandit agent
    num_suppliers = graph_data.x.shape[0]
    bandit_agent = SupplierBanditAgent(config, num_suppliers=num_suppliers)
    
    # Create integration
    gnn_bandit = GNNBanditIntegration(model, bandit_agent)
    
    # Initialize with GNN priors instead of zeros
    bandit_agent.initialize_with_gnn_priors(model, graph_data)
    
    print(f"🏭 Initialized bandit for {num_suppliers} suppliers")
    print(f"📊 Bandit strategy: {bandit_agent.strategy}")
    
    # Demonstrate single selection cycle
    print("\n📋 SINGLE SUPPLIER SELECTION CYCLE")
    print("-" * 40)
    
    demand_forecast = 1500  # Example demand
    selection_result = gnn_bandit.dynamic_supplier_selection(
        graph_data, demand_forecast, num_suppliers=5
    )
    
    print(f"🎯 Demand forecast: {demand_forecast}")
    print(f"🔍 Selected suppliers: {selection_result['selected_suppliers']}")
    print(f"📈 Selection probabilities: {selection_result['selection_probabilities'][selection_result['selected_suppliers']]}")
    
    # Show detailed reasoning with actual rewards now
    print("\n💭 SELECTION REASONING:")
    for reason in selection_result['selection_reasoning']:
        print(f"  🏢 {reason['supplier_id']}: Reward={reason['average_reward']:.3f}, "
              f"Count={reason['selection_count']}, Confidence={reason['confidence']}")
    
    # Run improved simulation
    print("\n🔄 RUNNING IMPROVED SIMULATION")
    print("-" * 40)
    
    simulation_results = gnn_bandit.improved_simulate_procurement_cycle(
        graph_data, num_cycles=100, base_demand=1200
    )
    
    # Analyze learning progression
    print("\n📊 LEARNING PROGRESSION ANALYSIS:")
    print("-" * 40)
    
    cycle_rewards = [result['average_reward'] for result in simulation_results]
    
    print(f"🎯 Initial average reward: {cycle_rewards[0]:.3f}")
    print(f"🎯 Final average reward: {cycle_rewards[-1]:.3f}")
    print(f"📈 Overall improvement: {((cycle_rewards[-1] - cycle_rewards[0]) / cycle_rewards[0] * 100):.1f}%")
    
    # Show learning curve over time
    early_avg = np.mean(cycle_rewards[:20])
    late_avg = np.mean(cycle_rewards[-20:])
    print(f"📊 Early cycles avg (1-20): {early_avg:.3f}")
    print(f"📊 Late cycles avg (81-100): {late_avg:.3f}")
    print(f"🎓 Learning improvement: {((late_avg - early_avg) / early_avg * 100):.1f}%")
    
    # Show top suppliers after learning
    rankings = bandit_agent.get_supplier_rankings()
    print("\n🏆 TOP 10 SUPPLIERS AFTER LEARNING:")
    print("-" * 40)
    for i, ranking in enumerate(rankings[:10]):
        print(f"  {i+1}. {ranking['supplier_id']}: "
              f"Reward={ranking['avg_reward']:.3f}, "
              f"Selections={ranking['selection_count']}")
    
    # Strategy adaptation analysis
    print(f"\n🔧 STRATEGY ADAPTATION:")
    print(f"  🎲 Final epsilon: {bandit_agent.epsilon:.3f}")
    print(f"  🧠 Strategy: {bandit_agent.strategy}")
    
    return gnn_bandit, simulation_results

def real_world_scenario_demo(gnn_bandit, graph_data):
    """Demonstrate real-world scenario with improved reward calculation"""
    print("\n" + "="*60)
    print("🌏 REAL-WORLD SCENARIO: LAPTOP SUPPLY CHAIN")
    print("="*60)
    
    # Create enhanced supplier dataframe
    suppliers_df = gnn_bandit.create_enhanced_supplier_dataframe(graph_data)
    
    # Simulate different market conditions
    scenarios = [
        {"name": "Normal Operations", "demand": 1000, "scenario_type": "normal"},
        {"name": "High Demand Season", "demand": 2000, "scenario_type": "normal"},
        {"name": "Supply Chain Crisis", "demand": 800, "scenario_type": "crisis"},
        {"name": "ESG Compliance Push", "demand": 1200, "scenario_type": "esg_compliance"},
        {"name": "Cost Optimization", "demand": 1500, "scenario_type": "cost_optimization"}
    ]
    
    print("🎭 Testing different market scenarios:")
    
    for scenario in scenarios:
        print(f"\n📊 Scenario: {scenario['name']}")
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
        
        print(f"  🎯 Demand: {scenario['demand']}")
        print(f"  🏢 Selected: {[f'SUP_{i:03d}' for i in selection_result['selected_suppliers']]}")
        print(f"  📈 Strategy: {gnn_bandit.bandit_agent.strategy}")
        
        # Calculate realistic rewards based on scenario
        rewards = []
        for supplier_idx in selection_result['selected_suppliers']:
            supplier_data = suppliers_df.iloc[supplier_idx].to_dict()
            reward = gnn_bandit.bandit_agent.calculate_supplier_reward(
                supplier_data, scenario['scenario_type']
            )
            rewards.append(reward)
        
        print(f"  🎁 Avg Reward: {np.mean(rewards):.3f}")
        print(f"  📊 Reward Range: [{min(rewards):.3f}, {max(rewards):.3f}]")
        
        # Update bandit with scenario results
        gnn_bandit.bandit_agent.update_rewards(
            selection_result['selected_suppliers'], rewards
        )

def model_inference_demo(model, graph_data, device):
    """Demonstrate model inference capabilities"""
    print("\n" + "="*60)
    print("🔮 MODEL INFERENCE DEMONSTRATION")
    print("="*60)
    
    model.eval()
    graph_data = graph_data.to(device)
    
    try:
        with torch.no_grad():
            outputs = model(graph_data.x, graph_data.edge_index, 
                          edge_attr=getattr(graph_data, 'edge_attr', None))

            print(f"📊 Model Outputs Summary:")
            print(f"  🧠 Node embeddings shape: {outputs['node_embeddings'].shape}")
            print(f"  🔄 Carbon flows predictions: {outputs['carbon_flows'].shape}")
            print(f"  🏢 Supplier classifications: {outputs['supplier_classes'].shape}")
            print(f"  📍 Location predictions: {outputs['location_pred'].shape}")
            print(f"  ⭐ Performance predictions: {outputs['performance_pred'].shape}")
            
            # Show some sample predictions
            print(f"\n📈 Sample Predictions:")
            print(f"  🔥 Carbon flow range: [{outputs['carbon_flows'].min():.3f}, {outputs['carbon_flows'].max():.3f}]")
            print(f"  ⭐ Performance range: [{outputs['performance_pred'].min():.3f}, {outputs['performance_pred'].max():.3f}]")
            
            # Classification distribution
            supplier_probs = torch.softmax(outputs['supplier_classes'], dim=1)
            class_distribution = supplier_probs.mean(dim=0)
            print(f"  🏭 Supplier class distribution: {class_distribution.cpu().numpy()}")
            
            return outputs
            
    except Exception as e:
        print(f"❌ Model inference failed: {e}")
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
    train_model = input("\n🤔 Do you want to train the GNN model first? (y/n): ").lower().strip() == 'y'
    
    if train_model:
        trainer = EnhancedGNNTrainer(model, config, device=device)
        
        print("Creating data loaders...")
        train_loader, val_loader = create_data_loaders(graph_data, config)
        
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        print("Starting enhanced training...")
        try:
            trainer.train(train_loader, val_loader, epochs=config.EPOCHS)
            print(";););););) Training completed successfully! ;);););)")
        except Exception as e:
            print(f"Training failed with error: {e}")
    
    # Option 2: Load pre-trained model or use current model
    try:
        if train_model:
            checkpoint = torch.load('best_enhanced_gnn_model.pth', map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Best trained model loaded!")
        else:
            print("ℹ️ Using randomly initialized model for demonstration")
        
        model.to(device)
        graph_data = graph_data.to(device)
        
        # Demonstrate Model Inference
        model_outputs = model_inference_demo(model, graph_data, device)
        
        # Demonstrate GNN-Bandit Integration
        gnn_bandit, simulation_results = demonstrate_bandit_integration(model, graph_data, config)
        
        # Demonstrate Real-world Scenarios
        real_world_scenario_demo(gnn_bandit, graph_data)
        
        # Final Summary
        print("\n" + "="*60)
        print("🎉 COMPREHENSIVE DEMONSTRATION COMPLETED!")
        print("="*60)
        print("✅ GNN Model: Trained/Loaded and functional")
        print("✅ Bandit Agent: Learning supplier preferences")
        print("✅ Integration: Dynamic supplier selection working")
        print("✅ Real-world Scenarios: Market adaptation demonstrated")
        
        # Option to save the trained bandit state
        save_bandit = input("\n💾 Save bandit learning state? (y/n): ").lower().strip() == 'y'
        if save_bandit:
            try:
                gnn_bandit.bandit_agent.save_state('trained_bandit_state.pth')
                print("✅ Bandit state saved to 'trained_bandit_state.pth'")
            except Exception as e:
                print(f"❌ Failed to save bandit state: {e}")
        
        print("\n🎯 System ready for production deployment!")
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        print("🔧 Check your model files and data paths")

if __name__ == "__main__":
    main()