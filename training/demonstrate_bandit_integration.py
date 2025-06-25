import logging
from typing import Dict, Any
from models.bandit_model import SupplierBanditAgent, GNNBanditIntegration
import pandas as pd
import numpy as np

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


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DemonstrateBandit:
    def __init__(self, config):
        self.config = config

    def demonstrate_bandit_integration(self, model, graph_data, demand_forecast=1500, num_suppliers=5, num_cycles=1000, base_demand=1200) -> Dict[str, Any]:
        
        logger.info("\n%s", "-" * 60)
        logger.info("DEMONSTRATING GNN-BANDIT INTEGRATION")
        logger.info("%s", "-" * 60)

        # Validate inputs
        if not hasattr(graph_data, 'x') or graph_data.x is None:
            raise ValueError("Invalid graph_data: Must contain node features (x).")
        if model is None:
            raise ValueError("Invalid model: Must provide a valid GNN model.")

        try:
            # Initialize bandit agent
            num_suppliers_total = graph_data.x.shape[0]
            bandit_agent = SupplierBanditAgent(self.config, num_suppliers=num_suppliers_total)

            # Create integration
            gnn_bandit = GNNBanditIntegration(model, bandit_agent)

            # Create supplier data for reward calculation
            suppliers_df = create_supplier_dataframe(graph_data)

            logger.info("Initialized bandit for %d suppliers", num_suppliers_total)
            logger.info("Bandit strategy: %s", bandit_agent.strategy)

            # Demonstrate single selection cycle
            logger.info("\nSINGLE SUPPLIER SELECTION CYCLE")
            logger.info("%s", "-" * 40)

            selection_result = gnn_bandit.dynamic_supplier_selection(
                graph_data, demand_forecast, num_suppliers=num_suppliers
            )
            
            logger.info("Demand forecast: %d", demand_forecast)
            logger.info("Selected suppliers: %s", selection_result['selected_suppliers'])
            logger.info("Selection probabilities: %s", selection_result['selection_probabilities'][selection_result['selected_suppliers']])

            # Show detailed reasoning
            logger.info("\nSELECTION REASONING:")
            for reason in selection_result['selection_reasoning']:
                logger.info(
                    '  %s: Reward=%.3f, Count=%d, Confidence=%s',
                    reason['supplier_id'], reason['average_reward'], reason['selection_count'], reason['confidence']
                )

            # Simulate procurement cycles
            logger.info("\nSIMULATING PROCUREMENT CYCLES")
            logger.info("%s", "-" * 40)

            simulation_results = gnn_bandit.simulate_procurement_cycle(
                graph_data, num_cycles=num_cycles, base_demand=base_demand
            )

            # Analyze learning progression
            logger.info("\nLEARNING PROGRESSION ANALYSIS:")
            logger.info("%s", "-" * 40)

            cycle_rewards = [result['average_reward'] for result in simulation_results]
            logger.info("Initial average reward: %.3f", cycle_rewards[0])
            logger.info("Final average reward: %.3f", cycle_rewards[-1])
            improvement = ((cycle_rewards[-1] - cycle_rewards[0]) / cycle_rewards[0] * 100) if cycle_rewards[0] != 0 else 0.0
            logger.info("Improvement: %.1f%%", improvement)

            # Show top suppliers after learning
            rankings = bandit_agent.get_supplier_rankings()
            logger.info("\nTOP 10 SUPPLIERS AFTER LEARNING:")
            logger.info("%s", "-" * 40)
            for i, ranking in enumerate(rankings[:10]):
                logger.info(
                    "  %d. %s: Reward=%.3f, Selections=%d",
                    i + 1, ranking['supplier_id'], ranking['avg_reward'], ranking['selection_count']
                )

            # Strategy adaptation analysis
            logger.info("\nSTRATEGY ADAPTATION:")
            logger.info("  Final epsilon: %.3f", bandit_agent.epsilon)
            logger.info("  Strategy: %s", bandit_agent.strategy)

            return {
                'gnn_bandit': gnn_bandit,
                'selection_result': selection_result,
                'simulation_results': simulation_results,
                'rankings': rankings
            }

        except Exception as e:
            logger.error("Error in bandit integration demonstration: %s", str(e))
            raise
