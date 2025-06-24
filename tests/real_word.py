import numpy as np
import logging
from typing import Dict, List

# Configure logging for better output management
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestingForRealWorld:
    def __init__(self, gnn_bandit, graph_data):
        """
        Initialize the real-world scenario tester for supply chain simulation.

        Args:
            gnn_bandit: GNNBanditIntegration object for supplier selection.
            graph_data: PyTorch Geometric Data object containing supplier graph.
        
        Raises:
            ValueError: If gnn_bandit or graph_data is invalid.
        """
        if not hasattr(gnn_bandit, 'bandit_agent') or not hasattr(gnn_bandit.bandit_agent, 'supplier_rewards'):
            raise ValueError("Invalid gnn_bandit: Must have a valid bandit_agent with supplier_rewards.")
        if not hasattr(graph_data, 'x') or graph_data.x is None:
            raise ValueError("Invalid graph_data: Must contain node features (x).")
        
        self.gnn_bandit = gnn_bandit
        self.graph_data = graph_data
        
        # Define market scenarios (configurable)
        self.scenarios = [
            {"name": "Normal Operations", "demand": 1000, "market_stress": 1.0},
            {"name": "High Demand Season", "demand": 2000, "market_stress": 1.2},
            {"name": "Supply Chain Crisis", "demand": 800, "market_stress": 0.7},
            {"name": "ESG Compliance Push", "demand": 1200, "market_stress": 1.1},
            {"name": "Cost Optimization", "demand": 1500, "market_stress": 0.9}
        ]

    def run_scenarios(self):
        """
        Run simulations for different market scenarios in the laptop supply chain.

        Returns:
            List[Dict]: Results of each scenario with metrics.
        """
        logger.info("\n%s", "-" * 60)
        logger.info("REAL-WORLD SCENARIO: LAPTOP SUPPLY CHAIN")
        logger.info("%s", "-" * 60)
        logger.info("Testing different market scenarios:")

        results = []

        for scenario in self.scenarios:
            logger.info("\nScenario: %s", scenario['name'])
            logger.info("%s", "-" * 30)

            try:
                # Adjust bandit parameters based on scenario
                self._adjust_bandit_strategy(scenario)

                # Select suppliers for this scenario
                selection_result = self.gnn_bandit.dynamic_supplier_selection(
                    self.graph_data, scenario['demand'], num_suppliers=3
                )

                # Log selection details
                logger.info("  Demand: %d", scenario['demand'])
                logger.info("  Selected: %s", [f'SUP_{i:03d}' for i in selection_result['selected_suppliers']])
                logger.info("  Strategy: %s", self.gnn_bandit.bandit_agent.strategy)

                # Simulate rewards based on scenario
                rewards = self._simulate_rewards(selection_result['selected_suppliers'], scenario['market_stress'])
                avg_reward = np.mean(rewards) if rewards else 0.0

                logger.info("  Avg Reward: %.3f", avg_reward)

                # Update bandit with scenario results
                self.gnn_bandit.bandit_agent.update_rewards(
                    selection_result['selected_suppliers'], rewards
                )

                # Store results
                results.append({
                    'scenario': scenario['name'],
                    'demand': scenario['demand'],
                    'selected_suppliers': selection_result['selected_suppliers'],
                    'avg_reward': avg_reward,
                    'strategy': self.gnn_bandit.bandit_agent.strategy
                })

            except Exception as e:
                logger.error("Error in scenario %s: %s", scenario['name'], str(e))
                results.append({
                    'scenario': scenario['name'],
                    'error': str(e)
                })

        return results

    def _adjust_bandit_strategy(self, scenario: Dict):
        """
        Adjust bandit strategy based on the scenario.

        Args:
            scenario: Dictionary containing scenario parameters (name, demand, market_stress).
        """
        if scenario['name'] == "ESG Compliance Push":
            # Increase exploration for sustainable suppliers
            self.gnn_bandit.bandit_agent.epsilon = 0.2
            self.gnn_bandit.bandit_agent.strategy = 'contextual_ucb'
        elif scenario['name'] == "Cost Optimization":
            # Reduce exploration, exploit known good suppliers
            self.gnn_bandit.bandit_agent.epsilon = 0.05
            self.gnn_bandit.bandit_agent.strategy = 'ucb'
        else:
            # Default strategy
            self.gnn_bandit.bandit_agent.epsilon = 0.1
            self.gnn_bandit.bandit_agent.strategy = 'contextual_ucb'

    def _simulate_rewards(self, selected_suppliers: List[int], market_stress: float) -> List[float]:
        """
        Simulate rewards for selected suppliers based on market conditions.

        Args:
            selected_suppliers: List of supplier indices.
            market_stress: Market stress factor to adjust rewards.

        Returns:
            List[float]: Simulated rewards for selected suppliers.
        """
        rewards = []
        num_suppliers = len(self.gnn_bandit.bandit_agent.supplier_rewards)

        for supplier_idx in selected_suppliers:
            if supplier_idx >= num_suppliers:
                logger.warning("Invalid supplier index %d, skipping.", supplier_idx)
                continue
            try:
                base_reward = self.gnn_bandit.bandit_agent.supplier_rewards[supplier_idx]
                scenario_adjusted = base_reward * market_stress
                rewards.append(scenario_adjusted)
            except IndexError:
                logger.error("Index error for supplier %d, using default reward.", supplier_idx)
                rewards.append(0.0)

        return rewards