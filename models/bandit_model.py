import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, deque
import random

class SupplierBanditAgent:
    """
    Multi-Armed Bandit for dynamic supplier selection
    Integrates with GNN embeddings for context-aware decisions
    """
    def __init__(self, config, num_suppliers=500):
        self.config = config
        self.num_suppliers = num_suppliers
        
        # Bandit parameters
        self.epsilon = 0.1  # Exploration rate
        self.alpha = 0.1    # Learning rate
        self.ucb_c = 2.0    # UCB confidence parameter
        
        # State tracking
        self.supplier_counts = np.zeros(num_suppliers)  # Selection counts
        self.supplier_rewards = np.zeros(num_suppliers)  # Average rewards
        self.supplier_total_rewards = np.zeros(num_suppliers)  # Total rewards
        
        # Context-aware components
        self.context_dim = config.OUTPUT_DIM if hasattr(config, 'OUTPUT_DIM') else 128
        self.context_weights = np.random.normal(0, 0.1, (num_suppliers, self.context_dim))
        
        # Recent performance tracking
        self.recent_performance = defaultdict(lambda: deque(maxlen=10))
        
        # Strategy selection
        self.strategy = 'contextual_ucb'  # Options: 'epsilon_greedy', 'ucb', 'contextual_ucb'
        
    def select_suppliers(self, context_embeddings, current_demand, num_select=5):
        """
        Select optimal suppliers based on context and bandit strategy
        
        Args:
            context_embeddings: GNN node embeddings [num_suppliers, embedding_dim]
            current_demand: Current procurement demand
            num_select: Number of suppliers to select
        
        Returns:
            selected_suppliers: List of supplier indices
            selection_probs: Selection probabilities for each supplier
        """
        if self.strategy == 'epsilon_greedy':
            return self._epsilon_greedy_selection(context_embeddings, num_select)
        elif self.strategy == 'ucb':
            return self._ucb_selection(context_embeddings, num_select)
        elif self.strategy == 'contextual_ucb':
            return self._contextual_ucb_selection(context_embeddings, current_demand, num_select)
        else:
            return self._random_selection(num_select)
    
    def _epsilon_greedy_selection(self, context_embeddings, num_select):
        """Epsilon-greedy supplier selection"""
        selected_suppliers = []
        selection_probs = np.zeros(self.num_suppliers)
        
        for _ in range(num_select):
            if random.random() < self.epsilon:
                # Explore: random selection
                available = [i for i in range(self.num_suppliers) if i not in selected_suppliers]
                if available:
                    supplier = random.choice(available)
            else:
                # Exploit: best known supplier
                available_rewards = np.copy(self.supplier_rewards)
                for selected in selected_suppliers:
                    available_rewards[selected] = -np.inf
                supplier = np.argmax(available_rewards)
            
            selected_suppliers.append(supplier)
            selection_probs[supplier] = 1.0
        
        return selected_suppliers, selection_probs / selection_probs.sum()
    
    def _ucb_selection(self, context_embeddings, num_select):
        """Upper Confidence Bound selection"""
        selected_suppliers = []
        selection_probs = np.zeros(self.num_suppliers)
        
        total_selections = np.sum(self.supplier_counts) + 1
        
        for _ in range(num_select):
            ucb_values = np.zeros(self.num_suppliers)
            
            for i in range(self.num_suppliers):
                if i in selected_suppliers:
                    ucb_values[i] = -np.inf
                    continue
                    
                if self.supplier_counts[i] == 0:
                    ucb_values[i] = np.inf  # Select unvisited suppliers first
                else:
                    confidence = self.ucb_c * np.sqrt(np.log(total_selections) / self.supplier_counts[i])
                    ucb_values[i] = self.supplier_rewards[i] + confidence
            
            supplier = np.argmax(ucb_values)
            selected_suppliers.append(supplier)
            selection_probs[supplier] = 1.0
        
        return selected_suppliers, selection_probs / selection_probs.sum()
    
    def _contextual_ucb_selection(self, context_embeddings, current_demand, num_select):
        """Context-aware UCB selection using GNN embeddings"""
        selected_suppliers = []
        selection_probs = np.zeros(self.num_suppliers)
        
        # Convert context embeddings to numpy if needed
        if torch.is_tensor(context_embeddings):
            context_embeddings = context_embeddings.detach().cpu().numpy()
        
        total_selections = np.sum(self.supplier_counts) + 1
        
        for _ in range(num_select):
            contextual_ucb_values = np.zeros(self.num_suppliers)
            
            for i in range(self.num_suppliers):
                if i in selected_suppliers:
                    contextual_ucb_values[i] = -np.inf
                    continue
                
                # Context-aware reward prediction
                if len(context_embeddings.shape) > 1 and i < context_embeddings.shape[0]:
                    context_reward = np.dot(self.context_weights[i], context_embeddings[i])
                else:
                    context_reward = 0
                
                # Demand-based adjustment
                demand_factor = min(1.0, current_demand / 1000.0)  # Normalize demand
                
                if self.supplier_counts[i] == 0:
                    contextual_ucb_values[i] = np.inf
                else:
                    # Combine historical reward with context
                    base_reward = self.supplier_rewards[i] * 0.7 + context_reward * 0.3
                    
                    # UCB confidence interval
                    confidence = self.ucb_c * np.sqrt(np.log(total_selections) / self.supplier_counts[i])
                    
                    # Demand adjustment
                    contextual_ucb_values[i] = (base_reward + confidence) * demand_factor
            
            supplier = np.argmax(contextual_ucb_values)
            selected_suppliers.append(supplier)
            selection_probs[supplier] = 1.0
        
        return selected_suppliers, selection_probs / selection_probs.sum()
    
    def _random_selection(self, num_select):
        """Random baseline selection"""
        selected_suppliers = random.sample(range(self.num_suppliers), num_select)
        selection_probs = np.zeros(self.num_suppliers)
        for supplier in selected_suppliers:
            selection_probs[supplier] = 1.0 / num_select
        return selected_suppliers, selection_probs
    
    def update_rewards(self, selected_suppliers, rewards):
        """
        Update bandit model with observed rewards
        
        Args:
            selected_suppliers: List of supplier indices that were selected
            rewards: List of rewards (carbon efficiency, cost savings, etc.)
        """
        for supplier, reward in zip(selected_suppliers, rewards):
            # Update counts
            self.supplier_counts[supplier] += 1
            
            # Update running average reward
            old_reward = self.supplier_rewards[supplier]
            self.supplier_rewards[supplier] = old_reward + self.alpha * (reward - old_reward)
            
            # Update total rewards
            self.supplier_total_rewards[supplier] += reward
            
            # Update recent performance
            self.recent_performance[supplier].append(reward)
            
            # Update context weights based on reward
            if hasattr(self, 'last_context') and self.last_context is not None:
                if supplier < len(self.context_weights):
                    # Simple gradient update
                    self.context_weights[supplier] += 0.01 * reward * self.last_context[supplier]
    
    def calculate_supplier_reward(self, supplier_data, gnn_outputs):
        """
        Calculate reward for a supplier based on multiple factors
        
        Args:
            supplier_data: Dict with supplier performance metrics
            gnn_outputs: GNN model outputs including predictions
        
        Returns:
            reward: Scalar reward value (higher is better)
        """
        # Carbon efficiency (primary factor)
        carbon_reward = 1.0 - supplier_data.get('carbon_intensity', 0.5)
        
        # Cost efficiency
        cost_reward = supplier_data.get('cost_efficiency', 0.5)
        
        # Performance score
        performance_reward = supplier_data.get('performance_score', 0.5)
        
        # Renewable energy usage
        renewable_reward = supplier_data.get('renewable_percentage', 0.3)
        
        # Delivery reliability (can be added from historical data)
        reliability_reward = supplier_data.get('delivery_reliability', 0.7)
        
        # Weighted combination
        reward = (
            0.3 * carbon_reward +           # Primary: Carbon efficiency
            0.2 * cost_reward +             # Cost considerations
            0.2 * performance_reward +      # Quality/performance
            0.15 * renewable_reward +       # Sustainability
            0.15 * reliability_reward       # Reliability
        )
        
        # Add noise to encourage exploration
        reward += np.random.normal(0, 0.01)
        
        return np.clip(reward, 0, 1)
    
    def get_supplier_rankings(self):
        """Get current supplier rankings based on bandit learning"""
        rankings = []
        for i in range(self.num_suppliers):
            avg_reward = self.supplier_rewards[i] if self.supplier_counts[i] > 0 else 0
            rankings.append({
                'supplier_id': f'SUP_{i:03d}',
                'avg_reward': avg_reward,
                'selection_count': int(self.supplier_counts[i]),
                'total_reward': self.supplier_total_rewards[i],
                'recent_performance': list(self.recent_performance[i]) if i in self.recent_performance else []
            })
        
        # Sort by average reward
        rankings.sort(key=lambda x: x['avg_reward'], reverse=True)
        return rankings
    
    def adapt_strategy(self, performance_window=50):
        """Adapt bandit strategy based on recent performance"""
        if np.sum(self.supplier_counts) < performance_window:
            return  # Not enough data yet
        
        # Calculate exploration vs exploitation balance
        recent_selections = np.sum(self.supplier_counts[-performance_window:]) if len(self.supplier_counts) > performance_window else np.sum(self.supplier_counts)
        unique_suppliers = np.sum(self.supplier_counts > 0)
        
        exploration_ratio = unique_suppliers / self.num_suppliers
        
        # Adjust epsilon based on exploration ratio
        if exploration_ratio < 0.3:  # Too little exploration
            self.epsilon = min(0.3, self.epsilon + 0.05)
        elif exploration_ratio > 0.7:  # Too much exploration
            self.epsilon = max(0.05, self.epsilon - 0.05)
    
    def save_state(self, filepath):
        """Save bandit state for persistence"""
        state = {
            'supplier_counts': self.supplier_counts,
            'supplier_rewards': self.supplier_rewards,
            'supplier_total_rewards': self.supplier_total_rewards,
            'context_weights': self.context_weights,
            'epsilon': self.epsilon,
            'recent_performance': dict(self.recent_performance)
        }
        torch.save(state, filepath)
    
    def load_state(self, filepath):
        """Load bandit state from file"""
        state = torch.load(filepath)
        self.supplier_counts = state['supplier_counts']
        self.supplier_rewards = state['supplier_rewards']
        self.supplier_total_rewards = state['supplier_total_rewards']
        self.context_weights = state['context_weights']
        self.epsilon = state['epsilon']
        self.recent_performance = defaultdict(lambda: deque(maxlen=10), state['recent_performance'])


class GNNBanditIntegration:
    """
    Integration layer between GNN and Bandit models
    """
    def __init__(self, gnn_model, bandit_agent):
        self.gnn_model = gnn_model
        self.bandit_agent = bandit_agent
        
    def dynamic_supplier_selection(self, graph_data, demand_forecast, num_suppliers=5):
        """
        End-to-end supplier selection pipeline
        
        Args:
            graph_data: PyTorch Geometric data object
            demand_forecast: Expected demand volume
            num_suppliers: Number of suppliers to select
        
        Returns:
            selection_results: Dict with selected suppliers and reasoning
        """
        # Get GNN embeddings and predictions
        self.gnn_model.eval()
        with torch.no_grad():
            gnn_outputs = self.gnn_model(graph_data.x, graph_data.edge_index, 
                                       edge_attr=getattr(graph_data, 'edge_attr', None))
        
        # Extract node embeddings for context
        node_embeddings = gnn_outputs['node_embeddings']
        
        # Bandit-based supplier selection
        selected_suppliers, selection_probs = self.bandit_agent.select_suppliers(
            node_embeddings, demand_forecast, num_suppliers
        )
        
        # Prepare results
        results = {
            'selected_suppliers': selected_suppliers,
            'selection_probabilities': selection_probs,
            'gnn_predictions': {
                'carbon_flows': gnn_outputs['carbon_flows'],
                'supplier_classes': gnn_outputs['supplier_classes']
            },
            'demand_forecast': demand_forecast,
            'selection_reasoning': self._generate_selection_reasoning(
                selected_suppliers, node_embeddings, gnn_outputs
            )
        }
        
        return results
    
    def _generate_selection_reasoning(self, selected_suppliers, embeddings, gnn_outputs):
        """Generate human-readable reasoning for supplier selection"""
        reasoning = []
        
        for supplier_idx in selected_suppliers:
            supplier_id = f'SUP_{supplier_idx:03d}'
            
            # Get supplier metrics
            reward_history = self.bandit_agent.recent_performance.get(supplier_idx, [])
            avg_reward = self.bandit_agent.supplier_rewards[supplier_idx]
            selection_count = self.bandit_agent.supplier_counts[supplier_idx]
            
            reason = {
                'supplier_id': supplier_id,
                'selection_count': int(selection_count),
                'average_reward': float(avg_reward),
                'recent_performance': [float(r) for r in reward_history],
                'strategy': self.bandit_agent.strategy,
                'confidence': 'High' if selection_count > 10 else 'Medium' if selection_count > 3 else 'Low'
            }
            
            reasoning.append(reason)
        
        return reasoning
    
    def simulate_procurement_cycle(self, graph_data, num_cycles=10, base_demand=1000):
        """
        Simulate multiple procurement cycles to test bandit learning
        
        Args:
            graph_data: Graph data
            num_cycles: Number of simulation cycles
            base_demand: Base demand level
        
        Returns:
            simulation_results: Results from each cycle
        """
        results = []
        
        for cycle in range(num_cycles):
            # Vary demand over time
            current_demand = base_demand * (1 + 0.2 * np.sin(cycle / 5))
            
            # Select suppliers
            selection_result = self.dynamic_supplier_selection(
                graph_data, current_demand, num_suppliers=5
            )
            
            # Simulate rewards (in real scenario, these would come from actual performance)
            rewards = self._simulate_cycle_rewards(selection_result['selected_suppliers'])
            
            # Update bandit with observed rewards
            self.bandit_agent.update_rewards(selection_result['selected_suppliers'], rewards)
            
            # Adapt strategy
            if cycle % 5 == 0:
                self.bandit_agent.adapt_strategy()
            
            results.append({
                'cycle': cycle,
                'demand': current_demand,
                'selected_suppliers': selection_result['selected_suppliers'],
                'rewards': rewards,
                'average_reward': np.mean(rewards),
                'strategy': self.bandit_agent.strategy,
                'epsilon': self.bandit_agent.epsilon
            })
        
        return results
    
    def _simulate_cycle_rewards(self, selected_suppliers):
        """Simulate realistic rewards for selected suppliers"""
        rewards = []
        for supplier_idx in selected_suppliers:
            # Base reward with some randomness
            base_reward = 0.6 + 0.3 * np.random.random()
            
            # Add supplier-specific performance characteristics
            if supplier_idx < 100:  # "Premium" suppliers
                base_reward += 0.2 * np.random.random()
            elif supplier_idx > 400:  # "Budget" suppliers
                base_reward -= 0.1 * np.random.random()
            
            # Time-based variations (simulate market changes)
            seasonal_factor = 1 + 0.1 * np.sin(supplier_idx / 50)
            
            final_reward = np.clip(base_reward * seasonal_factor, 0, 1)
            rewards.append(final_reward)
        
        return rewards