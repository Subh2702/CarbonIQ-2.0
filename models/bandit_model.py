import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, deque
import random

class SupplierBanditAgent:
    def __init__(self, config, num_suppliers=500):
        self.config = config
        self.num_suppliers = num_suppliers
        
        self.epsilon = 0.1  # Exploration rate
        self.alpha = 0.1    # Learning rate
        self.ucb_c = 2.0    # UCB confidence parameter
        
        # State set karna
        self.supplier_counts = np.zeros(num_suppliers)  
        self.supplier_rewards = np.zeros(num_suppliers) 
        self.supplier_total_rewards = np.zeros(num_suppliers) 
        
        # Context dimensions set karna
        self.context_dim = config.OUTPUT_DIM if hasattr(config, 'OUTPUT_DIM') else 128
        self.context_weights = np.random.normal(0, 0.1, (num_suppliers, self.context_dim))
        
        # Recent performance tracking
        self.recent_performance = defaultdict(lambda: deque(maxlen=10))
        
        # FIXED: Last context ko properly track karna
        self.last_context = None
        self.last_selected_suppliers = None
        
        self.strategy = 'contextual_ucb'
        
    def select_suppliers(self, context_embeddings, current_demand, num_select=5):
        # FIXED: Context ko store karna for later use
        if torch.is_tensor(context_embeddings):
            self.last_context = context_embeddings.detach().cpu().numpy()
        else:
            self.last_context = np.array(context_embeddings)
        
        if self.strategy == 'epsilon_greedy':
            selected, probs = self._epsilon_greedy_selection(context_embeddings, num_select)
        elif self.strategy == 'ucb':
            selected, probs = self._ucb_selection(context_embeddings, num_select)
        elif self.strategy == 'contextual_ucb':
            selected, probs = self._contextual_ucb_selection(context_embeddings, current_demand, num_select)
        else:
            selected, probs = self._random_selection(num_select)
        
        # FIXED: Selected suppliers ko bhi store karna
        self.last_selected_suppliers = selected
        return selected, probs
    
    def _epsilon_greedy_selection(self, context_embeddings, num_select):
        selected_suppliers = []
        selection_probs = np.zeros(self.num_suppliers)
        
        # Convert context if needed
        if torch.is_tensor(context_embeddings):
            context_array = context_embeddings.detach().cpu().numpy()
        else:
            context_array = np.array(context_embeddings)
        
        for _ in range(num_select):
            if random.random() < self.epsilon:
                # Explore: random selection
                available = [i for i in range(self.num_suppliers) if i not in selected_suppliers]
                if available:
                    supplier = random.choice(available)
            else:
                # Exploit: context-aware best supplier
                available_rewards = np.copy(self.supplier_rewards)
                
                # Add context reward to base reward
                for i in range(self.num_suppliers):
                    if i not in selected_suppliers and i < len(context_array):
                        context_reward = np.dot(self.context_weights[i], context_array[i])
                        available_rewards[i] += 0.3 * context_reward
                    elif i in selected_suppliers:
                        available_rewards[i] = -np.inf
                
                supplier = np.argmax(available_rewards)
            
            selected_suppliers.append(supplier)
            selection_probs[supplier] = 1.0
        
        return selected_suppliers, selection_probs / max(selection_probs.sum(), 1e-8)
    
    def _ucb_selection(self, context_embeddings, num_select):
        selected_suppliers = []
        selection_probs = np.zeros(self.num_suppliers)
        
        # Convert context if needed
        if torch.is_tensor(context_embeddings):
            context_array = context_embeddings.detach().cpu().numpy()
        else:
            context_array = np.array(context_embeddings)
        
        total_selections = np.sum(self.supplier_counts) + 1
        
        for _ in range(num_select):
            ucb_values = np.zeros(self.num_suppliers)
            
            for i in range(self.num_suppliers):
                if i in selected_suppliers:
                    ucb_values[i] = -np.inf
                    continue
                    
                if self.supplier_counts[i] == 0:
                    ucb_values[i] = np.inf
                else:
                    confidence = self.ucb_c * np.sqrt(np.log(total_selections) / self.supplier_counts[i])
                    base_reward = self.supplier_rewards[i]
                    
                    # FIXED: Context reward properly add karna
                    context_reward = 0
                    if i < len(context_array):
                        context_reward = np.dot(self.context_weights[i], context_array[i])
                    
                    ucb_values[i] = base_reward + 0.3 * context_reward + confidence
            
            supplier = np.argmax(ucb_values)
            selected_suppliers.append(supplier)
            selection_probs[supplier] = 1.0
        
        return selected_suppliers, selection_probs / max(selection_probs.sum(), 1e-8)
    
    def _contextual_ucb_selection(self, context_embeddings, current_demand, num_select):
        selected_suppliers = []
        selection_probs = np.zeros(self.num_suppliers)
        
        if torch.is_tensor(context_embeddings):
            context_embeddings = context_embeddings.detach().cpu().numpy()
        
        total_selections = np.sum(self.supplier_counts) + 1
        
        for _ in range(num_select):
            contextual_ucb_values = np.zeros(self.num_suppliers)
            
            for i in range(self.num_suppliers):
                if i in selected_suppliers:
                    contextual_ucb_values[i] = -np.inf
                    continue
                
                # Context reward calculate karna
                context_reward = 0
                if len(context_embeddings.shape) > 1 and i < context_embeddings.shape[0]:
                    context_reward = np.dot(self.context_weights[i], context_embeddings[i])
                elif len(context_embeddings.shape) == 1:
                    # If single context vector, use it for all suppliers
                    context_reward = np.dot(self.context_weights[i], context_embeddings)
                
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
        
        return selected_suppliers, selection_probs / max(selection_probs.sum(), 1e-8)
    
    def _random_selection(self, num_select):
        selected_suppliers = random.sample(range(self.num_suppliers), min(num_select, self.num_suppliers))
        selection_probs = np.zeros(self.num_suppliers)
        for supplier in selected_suppliers:
            selection_probs[supplier] = 1.0 / len(selected_suppliers)
        return selected_suppliers, selection_probs
    
    def update_rewards(self, selected_suppliers, rewards):
        """FIXED: Proper context usage in reward updates"""
        for i, (supplier, reward) in enumerate(zip(selected_suppliers, rewards)):
            # Update counts
            self.supplier_counts[supplier] += 1
            
            # Update running average reward
            old_reward = self.supplier_rewards[supplier]
            self.supplier_rewards[supplier] = old_reward + self.alpha * (reward - old_reward)
            
            # Update total rewards
            self.supplier_total_rewards[supplier] += reward
            
            # Update recent performance
            self.recent_performance[supplier].append(reward)
            
            # FIXED: Context weights ko properly update karna
            if self.last_context is not None:
                # Get context for this specific supplier
                if len(self.last_context.shape) > 1:
                    # Multiple contexts - use supplier-specific context
                    if supplier < self.last_context.shape[0]:
                        supplier_context = self.last_context[supplier]
                    else:
                        # Use mean context if supplier index out of bounds
                        supplier_context = np.mean(self.last_context, axis=0)
                else:
                    # Single context vector - use for all
                    supplier_context = self.last_context
                
                # Ensure context dimension matches
                if len(supplier_context) == self.context_weights.shape[1]:
                    # Gradient-based update with reward signal
                    reward_error = reward - 0.5  # Center around 0
                    learning_rate = 0.01
                    
                    # Update context weights
                    self.context_weights[supplier] += learning_rate * reward_error * supplier_context
                    
                    # Add L2 regularization to prevent overfitting
                    self.context_weights[supplier] *= 0.999
    
    def get_context_insights(self):
        """New method to understand what contexts are being learned"""
        insights = {}
        
        # Top performing suppliers
        top_suppliers = np.argsort(self.supplier_rewards)[-10:][::-1]
        
        insights['top_suppliers'] = []
        for supplier in top_suppliers:
            if self.supplier_counts[supplier] > 0:
                insights['top_suppliers'].append({
                    'supplier_id': f'SUP_{supplier:03d}',
                    'avg_reward': float(self.supplier_rewards[supplier]),
                    'count': int(self.supplier_counts[supplier]),
                    'context_weight_norm': float(np.linalg.norm(self.context_weights[supplier]))
                })
        
        # Context weight statistics
        insights['context_stats'] = {
            'mean_weight_norm': float(np.mean([np.linalg.norm(w) for w in self.context_weights])),
            'std_weight_norm': float(np.std([np.linalg.norm(w) for w in self.context_weights])),
            'context_dimension': int(self.context_dim)
        }
        
        return insights
    
    def calculate_supplier_reward(self, supplier_data):
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
        rankings = []
        for i in range(self.num_suppliers):
            avg_reward = self.supplier_rewards[i] if self.supplier_counts[i] > 0 else 0
            rankings.append({
                'supplier_id': f'SUP_{i:03d}',
                'avg_reward': avg_reward,
                'selection_count': int(self.supplier_counts[i]),
                'total_reward': self.supplier_total_rewards[i],
                'recent_performance': list(self.recent_performance[i]) if i in self.recent_performance else [],
                'context_weight_norm': float(np.linalg.norm(self.context_weights[i]))
            })
        rankings.sort(key=lambda x: x['avg_reward'], reverse=True)
        return rankings
    
    def adapt_strategy(self, performance_window=50):
        if np.sum(self.supplier_counts) < performance_window:
            return  # data kam hai
        
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
        state = {
            'supplier_counts': self.supplier_counts,
            'supplier_rewards': self.supplier_rewards,
            'supplier_total_rewards': self.supplier_total_rewards,
            'context_weights': self.context_weights,
            'epsilon': self.epsilon,
            'recent_performance': dict(self.recent_performance),
            'last_context': self.last_context,
            'last_selected_suppliers': self.last_selected_suppliers
        }
        torch.save(state, filepath)
    
    def load_state(self, filepath):
        state = torch.load(filepath)
        self.supplier_counts = state['supplier_counts']
        self.supplier_rewards = state['supplier_rewards']
        self.supplier_total_rewards = state['supplier_total_rewards']
        self.context_weights = state['context_weights']
        self.epsilon = state['epsilon']
        self.recent_performance = defaultdict(lambda: deque(maxlen=10), state['recent_performance'])
        self.last_context = state.get('last_context', None)
        self.last_selected_suppliers = state.get('last_selected_suppliers', None)


class GNNBanditIntegration:
    def __init__(self, gnn_model, bandit_agent):
        self.gnn_model = gnn_model
        self.bandit_agent = bandit_agent
        
    def dynamic_supplier_selection(self, graph_data, demand_forecast, num_suppliers=5):
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
                selected_suppliers
            )
        }
        
        return results
    
    def _generate_selection_reasoning(self, selected_suppliers):
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