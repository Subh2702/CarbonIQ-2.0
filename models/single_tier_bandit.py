import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, deque
import random
import copy

class SingleTierSupplierOptimizer:
    """
    Single-tier supplier optimization ke liye bandit agent
    Goal: Carbon-heavy suppliers ko best alternatives se replace karna
    """
    def __init__(self, config, num_suppliers=500):
        self.config = config
        self.num_suppliers = num_suppliers
        
        # Original bandit parameters
        self.epsilon = 0.1
        self.alpha = 0.1
        self.ucb_c = 2.0
        
        # Supplier performance tracking
        self.supplier_counts = np.zeros(num_suppliers)
        self.supplier_rewards = np.zeros(num_suppliers)
        self.supplier_carbon_scores = np.zeros(num_suppliers)  # Carbon efficiency scores
        
        # Context dimensions
        self.context_dim = config.OUTPUT_DIM if hasattr(config, 'OUTPUT_DIM') else 128
        self.context_weights = np.random.normal(0, 0.1, (num_suppliers, self.context_dim))
        
        # NEW: Replacement tracking
        self.replacement_history = {}  # {original_supplier: [replacements_tried]}
        self.replacement_rewards = {}  # {(original, replacement): reward}
        
        # Carbon emission thresholds
        self.carbon_threshold = 0.7  # Suppliers with carbon > this need replacement
        
    def identify_high_carbon_suppliers(self, graph_data, gnn_model, threshold=None):
        """
        GNN se high carbon emission wale suppliers identify karo
        """
        threshold = threshold or self.carbon_threshold
        
        # GNN prediction
        gnn_model.eval()
        with torch.no_grad():
            outputs = gnn_model(graph_data.x, graph_data.edge_index, 
                              edge_attr=getattr(graph_data, 'edge_attr', None))
        
        # Carbon flows extract karo
        carbon_flows = outputs['carbon_flows'].cpu().numpy()
        
        # High carbon edges identify karo
        high_carbon_edges = []
        for i, flow in enumerate(carbon_flows):
            if flow > threshold:
                source_idx = graph_data.edge_index[0][i].item()
                target_idx = graph_data.edge_index[1][i].item()
                high_carbon_edges.append((source_idx, target_idx, flow))
        
        # Suppliers ko count karo jo high carbon edges mein involved hain
        supplier_carbon_count = defaultdict(int)
        supplier_carbon_total = defaultdict(float)
        
        for source, target, flow in high_carbon_edges:
            supplier_carbon_count[source] += 1
            supplier_carbon_count[target] += 1
            supplier_carbon_total[source] += flow
            supplier_carbon_total[target] += flow
        
        # High carbon suppliers ki list banao
        high_carbon_suppliers = []
        for supplier, count in supplier_carbon_count.items():
            avg_carbon = supplier_carbon_total[supplier] / count
            if avg_carbon > threshold:
                high_carbon_suppliers.append((supplier, avg_carbon, count))
        
        # Sort by carbon emission (descending)
        high_carbon_suppliers.sort(key=lambda x: x[1], reverse=True)
        
        return high_carbon_suppliers[:10]  # Top 10 worst suppliers
    
    def find_best_replacement(self, gnn_model, graph_data, target_supplier_idx, candidate_suppliers=None):
        """
        Target supplier ke liye best replacement find karo
        """
        if candidate_suppliers is None:
            # All other suppliers as candidates
            candidate_suppliers = [i for i in range(self.num_suppliers) 
                                 if i != target_supplier_idx]
        
        best_replacement = None
        best_carbon_reduction = 0
        best_reward = -float('inf')
        
        # Original carbon emission calculate karo
        original_carbon = self._calculate_supplier_carbon_impact(
            gnn_model, graph_data, target_supplier_idx
        )
        
        # Har candidate ke liye test karo
        for candidate in candidate_suppliers[:20]:  # Top 20 candidates test karo
            # Replacement reward calculate karo
            replacement_reward = self._simulate_replacement_reward(
                gnn_model, graph_data, target_supplier_idx, candidate
            )
            
            # Carbon reduction calculate karo
            new_carbon = self._calculate_supplier_carbon_impact(
                gnn_model, graph_data, candidate
            )
            carbon_reduction = original_carbon - new_carbon
            
            # Combined score (carbon reduction + other factors)
            combined_score = (
                0.6 * carbon_reduction +  # Primary: Carbon reduction
                0.4 * replacement_reward  # Secondary: Other factors
            )
            
            if combined_score > best_reward:
                best_reward = combined_score
                best_replacement = candidate
                best_carbon_reduction = carbon_reduction
        
        return {
            'best_replacement': best_replacement,
            'carbon_reduction': best_carbon_reduction,
            'combined_score': best_reward,
            'original_carbon': original_carbon
        }
    
    def _calculate_supplier_carbon_impact(self, gnn_model, graph_data, supplier_idx):
        """
        Specific supplier ka carbon impact calculate karo
        """
        # Find all edges involving this supplier
        supplier_edges = []
        for i in range(graph_data.edge_index.shape[1]):
            source = graph_data.edge_index[0][i].item()
            target = graph_data.edge_index[1][i].item()
            if source == supplier_idx or target == supplier_idx:
                supplier_edges.append(i)
        
        if not supplier_edges:
            return 0.0
        
        # GNN se carbon flows predict karo
        gnn_model.eval()
        with torch.no_grad():
            outputs = gnn_model(graph_data.x, graph_data.edge_index,
                              edge_attr=getattr(graph_data, 'edge_attr', None))
        
        # Supplier ki total carbon impact
        carbon_flows = outputs['carbon_flows'].cpu().numpy()
        total_carbon = sum(carbon_flows[i] for i in supplier_edges)
        
        return total_carbon / len(supplier_edges)  # Average carbon per edge
    
    def _simulate_replacement_reward(self, gnn_model, graph_data, original_supplier, replacement_supplier):
        """
        Replacement ka reward simulate karo
        """
        # Replacement supplier ka data nikalo
        replacement_features = graph_data.x[replacement_supplier].cpu().numpy()
        
        # Multi-factor reward calculation
        carbon_intensity = replacement_features[0]
        performance_score = replacement_features[1]
        renewable_percentage = replacement_features[5] if len(replacement_features) > 5 else 0.5
        cost_efficiency = replacement_features[7] if len(replacement_features) > 7 else 0.5
        
        # Reward components
        carbon_reward = 1.0 - carbon_intensity  # Lower carbon = higher reward
        performance_reward = performance_score
        renewable_reward = renewable_percentage
        cost_reward = cost_efficiency
        
        # Weighted combination
        total_reward = (
            0.4 * carbon_reward +      # Primary: Carbon efficiency
            0.25 * performance_reward + # Quality
            0.2 * renewable_reward +   # Sustainability
            0.15 * cost_reward         # Cost
        )
        
        return np.clip(total_reward, 0, 1)
    
    def optimize_single_tier_suppliers(self, gnn_model, graph_data, max_replacements=5):
        """
        Main optimization function - single tier suppliers ko optimize karo
        """
        optimization_results = []
        
        # Step 1: High carbon suppliers identify karo
        high_carbon_suppliers = self.identify_high_carbon_suppliers(graph_data, gnn_model)
        
        print(f"Found {len(high_carbon_suppliers)} high carbon suppliers")
        
        # Step 2: Har high carbon supplier ke liye replacement find karo
        for supplier_idx, avg_carbon, edge_count in high_carbon_suppliers[:max_replacements]:
            print(f"\nOptimizing Supplier {supplier_idx} (Carbon: {avg_carbon.item():.3f})")
            
            # Best replacement find karo
            replacement_result = self.find_best_replacement(
                gnn_model, graph_data, supplier_idx
            )
            
            if replacement_result['best_replacement'] is not None:
                # Replacement ko track karo
                self.replacement_history[supplier_idx] = replacement_result
                
                # Reward update karo
                self.update_replacement_rewards(supplier_idx, replacement_result)
                
                optimization_results.append({
                    'original_supplier': supplier_idx,
                    'replacement_supplier': replacement_result['best_replacement'],
                    'carbon_reduction': replacement_result['carbon_reduction'],
                    'combined_score': replacement_result['combined_score'],
                    'original_carbon': replacement_result['original_carbon']
                })
                
                print(f"  Best replacement: Supplier {replacement_result['best_replacement']}")
                # Ensure carbon_reduction is a float for formatting
                carbon_reduction = float(replacement_result['carbon_reduction']) if isinstance(replacement_result['carbon_reduction'], np.ndarray) else replacement_result['carbon_reduction']
                print(f"  Carbon reduction: {carbon_reduction:.3f}")
                combined_score = float(replacement_result['combined_score']) if isinstance(replacement_result['combined_score'], np.ndarray) else replacement_result['combined_score']
                print(f"  Combined score: {combined_score:.3f}")
            else:
                print(f"  No suitable replacement found for Supplier {supplier_idx}")
        
        return optimization_results
    
    def update_replacement_rewards(self, original_supplier, replacement_result):
        """
        Replacement rewards ko update karo
        """
        replacement_supplier = replacement_result['best_replacement']
        reward = replacement_result['combined_score']
        
        # Update replacement reward
        self.replacement_rewards[(original_supplier, replacement_supplier)] = reward
        
        # Update supplier counts and rewards
        self.supplier_counts[replacement_supplier] += 1
        old_reward = self.supplier_rewards[replacement_supplier]
        self.supplier_rewards[replacement_supplier] = old_reward + self.alpha * (reward - old_reward)
    
    def get_optimization_summary(self):
        """
        Optimization ka summary nikalo
        """
        summary = {
            'total_replacements': len(self.replacement_history),
            'average_carbon_reduction': 0.0,
            'total_carbon_saved': 0.0,
            'replacement_details': []
        }
        
        if self.replacement_history:
            carbon_reductions = []
            for original, result in self.replacement_history.items():
                carbon_reduction = result['carbon_reduction']
                carbon_reductions.append(carbon_reduction)
                
                summary['replacement_details'].append({
                    'original_supplier': f'SUP_{original:03d}',
                    'replacement_supplier': f'SUP_{result["best_replacement"]:03d}',
                    'carbon_reduction': carbon_reduction,
                    'combined_score': result['combined_score']
                })
            
            summary['average_carbon_reduction'] = np.mean(carbon_reductions)
            summary['total_carbon_saved'] = np.sum(carbon_reductions)
        
        return summary


class SingleTierGNNBanditIntegration:
    """
    Single-tier optimization ke liye GNN-Bandit integration
    """
    def __init__(self, gnn_model, optimizer):
        self.gnn_model = gnn_model
        self.optimizer = optimizer
    
    def run_single_tier_optimization(self, graph_data, max_replacements=5):
        """
        Complete single-tier optimization pipeline
        """
        print("="*60)
        print("SINGLE-TIER SUPPLIER OPTIMIZATION")
        print("="*60)
        
        # Run optimization
        results = self.optimizer.optimize_single_tier_suppliers(
            self.gnn_model, graph_data, max_replacements
        )
        
        # Get summary
        summary = self.optimizer.get_optimization_summary()
        
        # Print results
        print(f"\nOPTIMIZATION SUMMARY:")
        print(f"Total replacements: {summary['total_replacements']}")
        print(f"Average carbon reduction: {summary['average_carbon_reduction']:.3f}")
        print(f"Total carbon saved: {summary['total_carbon_saved']:.3f}")
        
        print(f"\nDETAILED REPLACEMENTS:")
        for detail in summary['replacement_details']:
            print(f"  {detail['original_supplier']} -> {detail['replacement_supplier']}")
            # Ensure carbon_reduction and combined_score are floats for formatting
            carbon_reduction = float(detail['carbon_reduction']) if isinstance(detail['carbon_reduction'], np.ndarray) else detail['carbon_reduction']
            combined_score = float(detail['combined_score']) if isinstance(detail['combined_score'], np.ndarray) else detail['combined_score']
            print(f"    Carbon reduction: {carbon_reduction:.3f}")
            print(f"    Combined score: {combined_score:.3f}")
        
        return {
            'optimization_results': results,
            'summary': summary
        }
    
    def simulate_replacement_impact(self, graph_data, replacements):
        """
        Replacements ka actual impact simulate karo
        """
        # Create modified graph with replacements
        modified_graph = self._apply_replacements(graph_data, replacements)
        
        # Compare original vs modified performance
        original_performance = self._calculate_graph_performance(graph_data)
        modified_performance = self._calculate_graph_performance(modified_graph)
        
        return {
            'original_performance': original_performance,
            'modified_performance': modified_performance,
            'improvement': modified_performance - original_performance
        }
    
    def _apply_replacements(self, graph_data, replacements):
        """
        Graph me replacements apply karo
        """
        modified_graph = copy.deepcopy(graph_data)
        
        for replacement in replacements:
            original_idx = replacement['original_supplier']
            replacement_idx = replacement['replacement_supplier']
            
            # Node features replace karo
            modified_graph.x[original_idx] = graph_data.x[replacement_idx].clone()
        
        return modified_graph
    
    def _calculate_graph_performance(self, graph_data):
        """
        Graph ka overall performance calculate karo
        """
        self.gnn_model.eval()
        with torch.no_grad():
            outputs = self.gnn_model(graph_data.x, graph_data.edge_index,
                                   edge_attr=getattr(graph_data, 'edge_attr', None))
        
        # Performance metrics
        carbon_flows = outputs['carbon_flows'].cpu().numpy()
        avg_carbon = np.mean(carbon_flows)
        
        # Carbon efficiency score (lower is better)
        performance_score = 1.0 - (avg_carbon / (np.max(carbon_flows) + 1e-8))
        
        return performance_score