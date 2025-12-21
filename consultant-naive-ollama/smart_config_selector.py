"""
Smart Configuration Selector for Alpha Mining

This module implements intelligent configuration selection to:
1. Reduce testing from 1000+ configs to ~50 most promising configs
2. Learn from historical success rates
3. Prioritize configurations based on past performance
4. Implement early stopping for unpromising alphas

Priority: [CAO] - Reduces testing time by 95%

Author: AI Assistant
Date: 2025-12-19
"""

import json
import os
import logging
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartConfigSelector:
    """
    Intelligent configuration selector for alpha mining.
    
    Learns from historical success rates and prioritizes
    configurations that have worked well in the past.
    """
    
    def __init__(self, history_file: str = "config_success_history.json"):
        """
        Initialize smart config selector.
        
        Args:
            history_file: Path to configuration success history file
        """
        self.history_file = history_file
        self.config_stats = defaultdict(lambda: {'success': 0, 'total': 0, 'avg_fitness': 0.0})
        self.load_history()
    
    def load_history(self) -> None:
        """Load configuration success history from file."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                    # Convert to defaultdict
                    for config_key, stats in data.items():
                        self.config_stats[config_key] = stats
                logger.info(f"Loaded success history for {len(self.config_stats)} configurations")
            except Exception as e:
                logger.error(f"Error loading config history: {e}")
        else:
            logger.info("No config history found, starting fresh")
    
    def save_history(self) -> None:
        """Save configuration success history to file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(dict(self.config_stats), f, indent=2)
            logger.info(f"Saved success history for {len(self.config_stats)} configurations")
        except Exception as e:
            logger.error(f"Error saving config history: {e}")
    
    def config_to_key(self, config: Dict) -> str:
        """
        Convert configuration dict to a unique key string.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Unique key string
        """
        settings = config.get('settings', {})
        key_parts = [
            settings.get('region', 'USA'),
            settings.get('universe', 'TOP3000'),
            settings.get('neutralization', 'INDUSTRY'),
            str(settings.get('delay', 1)),
            settings.get('pasteurization', 'ON'),
            settings.get('nanHandling', 'OFF')
        ]
        return '|'.join(key_parts)
    
    def update_config_stats(self, config: Dict, fitness: float, success: bool) -> None:
        """
        Update statistics for a configuration.
        
        Args:
            config: Configuration dictionary
            fitness: Fitness score achieved
            success: Whether the alpha was successful (fitness > threshold)
        """
        key = self.config_to_key(config)
        stats = self.config_stats[key]
        
        # Update counts
        stats['total'] += 1
        if success:
            stats['success'] += 1
        
        # Update average fitness (running average)
        n = stats['total']
        old_avg = stats['avg_fitness']
        stats['avg_fitness'] = old_avg + (fitness - old_avg) / n
        
        # Save periodically
        if stats['total'] % 10 == 0:
            self.save_history()
    
    def get_config_score(self, config: Dict) -> float:
        """
        Calculate priority score for a configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Priority score (higher is better)
        """
        key = self.config_to_key(config)
        stats = self.config_stats[key]
        
        total = stats['total']
        success = stats['success']
        avg_fitness = stats['avg_fitness']
        
        # If never tested, give medium priority
        if total == 0:
            return 0.5
        
        # Calculate success rate
        success_rate = success / total
        
        # Combine success rate and average fitness
        # Weight: 60% success rate, 40% average fitness
        score = 0.6 * success_rate + 0.4 * avg_fitness
        
        # Add exploration bonus for under-tested configs (UCB-like)
        exploration_bonus = np.sqrt(2 * np.log(sum(s['total'] for s in self.config_stats.values()) + 1) / (total + 1))
        score += 0.1 * exploration_bonus
        
        return score
    
    def select_top_configs(self, all_configs: List[Dict], top_k: int = 50) -> List[Dict]:
        """
        Select top-k configurations based on historical performance.
        
        Args:
            all_configs: List of all possible configurations
            top_k: Number of configurations to select
            
        Returns:
            List of top-k configurations
        """
        # Score all configurations
        scored_configs = []
        for config in all_configs:
            score = self.get_config_score(config)
            scored_configs.append((score, config))
        
        # Sort by score (descending)
        scored_configs.sort(reverse=True, key=lambda x: x[0])
        
        # Return top-k
        top_configs = [config for score, config in scored_configs[:top_k]]
        
        logger.info(f"Selected top {len(top_configs)} configurations out of {len(all_configs)}")
        logger.info(f"Top config score: {scored_configs[0][0]:.3f}")
        logger.info(f"Bottom config score: {scored_configs[-1][0]:.3f}")
        
        return top_configs
    
    def should_stop_early(self, results: List[Dict], min_configs: int = 10,
                         fitness_threshold: float = 0.3) -> bool:
        """
        Determine if testing should stop early for an alpha.
        
        Args:
            results: List of test results so far
            min_configs: Minimum configs to test before stopping
            fitness_threshold: Minimum fitness to continue testing
            
        Returns:
            True if should stop early, False otherwise
        """
        # Need at least min_configs results
        if len(results) < min_configs:
            return False
        
        # Check if any result exceeds threshold
        max_fitness = max([r.get('fitness', 0) for r in results])
        
        if max_fitness < fitness_threshold:
            logger.info(f"Early stopping: max fitness {max_fitness:.3f} < threshold {fitness_threshold}")
            return True
        
        return False
    
    def get_statistics(self) -> Dict:
        """Get statistics about configuration performance."""
        if not self.config_stats:
            return {}
        
        all_stats = list(self.config_stats.values())
        
        return {
            'total_configs_tested': len(self.config_stats),
            'total_tests': sum(s['total'] for s in all_stats),
            'total_successes': sum(s['success'] for s in all_stats),
            'overall_success_rate': sum(s['success'] for s in all_stats) / max(1, sum(s['total'] for s in all_stats)),
            'avg_fitness': np.mean([s['avg_fitness'] for s in all_stats if s['total'] > 0]),
            'best_config': max(self.config_stats.items(), key=lambda x: x[1]['avg_fitness'])[0] if self.config_stats else None
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize selector
    selector = SmartConfigSelector()
    
    # Example: Generate some test configurations
    test_configs = [
        {'settings': {'region': 'USA', 'universe': 'TOP3000', 'neutralization': 'INDUSTRY', 'delay': 1, 'pasteurization': 'ON', 'nanHandling': 'OFF'}},
        {'settings': {'region': 'USA', 'universe': 'TOP1000', 'neutralization': 'SECTOR', 'delay': 1, 'pasteurization': 'ON', 'nanHandling': 'OFF'}},
        {'settings': {'region': 'EUR', 'universe': 'TOP2500', 'neutralization': 'MARKET', 'delay': 0, 'pasteurization': 'OFF', 'nanHandling': 'ON'}},
    ]
    
    # Select top configs
    top_configs = selector.select_top_configs(test_configs, top_k=2)
    print(f"\nSelected {len(top_configs)} top configurations")
    
    # Get statistics
    stats = selector.get_statistics()
    print(f"\n=== Configuration Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

