"""
Feedback Loop System for Continuous Improvement

This module implements a feedback loop to learn from:
1. Failed alphas (why they failed)
2. Successful alphas (what made them work)
3. Model performance (which models generate best alphas)

Priority: [TRUNG BÌNH] - Long-term improvement

Author: AI Assistant
Date: 2025-12-19
"""

import json
import os
import logging
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackLoopSystem:
    """
    Feedback loop system for continuous learning and improvement.
    
    Tracks failures, successes, and model performance to
    improve future alpha generation.
    """
    
    def __init__(self, feedback_file: str = "feedback_history.json"):
        """
        Initialize feedback loop system.
        
        Args:
            feedback_file: Path to feedback history file
        """
        self.feedback_file = feedback_file
        self.feedback_data = {
            'failures': [],
            'successes': [],
            'model_performance': defaultdict(lambda: {'total': 0, 'successful': 0, 'avg_fitness': 0.0}),
            'operator_performance': defaultdict(lambda: {'total': 0, 'successful': 0, 'avg_fitness': 0.0}),
            'failure_patterns': defaultdict(int),
            'success_patterns': defaultdict(int)
        }
        self.load_feedback()
    
    def load_feedback(self) -> None:
        """Load feedback history from file."""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r') as f:
                    data = json.load(f)
                    self.feedback_data.update(data)
                    # Convert defaultdicts
                    self.feedback_data['model_performance'] = defaultdict(
                        lambda: {'total': 0, 'successful': 0, 'avg_fitness': 0.0},
                        self.feedback_data.get('model_performance', {})
                    )
                    self.feedback_data['operator_performance'] = defaultdict(
                        lambda: {'total': 0, 'successful': 0, 'avg_fitness': 0.0},
                        self.feedback_data.get('operator_performance', {})
                    )
                    self.feedback_data['failure_patterns'] = defaultdict(
                        int,
                        self.feedback_data.get('failure_patterns', {})
                    )
                    self.feedback_data['success_patterns'] = defaultdict(
                        int,
                        self.feedback_data.get('success_patterns', {})
                    )
                logger.info(f"Loaded feedback history with {len(self.feedback_data['failures'])} failures and {len(self.feedback_data['successes'])} successes")
            except Exception as e:
                logger.error(f"Error loading feedback history: {e}")
    
    def save_feedback(self) -> None:
        """Save feedback history to file."""
        try:
            # Convert defaultdicts to regular dicts for JSON serialization
            save_data = {
                'failures': self.feedback_data['failures'],
                'successes': self.feedback_data['successes'],
                'model_performance': dict(self.feedback_data['model_performance']),
                'operator_performance': dict(self.feedback_data['operator_performance']),
                'failure_patterns': dict(self.feedback_data['failure_patterns']),
                'success_patterns': dict(self.feedback_data['success_patterns'])
            }
            
            with open(self.feedback_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            logger.info(f"Saved feedback history")
        except Exception as e:
            logger.error(f"Error saving feedback history: {e}")
    
    def extract_operators(self, expression: str) -> List[str]:
        """
        Extract operators from alpha expression.
        
        Args:
            expression: Alpha expression string
            
        Returns:
            List of operator names
        """
        # Extract function names (words followed by parentheses)
        operators = re.findall(r'(\w+)\(', expression)
        return operators
    
    def analyze_failure(self, alpha: Dict, reason: str = "low_fitness") -> None:
        """
        Analyze a failed alpha and extract patterns.
        
        Args:
            alpha: Alpha dictionary with expression and metrics
            reason: Reason for failure (low_fitness, error, timeout, etc.)
        """
        expression = alpha.get('expression', '')
        fitness = alpha.get('fitness', 0)
        model = alpha.get('model', 'unknown')
        
        # Record failure
        failure_record = {
            'timestamp': datetime.now().isoformat(),
            'expression': expression,
            'fitness': fitness,
            'reason': reason,
            'model': model
        }
        self.feedback_data['failures'].append(failure_record)
        
        # Update failure patterns
        self.feedback_data['failure_patterns'][reason] += 1
        
        # Extract operators and update performance
        operators = self.extract_operators(expression)
        for op in operators:
            stats = self.feedback_data['operator_performance'][op]
            stats['total'] += 1
            # Update running average
            n = stats['total']
            old_avg = stats['avg_fitness']
            stats['avg_fitness'] = old_avg + (fitness - old_avg) / n
        
        # Update model performance
        model_stats = self.feedback_data['model_performance'][model]
        model_stats['total'] += 1
        n = model_stats['total']
        old_avg = model_stats['avg_fitness']
        model_stats['avg_fitness'] = old_avg + (fitness - old_avg) / n
        
        # Save periodically
        if len(self.feedback_data['failures']) % 10 == 0:
            self.save_feedback()
    
    def analyze_success(self, alpha: Dict) -> None:
        """
        Analyze a successful alpha and extract patterns.
        
        Args:
            alpha: Alpha dictionary with expression and metrics
        """
        expression = alpha.get('expression', '')
        fitness = alpha.get('fitness', 0)
        model = alpha.get('model', 'unknown')
        sharpe = alpha.get('sharpe', 0)
        turnover = alpha.get('turnover', 0)
        
        # Record success
        success_record = {
            'timestamp': datetime.now().isoformat(),
            'expression': expression,
            'fitness': fitness,
            'sharpe': sharpe,
            'turnover': turnover,
            'model': model
        }
        self.feedback_data['successes'].append(success_record)
        
        # Extract operators and update performance
        operators = self.extract_operators(expression)
        for op in operators:
            stats = self.feedback_data['operator_performance'][op]
            stats['total'] += 1
            stats['successful'] += 1
            # Update running average
            n = stats['total']
            old_avg = stats['avg_fitness']
            stats['avg_fitness'] = old_avg + (fitness - old_avg) / n
            
            # Track success pattern
            self.feedback_data['success_patterns'][op] += 1
        
        # Update model performance
        model_stats = self.feedback_data['model_performance'][model]
        model_stats['total'] += 1
        model_stats['successful'] += 1
        n = model_stats['total']
        old_avg = model_stats['avg_fitness']
        model_stats['avg_fitness'] = old_avg + (fitness - old_avg) / n
        
        # Save periodically
        if len(self.feedback_data['successes']) % 5 == 0:
            self.save_feedback()
    
    def get_operator_blacklist(self, min_total: int = 10, max_success_rate: float = 0.1) -> List[str]:
        """
        Get list of operators that consistently fail.
        
        Args:
            min_total: Minimum number of times operator must be used
            max_success_rate: Maximum success rate to be blacklisted
            
        Returns:
            List of operator names to avoid
        """
        blacklist = []
        
        for op, stats in self.feedback_data['operator_performance'].items():
            if stats['total'] >= min_total:
                success_rate = stats['successful'] / stats['total']
                if success_rate <= max_success_rate:
                    blacklist.append(op)
        
        return blacklist
    
    def get_operator_whitelist(self, min_total: int = 5, min_success_rate: float = 0.3) -> List[str]:
        """
        Get list of operators that consistently succeed.
        
        Args:
            min_total: Minimum number of times operator must be used
            min_success_rate: Minimum success rate to be whitelisted
            
        Returns:
            List of operator names to prioritize
        """
        whitelist = []
        
        for op, stats in self.feedback_data['operator_performance'].items():
            if stats['total'] >= min_total:
                success_rate = stats['successful'] / stats['total']
                if success_rate >= min_success_rate:
                    whitelist.append(op)
        
        return whitelist
    
    def get_best_model(self) -> Optional[str]:
        """
        Get the model with best performance.
        
        Returns:
            Model name with highest success rate
        """
        best_model = None
        best_score = 0
        
        for model, stats in self.feedback_data['model_performance'].items():
            if stats['total'] >= 5:  # Minimum sample size
                success_rate = stats['successful'] / stats['total']
                # Combine success rate and average fitness
                score = 0.6 * success_rate + 0.4 * stats['avg_fitness']
                if score > best_score:
                    best_score = score
                    best_model = model
        
        return best_model
    
    def get_statistics(self) -> Dict:
        """Get feedback loop statistics."""
        total_alphas = len(self.feedback_data['failures']) + len(self.feedback_data['successes'])
        
        return {
            'total_alphas': total_alphas,
            'total_failures': len(self.feedback_data['failures']),
            'total_successes': len(self.feedback_data['successes']),
            'success_rate': len(self.feedback_data['successes']) / max(1, total_alphas),
            'operators_tracked': len(self.feedback_data['operator_performance']),
            'models_tracked': len(self.feedback_data['model_performance']),
            'best_model': self.get_best_model(),
            'operator_blacklist': self.get_operator_blacklist(),
            'operator_whitelist': self.get_operator_whitelist()
        }


# Example usage
if __name__ == "__main__":
    # Initialize feedback loop
    feedback = FeedbackLoopSystem()
    
    # Get statistics
    stats = feedback.get_statistics()
    print("\n=== Feedback Loop Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

