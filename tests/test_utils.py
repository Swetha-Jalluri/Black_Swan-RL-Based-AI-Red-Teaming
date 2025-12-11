"""
Test utility functions
"""
from src.utils import estimate_api_cost, print_api_estimate

# Test API cost estimation
print("Testing API cost estimation:\n")

# Small test run
print_api_estimate(num_episodes=10, avg_turns=5)

# Full training run
print_api_estimate(num_episodes=100, avg_turns=7)

# Large experiment
print_api_estimate(num_episodes=300, avg_turns=7)