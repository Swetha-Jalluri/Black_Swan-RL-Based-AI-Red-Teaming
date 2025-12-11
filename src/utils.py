"""
Utility Functions
Helper functions for logging, visualization, and analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ExperimentConfig, EnvConfig, TaxonomyConfig


def save_episode_results(
    episode_data: List[Dict],
    filename: Optional[str] = None,
    difficulty: str = "mixed"
) -> str:
    """
    Save episode results to CSV file
    
    Args:
        episode_data: List of episode dictionaries
        filename: Custom filename (optional)
        difficulty: Difficulty level for default naming
    
    Returns:
        filepath: Path where file was saved
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"episodes_{difficulty}_{timestamp}.csv"
    
    filepath = os.path.join(ExperimentConfig.RESULTS_DIR, filename)
    
    # Convert to DataFrame
    df = pd.DataFrame(episode_data)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    
    print(f"Episode results saved to: {filepath}")
    return filepath


def load_episode_results(filepath: str) -> pd.DataFrame:
    """
    Load episode results from CSV
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        df: DataFrame with episode data
    """
    df = pd.read_csv(filepath)
    return df


def save_training_metadata(
    metadata: Dict,
    filename: Optional[str] = None
) -> str:
    """
    Save training configuration and metadata to JSON
    
    Args:
        metadata: Dictionary with training parameters
        filename: Custom filename (optional)
    
    Returns:
        filepath: Path where file was saved
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metadata_{timestamp}.json"
    
    filepath = os.path.join(ExperimentConfig.RESULTS_DIR, filename)
    
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Training metadata saved to: {filepath}")
    return filepath


def calculate_success_rate(episode_data: List[Dict], window: int = 10) -> List[float]:
    """
    Calculate rolling success rate over episodes
    
    Args:
        episode_data: List of episode dictionaries
        window: Window size for rolling average
    
    Returns:
        success_rates: List of success rates over time
    """
    successes = [1 if ep.get('success', False) else 0 for ep in episode_data]
    
    success_rates = []
    for i in range(len(successes)):
        start_idx = max(0, i - window + 1)
        window_successes = successes[start_idx:i+1]
        success_rate = sum(window_successes) / len(window_successes) * 100
        success_rates.append(success_rate)
    
    return success_rates


def calculate_average_reward(episode_data: List[Dict], window: int = 10) -> List[float]:
    """
    Calculate rolling average reward over episodes
    
    Args:
        episode_data: List of episode dictionaries
        window: Window size for rolling average
    
    Returns:
        avg_rewards: List of average rewards over time
    """
    rewards = [ep.get('total_reward', 0) for ep in episode_data]
    
    avg_rewards = []
    for i in range(len(rewards)):
        start_idx = max(0, i - window + 1)
        window_rewards = rewards[start_idx:i+1]
        avg_reward = sum(window_rewards) / len(window_rewards)
        avg_rewards.append(avg_reward)
    
    return avg_rewards


def plot_learning_curves(
    episode_data: List[Dict],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot comprehensive learning curves
    
    Creates a multi-panel figure showing:
    - Success rate over time
    - Average reward over time
    - Episode length over time
    - Vulnerability categories found
    
    Args:
        episode_data: List of episode dictionaries
        save_path: Path to save figure (optional)
        show: If True, display the plot
    """
    # Set style
    plt.style.use(ExperimentConfig.PLOT_STYLE)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Red Team Agent Learning Progress', fontsize=16, fontweight='bold')
    
    episodes = range(1, len(episode_data) + 1)
    
    # Plot 1: Success Rate
    ax1 = axes[0, 0]
    success_rates = calculate_success_rate(episode_data, window=10)
    ax1.plot(episodes, success_rates, linewidth=2, color='green', alpha=0.7)
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Success Rate (10-episode rolling average)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average Reward
    ax2 = axes[0, 1]
    avg_rewards = calculate_average_reward(episode_data, window=10)
    ax2.plot(episodes, avg_rewards, linewidth=2, color='blue', alpha=0.7)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Zero reward')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Reward')
    ax2.set_title('Average Reward (10-episode rolling average)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Episode Length
    ax3 = axes[1, 0]
    turns = [ep.get('turns', 0) for ep in episode_data]
    ax3.plot(episodes, turns, linewidth=2, color='orange', alpha=0.7)
    ax3.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Max turns')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Turns to Success/Timeout')
    ax3.set_title('Episode Length')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Vulnerability Categories
    ax4 = axes[1, 1]
    categories = [ep.get('vulnerability_category', 'none') for ep in episode_data if ep.get('success')]
    if categories:
        category_counts = pd.Series(categories).value_counts()
        category_counts.plot(kind='bar', ax=ax4, color='purple', alpha=0.7)
        ax4.set_xlabel('Vulnerability Type')
        ax4.set_ylabel('Count')
        ax4.set_title('Vulnerability Categories Discovered')
        ax4.tick_params(axis='x', rotation=45)
    else:
        ax4.text(0.5, 0.5, 'No vulnerabilities found yet', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Vulnerability Categories Discovered')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=ExperimentConfig.PLOT_DPI, bbox_inches='tight')
        print(f"Learning curves saved to: {save_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison(
    results_dict: Dict[str, List[Dict]],
    metric: str = 'success_rate',
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot comparison of multiple training runs
    
    Args:
        results_dict: Dictionary mapping labels to episode data
                     e.g., {"Easy": episodes_easy, "Hard": episodes_hard}
        metric: What to compare ('success_rate', 'avg_reward', 'turns')
        save_path: Path to save figure (optional)
        show: If True, display the plot
    """
    plt.style.use(ExperimentConfig.PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for label, episode_data in results_dict.items():
        episodes = range(1, len(episode_data) + 1)
        
        if metric == 'success_rate':
            values = calculate_success_rate(episode_data, window=10)
            ylabel = 'Success Rate (%)'
            title = 'Success Rate Comparison'
        elif metric == 'avg_reward':
            values = calculate_average_reward(episode_data, window=10)
            ylabel = 'Average Reward'
            title = 'Average Reward Comparison'
        elif metric == 'turns':
            values = [ep.get('turns', 0) for ep in episode_data]
            ylabel = 'Turns'
            title = 'Episode Length Comparison'
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        ax.plot(episodes, values, linewidth=2, label=label, alpha=0.7)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=ExperimentConfig.PLOT_DPI, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def print_training_summary(episode_data: List[Dict], difficulty: str = "unknown"):
    """
    Print formatted training summary statistics
    
    Args:
        episode_data: List of episode dictionaries
        difficulty: Difficulty level
    """
    total_episodes = len(episode_data)
    successes = sum(1 for ep in episode_data if ep.get('success', False))
    success_rate = successes / total_episodes * 100 if total_episodes > 0 else 0
    
    rewards = [ep.get('total_reward', 0) for ep in episode_data]
    avg_reward = np.mean(rewards) if rewards else 0
    max_reward = np.max(rewards) if rewards else 0
    
    turns = [ep.get('turns', 0) for ep in episode_data if ep.get('success')]
    avg_turns_success = np.mean(turns) if turns else 0
    
    categories = [ep.get('vulnerability_category') for ep in episode_data if ep.get('success')]
    category_counts = pd.Series(categories).value_counts() if categories else pd.Series()
    
    print("\n" + "="*70)
    print(f"TRAINING SUMMARY: {difficulty.upper()} DIFFICULTY")
    print("="*70)
    print(f"Total Episodes:        {total_episodes}")
    print(f"Successful Attacks:    {successes} ({success_rate:.1f}%)")
    print(f"Average Reward:        {avg_reward:.2f}")
    print(f"Maximum Reward:        {max_reward:.2f}")
    if turns:
        print(f"Avg Turns to Success:  {avg_turns_success:.1f}")
    
    if not category_counts.empty:
        print(f"\nVulnerabilities Found:")
        for category, count in category_counts.items():
            print(f"  - {category}: {count}")
    
    print("="*70 + "\n")


def create_vulnerability_heatmap(
    episode_data: List[Dict],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Create heatmap showing which attack strategies found which vulnerabilities
    
    Args:
        episode_data: List of episode dictionaries
        save_path: Path to save figure (optional)
        show: If True, display the plot
    """
    plt.style.use(ExperimentConfig.PLOT_STYLE)
    
    # Extract attack strategy and vulnerability pairs from successful episodes
    strategy_vuln_pairs = []
    for ep in episode_data:
        if ep.get('success') and 'conversation_history' in ep:
            # Get the attack strategy that succeeded
            for msg in ep['conversation_history']:
                if msg.get('role') == 'attacker' and 'strategy' in msg:
                    strategy = msg['strategy']
                    vuln_category = ep.get('vulnerability_category', 'unknown')
                    strategy_vuln_pairs.append((strategy, vuln_category))
    
    if not strategy_vuln_pairs:
        print("No successful attacks to visualize")
        return
    
    # Create matrix
    strategies = list(EnvConfig.ATTACK_STRATEGIES.values())
    categories = list(TaxonomyConfig.VULNERABILITY_CATEGORIES.keys())
    
    matrix = np.zeros((len(strategies), len(categories)))
    
    for strategy_id, vuln_cat in strategy_vuln_pairs:
        if strategy_id < len(strategies) and vuln_cat in categories:
            strategy_idx = strategy_id
            cat_idx = categories.index(vuln_cat)
            matrix[strategy_idx, cat_idx] += 1
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        matrix,
        xticklabels=categories,
        yticklabels=strategies,
        annot=True,
        fmt='.0f',
        cmap='YlOrRd',
        ax=ax,
        cbar_kws={'label': 'Number of Successful Attacks'}
    )
    
    ax.set_xlabel('Vulnerability Category')
    ax.set_ylabel('Attack Strategy')
    ax.set_title('Attack Strategy Effectiveness by Vulnerability Type')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=ExperimentConfig.PLOT_DPI, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def estimate_api_cost(num_episodes: int, avg_turns: int = 7) -> Dict:
    """
    Estimate API usage and costs for training
    
    Args:
        num_episodes: Number of episodes to run
        avg_turns: Average turns per episode
    
    Returns:
        cost_estimate: Dictionary with usage statistics
    """
    # API calls per turn: attack generation + target response + judge evaluation
    calls_per_turn = 3
    
    total_calls = num_episodes * avg_turns * calls_per_turn
    
    # Free tier limits
    calls_per_minute = 4  # Conservative estimate from config
    calls_per_day = 1500
    
    # Time estimates
    minutes_needed = total_calls / calls_per_minute
    hours_needed = minutes_needed / 60
    
    # Check if within daily limit
    within_daily_limit = total_calls <= calls_per_day
    
    return {
        "total_api_calls": total_calls,
        "estimated_hours": round(hours_needed, 2),
        "estimated_minutes": round(minutes_needed, 1),
        "calls_per_minute": calls_per_minute,
        "within_daily_limit": within_daily_limit,
        "daily_limit": calls_per_day,
        "cost_usd": 0.0  # Free tier
    }


def print_api_estimate(num_episodes: int, avg_turns: int = 7):
    """
    Print formatted API usage estimate
    
    Args:
        num_episodes: Number of episodes to run
        avg_turns: Average turns per episode
    """
    estimate = estimate_api_cost(num_episodes, avg_turns)
    
    print("\n" + "="*70)
    print("API USAGE ESTIMATE")
    print("="*70)
    print(f"Episodes:              {num_episodes}")
    print(f"Avg turns per episode: {avg_turns}")
    print(f"Total API calls:       {estimate['total_api_calls']}")
    print(f"Estimated time:        {estimate['estimated_hours']:.1f} hours ({estimate['estimated_minutes']:.0f} minutes)")
    print(f"Rate limit:            {estimate['calls_per_minute']} calls/minute")
    print(f"Daily limit:           {estimate['daily_limit']} calls/day")
    
    if estimate['within_daily_limit']:
        print(f"Status:                ✓ Within free tier limits")
    else:
        print(f"Status:                ✗ Exceeds daily limit - split into multiple days")
    
    print(f"Cost:                  $0.00 (FREE TIER)")
    print("="*70 + "\n")