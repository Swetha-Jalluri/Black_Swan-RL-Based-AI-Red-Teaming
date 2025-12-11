"""
Statistical Analysis Module
Provides rigorous statistical evaluation of RL agent performance
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ExperimentConfig


def calculate_statistics(episode_data: List[Dict], metric: str = 'total_reward') -> Dict:
    """
    Calculate comprehensive statistics for a metric
    
    Args:
        episode_data: List of episode dictionaries
        metric: Which metric to analyze ('total_reward', 'turns', etc.)
    
    Returns:
        stats: Dictionary with mean, std, CI, etc.
    """
    values = [ep.get(metric, 0) for ep in episode_data]
    
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample standard deviation
    sem = stats.sem(values)  # Standard error of mean
    
    # 95% confidence interval
    confidence_level = 0.95
    ci = stats.t.interval(
        confidence_level,
        df=n-1,
        loc=mean,
        scale=sem
    )
    
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
        "median": np.median(values),
        "min": np.min(values),
        "max": np.max(values)
    }


def compare_agents_ttest(
    agent1_data: List[Dict],
    agent2_data: List[Dict],
    metric: str = 'total_reward',
    agent1_name: str = "Agent 1",
    agent2_name: str = "Agent 2"
) -> Dict:
    """
    Perform independent t-test comparing two agents
    
    Null hypothesis: Both agents have same mean performance
    Alternative: Agents have different mean performance
    
    Args:
        agent1_data: Episode data for first agent
        agent2_data: Episode data for second agent
        metric: Metric to compare
        agent1_name: Name of first agent
        agent2_name: Name of second agent
    
    Returns:
        result: Dictionary with t-test results
    """
    values1 = [ep.get(metric, 0) for ep in agent1_data]
    values2 = [ep.get(metric, 0) for ep in agent2_data]
    
    # Perform independent samples t-test
    t_statistic, p_value = stats.ttest_ind(values1, values2)
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(values1)**2 + np.std(values2)**2) / 2)
    cohens_d = (np.mean(values1) - np.mean(values2)) / pooled_std if pooled_std > 0 else 0
    
    # Determine significance
    alpha = 0.05
    is_significant = p_value < alpha
    
    return {
        "agent1_name": agent1_name,
        "agent2_name": agent2_name,
        "metric": metric,
        "agent1_mean": np.mean(values1),
        "agent2_mean": np.mean(values2),
        "agent1_std": np.std(values1),
        "agent2_std": np.std(values2),
        "t_statistic": t_statistic,
        "p_value": p_value,
        "is_significant": is_significant,
        "cohens_d": cohens_d,
        "interpretation": _interpret_ttest(p_value, cohens_d, agent1_name, agent2_name)
    }


def _interpret_ttest(p_value: float, cohens_d: float, agent1: str, agent2: str) -> str:
    """
    Interpret t-test results in plain language
    """
    alpha = 0.05
    
    if p_value >= alpha:
        return f"No significant difference between {agent1} and {agent2} (p={p_value:.4f})"
    
    # Significant difference exists
    better_agent = agent1 if cohens_d > 0 else agent2
    
    # Effect size interpretation
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        effect = "negligible"
    elif abs_d < 0.5:
        effect = "small"
    elif abs_d < 0.8:
        effect = "medium"
    else:
        effect = "large"
    
    return f"{better_agent} significantly outperforms (p={p_value:.4f}, d={cohens_d:.3f}, {effect} effect)"


def success_rate_comparison(
    agent1_data: List[Dict],
    agent2_data: List[Dict],
    agent1_name: str = "Agent 1",
    agent2_name: str = "Agent 2"
) -> Dict:
    """
    Compare success rates using proportion test
    
    Args:
        agent1_data: Episode data for first agent
        agent2_data: Episode data for second agent
        agent1_name: Name of first agent
        agent2_name: Name of second agent
    
    Returns:
        result: Dictionary with comparison results
    """
    successes1 = sum(1 for ep in agent1_data if ep.get('success', False))
    successes2 = sum(1 for ep in agent2_data if ep.get('success', False))
    
    n1 = len(agent1_data)
    n2 = len(agent2_data)
    
    rate1 = successes1 / n1 if n1 > 0 else 0
    rate2 = successes2 / n2 if n2 > 0 else 0
    
    # Two-proportion z-test
    pooled_rate = (successes1 + successes2) / (n1 + n2)
    se = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/n1 + 1/n2))
    
    if se > 0:
        z_statistic = (rate1 - rate2) / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))  # Two-tailed
    else:
        z_statistic = 0
        p_value = 1.0
    
    # Calculate confidence intervals for each rate
    ci1 = stats.binom.interval(0.95, n1, rate1)
    ci2 = stats.binom.interval(0.95, n2, rate2)
    
    return {
        "agent1_name": agent1_name,
        "agent2_name": agent2_name,
        "agent1_success_rate": rate1 * 100,
        "agent2_success_rate": rate2 * 100,
        "agent1_successes": successes1,
        "agent2_successes": successes2,
        "agent1_n": n1,
        "agent2_n": n2,
        "z_statistic": z_statistic,
        "p_value": p_value,
        "is_significant": p_value < 0.05,
        "difference": (rate1 - rate2) * 100
    }


def generate_statistical_report(
    qlearning_data: Dict[str, List[Dict]],
    ucb_data: Dict[str, List[Dict]],
    random_data: Dict[str, List[Dict]],
    save_path: str = None
) -> str:
    """
    Generate comprehensive statistical report
    
    Args:
        qlearning_data: Dict mapping difficulty to Q-Learning episodes
        ucb_data: Dict mapping difficulty to UCB episodes
        random_data: Dict mapping difficulty to Random episodes
        save_path: Path to save report (optional)
    
    Returns:
        report: Formatted statistical report text
    """
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("STATISTICAL ANALYSIS REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    
    difficulties = ["easy", "medium", "hard"]
    
    for difficulty in difficulties:
        report_lines.append(f"\n{'#'*80}")
        report_lines.append(f"# {difficulty.upper()} DIFFICULTY")
        report_lines.append(f"{'#'*80}\n")
        
        q_data = qlearning_data[difficulty]
        u_data = ucb_data[difficulty]
        r_data = random_data[difficulty]
        
        # Performance statistics
        report_lines.append("REWARD STATISTICS:")
        report_lines.append("-"*80)
        
        for agent_name, agent_data in [("Q-Learning", q_data), ("UCB", u_data), ("Random", r_data)]:
            stats_reward = calculate_statistics(agent_data, 'total_reward')
            report_lines.append(f"{agent_name:12s}: Mean={stats_reward['mean']:7.2f} ± {stats_reward['std']:6.2f} "
                              f"| 95% CI=[{stats_reward['ci_lower']:.2f}, {stats_reward['ci_upper']:.2f}]")
        
        report_lines.append("")
        
        # Success rate comparison
        report_lines.append("SUCCESS RATE STATISTICS:")
        report_lines.append("-"*80)
        
        for agent_name, agent_data in [("Q-Learning", q_data), ("UCB", u_data), ("Random", r_data)]:
            successes = sum(1 for ep in agent_data if ep.get('success', False))
            n = len(agent_data)
            rate = successes / n * 100
            
            # Binomial confidence interval
            ci_low, ci_high = stats.binom.interval(0.95, n, rate/100)
            ci_low_pct = ci_low / n * 100
            ci_high_pct = ci_high / n * 100
            
            report_lines.append(f"{agent_name:12s}: {rate:5.1f}% ({successes}/{n}) "
                              f"| 95% CI=[{ci_low_pct:.1f}%, {ci_high_pct:.1f}%]")
        
        report_lines.append("")
        
        # Pairwise comparisons
        report_lines.append("PAIRWISE T-TESTS (Reward):")
        report_lines.append("-"*80)
        
        # Q-Learning vs UCB
        comparison1 = compare_agents_ttest(q_data, u_data, 'total_reward', "Q-Learning", "UCB")
        report_lines.append(f"Q-Learning vs UCB:")
        report_lines.append(f"  {comparison1['interpretation']}")
        report_lines.append(f"  Mean difference: {comparison1['agent1_mean'] - comparison1['agent2_mean']:.2f}")
        report_lines.append("")
        
        # Q-Learning vs Random
        comparison2 = compare_agents_ttest(q_data, r_data, 'total_reward', "Q-Learning", "Random")
        report_lines.append(f"Q-Learning vs Random:")
        report_lines.append(f"  {comparison2['interpretation']}")
        report_lines.append(f"  Mean difference: {comparison2['agent1_mean'] - comparison2['agent2_mean']:.2f}")
        report_lines.append("")
        
        # UCB vs Random
        comparison3 = compare_agents_ttest(u_data, r_data, 'total_reward', "UCB", "Random")
        report_lines.append(f"UCB vs Random:")
        report_lines.append(f"  {comparison3['interpretation']}")
        report_lines.append(f"  Mean difference: {comparison3['agent1_mean'] - comparison3['agent2_mean']:.2f}")
        report_lines.append("")
        
        # Success rate proportion tests
        report_lines.append("PROPORTION TESTS (Success Rate):")
        report_lines.append("-"*80)
        
        succ_comp1 = success_rate_comparison(q_data, u_data, "Q-Learning", "UCB")
        report_lines.append(f"Q-Learning vs UCB:")
        report_lines.append(f"  Difference: {succ_comp1['difference']:.1f} percentage points")
        report_lines.append(f"  p-value: {succ_comp1['p_value']:.4f} "
                          f"({'significant' if succ_comp1['is_significant'] else 'not significant'})")
        report_lines.append("")
    
    report_lines.append("="*80)
    
    # Combine into single report
    report_text = "\n".join(report_lines)
    
    # Save if path provided
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"Statistical report saved to: {save_path}")
    
    return report_text


def plot_performance_with_ci(
    results_dict: Dict[str, List[Dict]],
    metric: str = 'total_reward',
    save_path: str = None,
    show: bool = True
):
    """
    Plot performance comparison with confidence intervals
    
    Args:
        results_dict: Dict mapping agent names to episode data
        metric: Metric to plot
        save_path: Path to save figure
        show: Whether to display plot
    """
    plt.style.use(ExperimentConfig.PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    agents = list(results_dict.keys())
    means = []
    stds = []
    cis_lower = []
    cis_upper = []
    
    for agent_name in agents:
        stats_dict = calculate_statistics(results_dict[agent_name], metric)
        means.append(stats_dict['mean'])
        stds.append(stats_dict['std'])
        cis_lower.append(stats_dict['ci_lower'])
        cis_upper.append(stats_dict['ci_upper'])
    
    x_pos = np.arange(len(agents))
    
    # Plot bars with error bars
    ax.bar(x_pos, means, yerr=stds, capsize=10, alpha=0.7, 
           error_kw={'linewidth': 2, 'elinewidth': 2})
    
    # Add confidence interval lines
    for i, (lower, upper) in enumerate(zip(cis_lower, cis_upper)):
        ax.plot([i-0.3, i+0.3], [lower, lower], 'r-', linewidth=2, label='95% CI' if i == 0 else "")
        ax.plot([i-0.3, i+0.3], [upper, upper], 'r-', linewidth=2)
    
    ax.set_xlabel('Agent', fontsize=12)
    ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12)
    ax.set_title(f'Performance Comparison with Confidence Intervals\n(Mean ± Std, with 95% CI)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(agents, rotation=0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=ExperimentConfig.PLOT_DPI, bbox_inches='tight')
        print(f"Statistical plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_comparison_table(
    qlearning_data: Dict[str, List[Dict]],
    ucb_data: Dict[str, List[Dict]],
    random_data: Dict[str, List[Dict]],
    save_path: str = None
) -> pd.DataFrame:
    """
    Create comprehensive comparison table with statistics
    
    Args:
        qlearning_data: Q-Learning results by difficulty
        ucb_data: UCB results by difficulty
        random_data: Random results by difficulty
        save_path: Path to save CSV (optional)
    
    Returns:
        df: Comparison table as DataFrame
    """
    rows = []
    
    difficulties = ["easy", "medium", "hard"]
    
    for difficulty in difficulties:
        # Success rates
        q_success = sum(1 for ep in qlearning_data[difficulty] if ep['success']) / len(qlearning_data[difficulty]) * 100
        u_success = sum(1 for ep in ucb_data[difficulty] if ep['success']) / len(ucb_data[difficulty]) * 100
        r_success = sum(1 for ep in random_data[difficulty] if ep['success']) / len(random_data[difficulty]) * 100
        
        # Reward statistics
        q_reward_stats = calculate_statistics(qlearning_data[difficulty], 'total_reward')
        u_reward_stats = calculate_statistics(ucb_data[difficulty], 'total_reward')
        r_reward_stats = calculate_statistics(random_data[difficulty], 'total_reward')
        
        # Turn statistics (only for successful episodes)
        q_turns = [ep['turns'] for ep in qlearning_data[difficulty] if ep['success']]
        u_turns = [ep['turns'] for ep in ucb_data[difficulty] if ep['success']]
        r_turns = [ep['turns'] for ep in random_data[difficulty] if ep['success']]
        
        rows.append({
            "Difficulty": difficulty.capitalize(),
            "Q-Learning Success (%)": f"{q_success:.1f}",
            "UCB Success (%)": f"{u_success:.1f}",
            "Random Success (%)": f"{r_success:.1f}",
            "Q-Learning Reward": f"{q_reward_stats['mean']:.1f}±{q_reward_stats['std']:.1f}",
            "UCB Reward": f"{u_reward_stats['mean']:.1f}±{u_reward_stats['std']:.1f}",
            "Random Reward": f"{r_reward_stats['mean']:.1f}±{r_reward_stats['std']:.1f}",
            "Q-Learning Turns": f"{np.mean(q_turns):.1f}±{np.std(q_turns):.1f}" if q_turns else "N/A",
            "UCB Turns": f"{np.mean(u_turns):.1f}±{np.std(u_turns):.1f}" if u_turns else "N/A",
            "Random Turns": f"{np.mean(r_turns):.1f}±{np.std(r_turns):.1f}" if r_turns else "N/A",
        })
    
    df = pd.DataFrame(rows)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Comparison table saved to: {save_path}")
    
    return df


def run_all_statistical_tests(
    qlearning_data: Dict[str, List[Dict]],
    ucb_data: Dict[str, List[Dict]],
    random_data: Dict[str, List[Dict]],
    output_dir: str = "experiments/results"
) -> Dict:
    """
    Run complete statistical analysis suite
    
    Generates:
    - Statistical report (TXT)
    - Comparison table (CSV)
    - Plots with confidence intervals (PNG)
    
    Args:
        qlearning_data: Q-Learning results by difficulty
        ucb_data: UCB results by difficulty
        random_data: Random results by difficulty
        output_dir: Directory to save outputs
    
    Returns:
        all_results: Dictionary with all statistical results
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*80}")
    print("RUNNING STATISTICAL ANALYSIS")
    print(f"{'='*80}\n")
    
    all_results = {}
    
    # Generate statistical report
    report_path = os.path.join(output_dir, f"statistical_report_{timestamp}.txt")
    report_text = generate_statistical_report(
        qlearning_data,
        ucb_data,
        random_data,
        save_path=report_path
    )
    
    # Print report to console
    print(report_text)
    
    # Generate comparison table
    table_path = os.path.join(output_dir, f"comparison_table_{timestamp}.csv")
    comparison_df = create_comparison_table(
        qlearning_data,
        ucb_data,
        random_data,
        save_path=table_path
    )
    
    print("\nCOMPARISON TABLE:")
    print(comparison_df.to_string(index=False))
    print("")
    
    # Generate plots with confidence intervals for hard difficulty
    for difficulty in ["easy", "medium", "hard"]:
        results_for_plot = {
            "Q-Learning": qlearning_data[difficulty],
            "UCB": ucb_data[difficulty],
            "Random": random_data[difficulty]
        }
        
        plot_path = os.path.join(
            ExperimentConfig.VIZ_DIR,
            f"statistical_comparison_{difficulty}_{timestamp}.png"
        )
        
        plot_performance_with_ci(
            results_for_plot,
            metric='total_reward',
            save_path=plot_path,
            show=False
        )
    
    print(f"\n{'='*80}")
    print("STATISTICAL ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
    
    return all_results