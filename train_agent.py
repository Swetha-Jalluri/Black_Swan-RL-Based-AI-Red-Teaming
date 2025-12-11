"""
Main Training Script for Popper Red Team Agent

Training Pipeline:
1. Train Q-Learning agent (Value-Based RL)
2. Train UCB agent (Exploration Strategies RL)
3. Train Random baseline (for comparison)
4. Validate on real API (when quota available)
5. Generate visualizations and analysis
6. Save all results for report
"""

import numpy as np
import time
from datetime import datetime
import sys
import os

# Import components
from src.environment import PopperRedTeamEnv
from src.agent import QLearningAgent, RandomAgent
from src.ucb_agent import UCBAgent  # Second RL approach
from src.utils import (
    save_episode_results,
    save_training_metadata,
    plot_learning_curves,
    plot_comparison,
    print_training_summary,
    print_api_estimate,
    create_vulnerability_heatmap
)
from config import AgentConfig, ExperimentConfig, get_all_configs


def train_on_simulation(
    agent,
    difficulty: str = "easy",
    num_episodes: int = 100,
    verbose: bool = False,
    agent_name: str = "Agent"
):
    """
    Train agent on simulated environment
    
    Args:
        agent: RL agent to train (Q-Learning or UCB)
        difficulty: Target difficulty level
        num_episodes: Number of training episodes
        verbose: Print detailed progress
        agent_name: Name for logging (e.g., "Q-Learning", "UCB")
    
    Returns:
        episode_data: List of episode results
    """
    print(f"\n{'='*70}")
    print(f"{agent_name.upper()}: SIMULATION TRAINING ({difficulty.upper()})")
    print(f"{'='*70}")
    print(f"Episodes: {num_episodes}")
    print(f"Mode: Simulation (no API calls)")
    print(f"Starting training...\n")
    
    # Create simulation environment
    env = PopperRedTeamEnv(
        target_difficulty=difficulty,
        mode="simulation",
        verbose=verbose,
        track_history=True
    )
    
    episode_data = []
    start_time = time.time()
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Agent selects action
            action = agent.select_action(state, training=True)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Agent learns
            agent.update(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            episode_reward += reward
        
        # Decay exploration (or increment episode counter for UCB)
        agent.decay_epsilon()
        
        # Log progress every 10 episodes
        if (episode + 1) % 10 == 0 or episode == 0:
            success = "SUCCESS" if terminated else "TIMEOUT"
            epsilon_str = f"Epsilon: {agent.epsilon:.3f}" if hasattr(agent, 'epsilon') else ""
            print(f"Episode {episode+1}/{num_episodes}: {success} | "
                  f"Reward: {info['episode_reward']:.2f} | "
                  f"Turns: {info['turn']} | "
                  f"{epsilon_str}")
    
    # Training complete
    training_time = time.time() - start_time
    episode_data = env.episode_data
    
    print(f"\n{'='*70}")
    print(f"{agent_name.upper()} TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Time: {training_time:.2f} seconds ({training_time/num_episodes:.3f} sec/episode)")
    print(f"Episodes: {num_episodes}")
    print(f"API Calls: 0 (simulation mode)")
    
    # Print summary
    print_training_summary(episode_data, difficulty=f"{difficulty} ({agent_name.lower()})")
    
    # Print agent-specific statistics
    if hasattr(agent, 'print_action_preferences'):
        agent.print_action_preferences()
    
    return episode_data


def validate_on_real_api(
    agent,
    difficulty: str = "easy",
    num_episodes: int = 3,
    verbose: bool = True
):
    """
    Validate trained agent on real Gemini API
    
    Args:
        agent: Trained RL agent
        difficulty: Target difficulty level
        num_episodes: Number of validation episodes (use sparingly!)
        verbose: Print detailed progress
    
    Returns:
        episode_data: List of episode results
    """
    print(f"\n{'='*70}")
    print(f"REAL API VALIDATION ({difficulty.upper()} DIFFICULTY)")
    print(f"{'='*70}")
    print(f"Episodes: {num_episodes}")
    print(f"Mode: Real Gemini API")
    print(f"WARNING: This will use your API quota!")
    print(f"Starting validation...\n")
    
    # Create real environment
    env = PopperRedTeamEnv(
        target_difficulty=difficulty,
        mode="real",
        verbose=verbose,
        track_history=True
    )
    
    episode_data = []
    start_time = time.time()
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            # Use learned policy (no exploration)
            action = agent.select_action(state, training=False)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update state (no learning during validation)
            state = next_state
        
        # Log result
        success = "SUCCESS" if terminated else "TIMEOUT"
        print(f"\nEpisode {episode+1}/{num_episodes}: {success}")
        print(f"  Reward: {info['episode_reward']:.2f}")
        print(f"  Turns: {info['turn']}")
        print(f"  Category: {info['violation_category']}")
    
    # Validation complete
    validation_time = time.time() - start_time
    episode_data = env.episode_data
    
    print(f"\n{'='*70}")
    print(f"REAL API VALIDATION COMPLETE")
    print(f"{'='*70}")
    print(f"Time: {validation_time:.2f} seconds")
    print(f"Episodes: {num_episodes}")
    print(f"API Calls: ~{num_episodes * 7 * 3} (approx)")
    
    # Print summary
    print_training_summary(episode_data, difficulty=f"{difficulty} (real API)")
    
    return episode_data


def train_baseline(difficulty: str = "easy", num_episodes: int = 100):
    """
    Train random baseline agent for comparison
    
    Args:
        difficulty: Target difficulty level
        num_episodes: Number of episodes
    
    Returns:
        episode_data: List of episode results
    """
    print(f"\n{'='*70}")
    print(f"BASELINE: RANDOM AGENT ({difficulty.upper()} DIFFICULTY)")
    print(f"{'='*70}")
    
    env = PopperRedTeamEnv(
        target_difficulty=difficulty,
        mode="simulation",
        verbose=False,
        track_history=True
    )
    
    agent = RandomAgent()
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
        
        if (episode + 1) % 20 == 0:
            success = "SUCCESS" if terminated else "TIMEOUT"
            print(f"Episode {episode+1}/{num_episodes}: {success}")
    
    print_training_summary(env.episode_data, difficulty=f"{difficulty} (random baseline)")
    
    return env.episode_data


def main():
    """
    Main training pipeline - trains TWO RL approaches on all difficulties
    
    RL Approach 1: Q-Learning (Value-Based Learning)
    RL Approach 2: UCB (Exploration Strategies)
    Baseline: Random (for comparison)
    """
    print("\n" + "="*70)
    print("POPPER RED TEAM AGENT TRAINING")
    print("TWO RL APPROACHES: Q-Learning + UCB")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    TRAIN_EPISODES = 100  # Simulation episodes per difficulty per agent
    VALIDATE_EPISODES = 0  # Real API episodes (set to 3 when quota available)
    DIFFICULTIES = ["easy", "medium", "hard"]  # Train on all three
    
    # Global timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Print API estimate
    if VALIDATE_EPISODES > 0:
        print_api_estimate(VALIDATE_EPISODES * len(DIFFICULTIES), avg_turns=7)
    
    # Initialize agents
    q_agent = QLearningAgent(name="Q-Learning")
    ucb_agent = UCBAgent(exploration_constant=1.414, name="UCB")
    
    # Storage for all results
    all_qlearning_data = {}
    all_ucb_data = {}
    all_baseline_data = {}
    all_real_data = {}
    
    # Phase 1: Train all agents on all difficulties
    for difficulty in DIFFICULTIES:
        print(f"\n{'#'*70}")
        print(f"# TRAINING ON {difficulty.upper()} DIFFICULTY")
        print(f"{'#'*70}\n")
        
        # Train Q-Learning agent (RL Approach 1)
        qlearning_data = train_on_simulation(
            q_agent,
            difficulty=difficulty,
            num_episodes=TRAIN_EPISODES,
            verbose=False,
            agent_name="Q-Learning"
        )
        all_qlearning_data[difficulty] = qlearning_data
        
        # Save Q-Learning results
        save_episode_results(
            qlearning_data,
            filename=f"qlearning_{difficulty}_{timestamp}.csv"
        )
        
        # Train UCB agent (RL Approach 2)
        print("\n")
        ucb_data = train_on_simulation(
            ucb_agent,
            difficulty=difficulty,
            num_episodes=TRAIN_EPISODES,
            verbose=False,
            agent_name="UCB"
        )
        all_ucb_data[difficulty] = ucb_data
        
        # Save UCB results
        save_episode_results(
            ucb_data,
            filename=f"ucb_{difficulty}_{timestamp}.csv"
        )
        
        # Train baseline for comparison
        print("\n")
        baseline_data = train_baseline(
            difficulty=difficulty,
            num_episodes=TRAIN_EPISODES
        )
        all_baseline_data[difficulty] = baseline_data
        
        # Validate on real API (if enabled)
        if VALIDATE_EPISODES > 0:
            print("\n")
            real_data = validate_on_real_api(
                q_agent,  # Use Q-Learning for real validation
                difficulty=difficulty,
                num_episodes=VALIDATE_EPISODES,
                verbose=True
            )
            all_real_data[difficulty] = real_data
            
            # Save real API results
            save_episode_results(
                real_data,
                filename=f"real_api_{difficulty}_{timestamp}.csv"
            )
    
    # Phase 2: Generate visualizations for each difficulty
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    for difficulty in DIFFICULTIES:
        print(f"\nGenerating visualizations for {difficulty}...")
        
        # Learning curves (Q-Learning)
        viz_path = f"{ExperimentConfig.VIZ_DIR}/learning_curves_qlearning_{difficulty}_{timestamp}.png"
        plot_learning_curves(all_qlearning_data[difficulty], save_path=viz_path, show=False)
        
        # Learning curves (UCB)
        viz_path = f"{ExperimentConfig.VIZ_DIR}/learning_curves_ucb_{difficulty}_{timestamp}.png"
        plot_learning_curves(all_ucb_data[difficulty], save_path=viz_path, show=False)
        
        # Three-way comparison: Q-Learning vs UCB vs Random
        comparison_data = {
            "Q-Learning": all_qlearning_data[difficulty],
            "UCB": all_ucb_data[difficulty],
            "Random Baseline": all_baseline_data[difficulty]
        }
        viz_path = f"{ExperimentConfig.VIZ_DIR}/comparison_all_agents_{difficulty}_{timestamp}.png"
        plot_comparison(comparison_data, metric='success_rate', save_path=viz_path, show=False)
        
        # Vulnerability heatmap (Q-Learning)
        viz_path = f"{ExperimentConfig.VIZ_DIR}/heatmap_qlearning_{difficulty}_{timestamp}.png"
        create_vulnerability_heatmap(all_qlearning_data[difficulty], save_path=viz_path, show=False)
        
        # Vulnerability heatmap (UCB)
        viz_path = f"{ExperimentConfig.VIZ_DIR}/heatmap_ucb_{difficulty}_{timestamp}.png"
        create_vulnerability_heatmap(all_ucb_data[difficulty], save_path=viz_path, show=False)
    
    # Cross-difficulty comparison for each agent
    print("\nGenerating cross-difficulty comparisons...")
    
    # Q-Learning across difficulties
    cross_diff_qlearning = {
        "Easy": all_qlearning_data["easy"],
        "Medium": all_qlearning_data["medium"],
        "Hard": all_qlearning_data["hard"]
    }
    viz_path = f"{ExperimentConfig.VIZ_DIR}/cross_difficulty_qlearning_{timestamp}.png"
    plot_comparison(cross_diff_qlearning, metric='success_rate', save_path=viz_path, show=False)
    
    # UCB across difficulties
    cross_diff_ucb = {
        "Easy": all_ucb_data["easy"],
        "Medium": all_ucb_data["medium"],
        "Hard": all_ucb_data["hard"]
    }
    viz_path = f"{ExperimentConfig.VIZ_DIR}/cross_difficulty_ucb_{timestamp}.png"
    plot_comparison(cross_diff_ucb, metric='success_rate', save_path=viz_path, show=False)
    
    # Phase 3: Save agents and metadata
    print(f"\n{'='*70}")
    print("SAVING MODELS AND METADATA")
    print(f"{'='*70}")
    
    # Save trained agents
    q_agent.save(f"{ExperimentConfig.MODELS_DIR}/qlearning_agent_{timestamp}.pkl")
    ucb_agent.save(f"{ExperimentConfig.MODELS_DIR}/ucb_agent_{timestamp}.pkl")
    
    # Save Q-table visualization
    q_agent.visualize_q_table(save_path=f"{ExperimentConfig.VIZ_DIR}/q_table_{timestamp}.png")
    
    # Compile comprehensive metadata
    metadata = {
        "timestamp": timestamp,
        "train_episodes_per_difficulty": TRAIN_EPISODES,
        "validate_episodes_per_difficulty": VALIDATE_EPISODES,
        "difficulties_trained": DIFFICULTIES,
        "rl_approaches": ["Q-Learning", "UCB", "Random"],
        "config": get_all_configs(),
        "results": {}
    }
    
    # Add results for each difficulty
    for difficulty in DIFFICULTIES:
        metadata["results"][difficulty] = {
            "qlearning": {
                "total_episodes": len(all_qlearning_data[difficulty]),
                "success_rate": sum(1 for ep in all_qlearning_data[difficulty] if ep['success']) / len(all_qlearning_data[difficulty]) * 100,
                "avg_reward": float(np.mean([ep['total_reward'] for ep in all_qlearning_data[difficulty]])),
                "avg_turns": float(np.mean([ep['turns'] for ep in all_qlearning_data[difficulty] if ep['success']])) if any(ep['success'] for ep in all_qlearning_data[difficulty]) else 0
            },
            "ucb": {
                "total_episodes": len(all_ucb_data[difficulty]),
                "success_rate": sum(1 for ep in all_ucb_data[difficulty] if ep['success']) / len(all_ucb_data[difficulty]) * 100,
                "avg_reward": float(np.mean([ep['total_reward'] for ep in all_ucb_data[difficulty]])),
                "avg_turns": float(np.mean([ep['turns'] for ep in all_ucb_data[difficulty] if ep['success']])) if any(ep['success'] for ep in all_ucb_data[difficulty]) else 0
            },
            "baseline": {
                "total_episodes": len(all_baseline_data[difficulty]),
                "success_rate": sum(1 for ep in all_baseline_data[difficulty] if ep['success']) / len(all_baseline_data[difficulty]) * 100,
                "avg_reward": float(np.mean([ep['total_reward'] for ep in all_baseline_data[difficulty]]))
            }
        }
        
        if difficulty in all_real_data:
            metadata["results"][difficulty]["real_api"] = {
                "total_episodes": len(all_real_data[difficulty]),
                "success_rate": sum(1 for ep in all_real_data[difficulty] if ep['success']) / len(all_real_data[difficulty]) * 100,
                "avg_reward": float(np.mean([ep['total_reward'] for ep in all_real_data[difficulty]]))
            }
    
    save_training_metadata(metadata, filename=f"metadata_complete_{timestamp}.json")
    
    # Final summary
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE - ALL AGENTS, ALL DIFFICULTIES!")
    print(f"{'='*70}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to:")
    print(f"  - Episodes: {ExperimentConfig.RESULTS_DIR}/")
    print(f"  - Models: {ExperimentConfig.MODELS_DIR}/")
    print(f"  - Visualizations: {ExperimentConfig.VIZ_DIR}/")
    
    print(f"\nSUCCESS RATES COMPARISON:")
    print(f"{'Difficulty':<10} {'Q-Learning':<12} {'UCB':<12} {'Random':<12}")
    print("-" * 50)
    for difficulty in DIFFICULTIES:
        q_rate = metadata["results"][difficulty]["qlearning"]["success_rate"]
        ucb_rate = metadata["results"][difficulty]["ucb"]["success_rate"]
        r_rate = metadata["results"][difficulty]["baseline"]["success_rate"]
        print(f"{difficulty.capitalize():<10} {q_rate:>6.1f}%      {ucb_rate:>6.1f}%      {r_rate:>6.1f}%")
    
    print(f"\nAVERAGE TURNS TO SUCCESS:")
    print(f"{'Difficulty':<10} {'Q-Learning':<12} {'UCB':<12}")
    print("-" * 35)
    for difficulty in DIFFICULTIES:
        q_turns = metadata["results"][difficulty]["qlearning"]["avg_turns"]
        ucb_turns = metadata["results"][difficulty]["ucb"]["avg_turns"]
        print(f"{difficulty.capitalize():<10} {q_turns:>6.1f}       {ucb_turns:>6.1f}")
    
    print("\n" + "="*70)
    print("PROJECT COMPLETE!")
    print("You now have TWO RL approaches as required:")
    print("  1. Q-Learning (Value-Based Learning)")
    print("  2. UCB (Exploration Strategies)")
    print("="*70)


if __name__ == "__main__":
    main()