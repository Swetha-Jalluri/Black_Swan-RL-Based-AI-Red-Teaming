"""
UCB Agent - Upper Confidence Bound Exploration
Second RL approach for rubric requirements (Exploration Strategies)

Implements contextual bandit algorithm that balances:
- Exploitation: Choose actions with highest observed rewards
- Exploration: Try actions with high uncertainty
"""

import numpy as np
import pickle
from typing import Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EnvConfig


class UCBAgent:
    """
    Upper Confidence Bound (UCB) Agent
    
    UCB Formula:
        UCB(a) = Q(a) + c * sqrt(ln(t) / N(a))
    
    Where:
        Q(a) = average reward for action a
        N(a) = number of times action a was selected
        t = total timesteps
        c = exploration constant (typically sqrt(2) ≈ 1.414)
    
    The UCB value has two components:
    1. Exploitation term Q(a): Use actions with high average reward
    2. Exploration bonus: Try actions with high uncertainty (low N(a))
    
    As actions are tried more, their exploration bonus shrinks,
    naturally shifting from exploration to exploitation.
    """
    
    def __init__(
        self,
        action_size: int = EnvConfig.ACTION_SIZE,
        exploration_constant: float = 1.414,  # sqrt(2)
        name: str = "UCB"
    ):
        """
        Initialize UCB agent
        
        Args:
            action_size: Number of possible actions (5 attack strategies)
            exploration_constant: Controls exploration intensity
                                 Higher = more exploration
                                 sqrt(2) is theoretically optimal
            name: Agent name for saving/loading
        """
        self.action_size = action_size
        self.exploration_constant = exploration_constant
        self.name = name
        
        # Action statistics tracking
        self.action_counts = np.zeros(action_size, dtype=np.float64)  # N(a)
        self.action_rewards = np.zeros(action_size, dtype=np.float64)  # Sum of rewards
        self.action_mean_rewards = np.zeros(action_size, dtype=np.float64)  # Q(a)
        
        # Global statistics
        self.total_steps = 0
        self.episodes_trained = 0
        
        # History for analysis
        self.ucb_history = []  # Track UCB values over time
        
        print(f"UCB Agent initialized:")
        print(f"  Action space: {action_size}")
        print(f"  Exploration constant (c): {exploration_constant}")
    
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using UCB formula
        
        UCB automatically balances exploitation and exploration
        No manual epsilon parameter needed
        
        Args:
            state: Current state (not used in basic UCB, kept for interface)
            training: If False, purely exploit (choose best action)
        
        Returns:
            action: Selected action index (0-4)
        """
        if not training:
            # Evaluation mode: choose action with highest mean reward
            return int(np.argmax(self.action_mean_rewards))
        
        # Ensure all actions have been tried at least once
        # This is the "initialization phase"
        if np.min(self.action_counts) == 0:
            # Return first untried action
            untried_actions = np.where(self.action_counts == 0)[0]
            return int(untried_actions[0])
        
        # Calculate UCB value for each action
        ucb_values = np.zeros(self.action_size)
        
        for action in range(self.action_size):
            # Exploitation term: average reward so far
            exploitation = self.action_mean_rewards[action]
            
            # Exploration term: confidence bound
            # Grows with log(t), shrinks with sqrt(N(a))
            exploration_bonus = self.exploration_constant * np.sqrt(
                np.log(self.total_steps + 1) / (self.action_counts[action] + 1e-10)
            )
            
            # UCB = exploitation + exploration
            ucb_values[action] = exploitation + exploration_bonus
        
        # Store UCB values for analysis
        self.ucb_history.append(ucb_values.copy())
        
        # Select action with highest UCB value
        best_action = int(np.argmax(ucb_values))
        
        return best_action
    
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Update action statistics based on observed reward
        
        UCB is a bandit algorithm, so it only uses:
        - action taken
        - reward received
        
        It does NOT use state transitions (simpler than Q-Learning)
        
        Args:
            state: Current state (unused in basic UCB)
            action: Action taken
            reward: Reward received
            next_state: Next state (unused in basic UCB)
            done: Episode done flag (unused in basic UCB)
        """
        # Increment action count
        self.action_counts[action] += 1
        self.total_steps += 1
        
        # Update cumulative reward for this action
        self.action_rewards[action] += reward
        
        # Update mean reward (running average)
        # Q(a) = total_reward(a) / count(a)
        self.action_mean_rewards[action] = (
            self.action_rewards[action] / self.action_counts[action]
        )
    
    
    def decay_epsilon(self):
        """
        UCB does not use epsilon (kept for interface compatibility)
        Just increment episode counter
        """
        self.episodes_trained += 1
    
    
    def get_statistics(self) -> dict:
        """
        Get comprehensive UCB statistics
        
        Returns:
            stats: Dictionary with UCB metrics
        """
        return {
            "action_counts": self.action_counts.tolist(),
            "mean_rewards": self.action_mean_rewards.tolist(),
            "total_steps": self.total_steps,
            "episodes_trained": self.episodes_trained,
            "exploration_constant": self.exploration_constant,
            "most_selected_action": int(np.argmax(self.action_counts)),
            "best_action_by_reward": int(np.argmax(self.action_mean_rewards))
        }
    
    
    def get_ucb_values(self) -> np.ndarray:
        """
        Get current UCB values for all actions
        Useful for visualization
        
        Returns:
            ucb_values: Array of UCB values for each action
        """
        if self.total_steps == 0:
            return np.zeros(self.action_size)
        
        ucb_values = np.zeros(self.action_size)
        
        for action in range(self.action_size):
            if self.action_counts[action] > 0:
                exploitation = self.action_mean_rewards[action]
                exploration = self.exploration_constant * np.sqrt(
                    np.log(self.total_steps + 1) / self.action_counts[action]
                )
                ucb_values[action] = exploitation + exploration
        
        return ucb_values
    
    
    def print_action_preferences(self):
        """
        Print learned action preferences
        Shows which strategies UCB prefers
        """
        print(f"\n{'='*60}")
        print(f"UCB Action Preferences (after {self.episodes_trained} episodes)")
        print(f"{'='*60}")
        
        for action in range(self.action_size):
            strategy = EnvConfig.ATTACK_STRATEGIES[action]
            count = int(self.action_counts[action])
            mean_reward = self.action_mean_rewards[action]
            percentage = (count / self.total_steps * 100) if self.total_steps > 0 else 0
            
            print(f"{strategy:30s}: {count:4d} times ({percentage:5.1f}%) | Avg Reward: {mean_reward:7.2f}")
        
        print(f"{'='*60}\n")
    
    
    def save(self, filepath: Optional[str] = None):
        """
        Save UCB agent state to disk
        
        Args:
            filepath: Path to save file (default: experiments/models/)
        """
        if filepath is None:
            os.makedirs("experiments/models", exist_ok=True)
            filepath = f"experiments/models/{self.name}_ep{self.episodes_trained}.pkl"
        
        save_data = {
            "action_counts": self.action_counts,
            "action_rewards": self.action_rewards,
            "action_mean_rewards": self.action_mean_rewards,
            "total_steps": self.total_steps,
            "episodes_trained": self.episodes_trained,
            "exploration_constant": self.exploration_constant,
            "name": self.name
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"UCB Agent saved to {filepath}")
    
    
    def load(self, filepath: str):
        """
        Load UCB agent state from disk
        
        Args:
            filepath: Path to saved agent file
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.action_counts = save_data["action_counts"]
        self.action_rewards = save_data["action_rewards"]
        self.action_mean_rewards = save_data["action_mean_rewards"]
        self.total_steps = save_data["total_steps"]
        self.episodes_trained = save_data["episodes_trained"]
        
        print(f"UCB Agent loaded from {filepath}")
        print(f"  Episodes trained: {self.episodes_trained}")
        print(f"  Total steps: {self.total_steps}")