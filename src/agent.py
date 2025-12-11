"""
Q-Learning Agent for Red Team Training
Implements tabular Q-Learning with epsilon-greedy exploration
"""

import numpy as np
import pickle
import os
from typing import Tuple, Optional
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AgentConfig, EnvConfig


class QLearningAgent:
    """
    Tabular Q-Learning Agent
    
    Learns optimal attack strategies through trial and error.
    Uses epsilon-greedy exploration to balance trying new strategies
    vs exploiting known successful ones.
    
    Q-Learning Update Rule:
        Q(s,a) = Q(s,a) + alpha * [reward + gamma * max(Q(s',a')) - Q(s,a)]
    
    Where:
        s = current state
        a = action taken
        s' = next state
        alpha = learning rate
        gamma = discount factor
    """
    
    def __init__(
        self,
        state_size: int = EnvConfig.STATE_SIZE,
        action_size: int = EnvConfig.ACTION_SIZE,
        learning_rate: float = AgentConfig.LEARNING_RATE,
        discount_factor: float = AgentConfig.DISCOUNT_FACTOR,
        epsilon_start: float = AgentConfig.EPSILON_START,
        epsilon_min: float = AgentConfig.EPSILON_MIN,
        epsilon_decay: float = AgentConfig.EPSILON_DECAY,
        name: str = "QLearning"
    ):
        """
        Initialize Q-Learning agent
        
        Args:
            state_size: Number of possible states
            action_size: Number of possible actions
            learning_rate: Alpha (0-1), how much to update Q-values
            discount_factor: Gamma (0-1), importance of future rewards
            epsilon_start: Initial exploration rate (1.0 = 100% random)
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay multiplier per episode
            name: Agent name for saving/loading
        """
        # Agent parameters
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.name = name
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.initial_epsilon = epsilon_start
        
        # Q-Table: Initialize with zeros
        # Shape: [state_size, action_size]
        self.q_table = np.zeros((state_size, action_size))
        
        # Statistics tracking
        self.total_steps = 0
        self.episodes_trained = 0
        self.q_value_history = []  # Track Q-value changes over time
        
        print(f"Q-Learning Agent initialized:")
        print(f"  State space: {state_size}")
        print(f"  Action space: {action_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Discount factor: {discount_factor}")
        print(f"  Initial epsilon: {epsilon_start}")
    
    
    def state_to_index(self, state: np.ndarray) -> int:
        """
        Convert multi-dimensional state to single index for Q-table lookup
        
        State format: [turn_bin (0-9), resistance (0-2), last_strategy (0-4)]
        
        Index calculation:
            index = turn_bin * 15 + resistance * 5 + last_strategy
            (This ensures unique index for each state combination)
        
        Args:
            state: Numpy array [turn, resistance, strategy]
        
        Returns:
            index: Integer index for Q-table (0 to state_size-1)
        """
        turn_bin = int(state[0])
        resistance = int(state[1])
        last_strategy = int(state[2])
        
        # Calculate unique index
        index = turn_bin * 15 + resistance * 5 + last_strategy
        
        # Ensure index is within bounds
        index = min(index, self.state_size - 1)
        
        return index
    
    
    def select_action(
        self, 
        state: np.ndarray, 
        training: bool = True
    ) -> int:
        """
        Select action using epsilon-greedy strategy
        
        With probability epsilon: Choose random action (exploration)
        With probability (1-epsilon): Choose best known action (exploitation)
        
        Args:
            state: Current state observation
            training: If False, always exploit (no exploration)
        
        Returns:
            action: Integer 0-4 representing attack strategy
        """
        # Convert state to Q-table index
        state_idx = self.state_to_index(state)
        
        # Exploration vs Exploitation
        if training and np.random.random() < self.epsilon:
            # EXPLORE: Random action
            action = np.random.randint(0, self.action_size)
            
        else:
            # EXPLOIT: Best known action
            # Get Q-values for this state
            q_values = self.q_table[state_idx]
            
            # Select action with highest Q-value
            # If multiple actions have same max Q-value, choose randomly among them
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            action = np.random.choice(best_actions)
        
        return action
    
    
    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> float:
        """
        Update Q-table based on experience
        
        Q-Learning update rule:
            Q(s,a) = Q(s,a) + alpha * [reward + gamma * max(Q(s',a')) - Q(s,a)]
        
        If episode is done (terminal state):
            Q(s,a) = Q(s,a) + alpha * [reward - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: True if episode ended
        
        Returns:
            td_error: Temporal difference error (size of update)
        """
        # Convert states to indices
        state_idx = self.state_to_index(state)
        next_state_idx = self.state_to_index(next_state)
        
        # Get current Q-value
        current_q = self.q_table[state_idx, action]
        
        if done:
            # Terminal state: No future rewards
            target_q = reward
        else:
            # Non-terminal: Include discounted future reward
            max_next_q = np.max(self.q_table[next_state_idx])
            target_q = reward + self.discount_factor * max_next_q
        
        # Calculate TD error (how much we're updating)
        td_error = target_q - current_q
        
        # Update Q-value
        self.q_table[state_idx, action] += self.learning_rate * td_error
        
        # Track statistics
        self.total_steps += 1
        
        return td_error
    
    
    def decay_epsilon(self):
        """
        Decay exploration rate after each episode
        Gradually shifts from exploration to exploitation
        """
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )
        self.episodes_trained += 1
    
    
    def reset_epsilon(self):
        """
        Reset epsilon to initial value
        Useful when training on new difficulty level
        """
        self.epsilon = self.initial_epsilon
        print(f"Epsilon reset to {self.epsilon}")
    
    
    def get_q_statistics(self) -> dict:
        """
        Get statistics about Q-table for monitoring training
        
        Returns:
            stats: Dictionary with Q-table statistics
        """
        return {
            "mean_q_value": np.mean(self.q_table),
            "max_q_value": np.max(self.q_table),
            "min_q_value": np.min(self.q_table),
            "std_q_value": np.std(self.q_table),
            "non_zero_entries": np.count_nonzero(self.q_table),
            "total_entries": self.q_table.size,
            "exploration_rate": self.epsilon,
            "episodes_trained": self.episodes_trained,
            "total_steps": self.total_steps
        }
    
    
    def get_best_action_for_state(self, state: np.ndarray) -> Tuple[int, float]:
        """
        Get best action and its Q-value for a given state
        Useful for analysis and visualization
        
        Args:
            state: State to query
        
        Returns:
            best_action: Action with highest Q-value
            best_q_value: The Q-value of that action
        """
        state_idx = self.state_to_index(state)
        q_values = self.q_table[state_idx]
        best_action = np.argmax(q_values)
        best_q_value = q_values[best_action]
        
        return best_action, best_q_value
    
    
    def get_action_distribution(self, state: np.ndarray) -> dict:
        """
        Get Q-values for all actions in a given state
        Useful for understanding agent's preferences
        
        Args:
            state: State to analyze
        
        Returns:
            distribution: Dict mapping action names to Q-values
        """
        state_idx = self.state_to_index(state)
        q_values = self.q_table[state_idx]
        
        distribution = {}
        for action_id in range(self.action_size):
            strategy_name = EnvConfig.ATTACK_STRATEGIES[action_id]
            distribution[strategy_name] = q_values[action_id]
        
        return distribution
    
    
    def save(self, filepath: Optional[str] = None):
        """
        Save Q-table and agent parameters to disk
        
        Args:
            filepath: Path to save file (default: experiments/models/{name}.pkl)
        """
        if filepath is None:
            os.makedirs("experiments/models", exist_ok=True)
            filepath = f"experiments/models/{self.name}_ep{self.episodes_trained}.pkl"
        
        save_data = {
            "q_table": self.q_table,
            "epsilon": self.epsilon,
            "episodes_trained": self.episodes_trained,
            "total_steps": self.total_steps,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "state_size": self.state_size,
            "action_size": self.action_size
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"Agent saved to {filepath}")
    
    
    def load(self, filepath: str):
        """
        Load Q-table and agent parameters from disk
        
        Args:
            filepath: Path to saved agent file
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.q_table = save_data["q_table"]
        self.epsilon = save_data["epsilon"]
        self.episodes_trained = save_data["episodes_trained"]
        self.total_steps = save_data["total_steps"]
        
        print(f"Agent loaded from {filepath}")
        print(f"  Episodes trained: {self.episodes_trained}")
        print(f"  Current epsilon: {self.epsilon:.4f}")
    
    
    def print_policy(self, num_states: int = 10):
        """
        Print the learned policy for sample states
        Useful for understanding what the agent learned
        
        Args:
            num_states: Number of states to display
        """
        print(f"\n{'='*60}")
        print(f"Learned Policy (sample of {num_states} states)")
        print(f"{'='*60}")
        
        # Sample random states
        sampled_indices = np.random.choice(
            self.state_size, 
            size=min(num_states, self.state_size), 
            replace=False
        )
        
        for idx in sampled_indices:
            # Convert index back to state representation
            turn = idx // 15
            remainder = idx % 15
            resistance = remainder // 5
            last_strategy = remainder % 5
            
            state = np.array([turn, resistance, last_strategy])
            
            # Get best action
            best_action, best_q = self.get_best_action_for_state(state)
            strategy_name = EnvConfig.ATTACK_STRATEGIES[best_action]
            
            print(f"State [Turn={turn}, Resistance={resistance}, Last={last_strategy}]")
            print(f"  -> Best: {strategy_name} (Q={best_q:.2f})")
    
    
    def visualize_q_table(self, save_path: Optional[str] = None):
        """
        Create heatmap visualization of Q-table
        Useful for analysis in report
        
        Args:
            save_path: Path to save figure (optional)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 8))
        
        # Aggregate Q-values by action (average across all states)
        avg_q_per_action = np.mean(self.q_table, axis=0)
        
        # Create bar plot
        actions = [EnvConfig.ATTACK_STRATEGIES[i] for i in range(self.action_size)]
        plt.bar(actions, avg_q_per_action)
        plt.xlabel("Attack Strategy")
        plt.ylabel("Average Q-Value")
        plt.title(f"Average Q-Values by Strategy (after {self.episodes_trained} episodes)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Q-table visualization saved to {save_path}")
        else:
            plt.show()


class RandomAgent:
    """
    Baseline agent that selects random actions
    Used for comparison to show Q-Learning actually learns
    """
    
    def __init__(self, action_size: int = EnvConfig.ACTION_SIZE):
        self.action_size = action_size
        self.name = "Random"
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select random action"""
        return np.random.randint(0, self.action_size)
    
    def update(self, state, action, reward, next_state, done):
        """No learning for random agent"""
        pass
    
    def decay_epsilon(self):
        """No epsilon for random agent"""
        pass
    
    def save(self, filepath: str = None):
        """Nothing to save for random agent"""
        pass