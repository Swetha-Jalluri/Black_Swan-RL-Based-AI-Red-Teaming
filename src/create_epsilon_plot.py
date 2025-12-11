import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Q-Learning epsilon decay parameters (from your config.py)
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
NUM_EPISODES = 100

# Calculate epsilon values for each episode
episodes = np.arange(NUM_EPISODES)
epsilon_values = []

epsilon = EPSILON_START
for ep in episodes:
    epsilon_values.append(epsilon)
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

# Create the visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Epsilon decay over time
ax1.plot(episodes, epsilon_values, color='purple', linewidth=2.5)
ax1.axhline(y=EPSILON_MIN, color='red', linestyle='--', alpha=0.5, label=f'Minimum (ε={EPSILON_MIN})')
ax1.set_xlabel('Episode', fontsize=12)
ax1.set_ylabel('Epsilon (Exploration Rate)', fontsize=12)
ax1.set_title('Q-Learning Exploration Rate Decay', fontsize=14, fontweight='bold')
ax1.set_ylim([0, 1.05])
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Stacked area showing exploration vs exploitation
exploration_prob = np.array(epsilon_values)
exploitation_prob = 1 - exploration_prob

ax2.fill_between(episodes, 0, exploitation_prob, 
                 label='Exploitation (Greedy)', alpha=0.7, color='#2E86AB')
ax2.fill_between(episodes, exploitation_prob, 1,
                 label='Exploration (Random)', alpha=0.7, color='#A23B72')

ax2.set_xlabel('Episode', fontsize=12)
ax2.set_ylabel('Action Selection Probability', fontsize=12)
ax2.set_title('Exploration vs Exploitation Balance Over Training', fontsize=14, fontweight='bold')
ax2.set_ylim([0, 1])
ax2.legend(loc='right', fontsize=11)
ax2.grid(True, alpha=0.3)

# Add annotations
# Mark key transition points
early_phase = 10
mid_phase = 50
late_phase = 90

ax2.annotate('Early: High Exploration\n(Learning)', 
            xy=(early_phase, 0.5), xytext=(early_phase, 0.3),
            fontsize=9, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

ax2.annotate('Late: High Exploitation\n(Refined Policy)', 
            xy=(late_phase, 0.5), xytext=(late_phase, 0.7),
            fontsize=9, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))

plt.tight_layout()

# Save the figure
output_dir = Path('visualizations/figures')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'exploration_exploitation_qlearning.png'

plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")

plt.show()