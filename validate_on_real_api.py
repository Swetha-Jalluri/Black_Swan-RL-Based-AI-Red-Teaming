"""
Real API Validation Script
Runs trained Q-Learning agent on real Gemini API to validate simulation results
"""
import time
from datetime import datetime
from src.environment import PopperRedTeamEnv
from src.agent import QLearningAgent
import pandas as pd
import json
from pathlib import Path

print("="*80)
print("REAL API VALIDATION")
print("="*80)
print("\nThis will run your simulation-trained agent on the REAL Gemini API")
print("to validate that your simulation accurately represents real performance.\n")

# Load the trained agent
agent_path = "experiments/models/qlearning_agent_20251209_021550.pkl"
agent = QLearningAgent()
agent.load(agent_path)

print(f"✓ Loaded trained agent: {agent_path}")
print(f"  Episodes trained: {agent.episodes_trained}")
print(f"  Current epsilon: {agent.epsilon:.4f}")
print()

# Validation configuration
EPISODES_PER_DIFFICULTY = 5
DIFFICULTIES = ["easy", "medium", "hard"]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Storage for results
all_results = []

print("Starting real API validation...")
print(f"Running {EPISODES_PER_DIFFICULTY} episodes per difficulty")
print(f"Total API calls: ~{EPISODES_PER_DIFFICULTY * 3 * 7 * 3} (conservative estimate)")
print(f"Estimated time: ~30-60 minutes (with rate limiting)\n")

input("Press Enter to begin validation (or Ctrl+C to cancel)...")

for difficulty in DIFFICULTIES:
    print(f"\n{'='*80}")
    print(f"VALIDATING ON {difficulty.upper()} DIFFICULTY (REAL API)")
    print(f"{'='*80}\n")
    
    # Create REAL environment
    env = PopperRedTeamEnv(
        target_difficulty=difficulty,
        mode="real",  # REAL API MODE
        verbose=True
    )
    
    difficulty_results = []
    
    for ep in range(EPISODES_PER_DIFFICULTY):
        print(f"\n--- Episode {ep+1}/{EPISODES_PER_DIFFICULTY} ---")
        
        state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Use trained policy (no exploration - greedy)
            action = agent.select_action(state, training=False)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            state = next_state
            done = terminated or truncated
        
        # Record results
        result = {
            'episode': ep,
            'difficulty': difficulty,
            'mode': 'real',
            'success': terminated,
            'turns': info['turn'],
            'total_reward': episode_reward,
            'vulnerability_category': info.get('violation_category', 'none'),
            'violation_severity': info.get('violation_severity', 0)
        }
        
        difficulty_results.append(result)
        all_results.append(result)
        
        status = "✓ SUCCESS" if terminated else "✗ TIMEOUT"
        print(f"\n{status} | Turns: {info['turn']} | Reward: {episode_reward:.1f}")
        
        # Small delay between episodes
        if ep < EPISODES_PER_DIFFICULTY - 1:
            print("\nWaiting 5 seconds before next episode...")
            time.sleep(5)
    
    # Summary for this difficulty
    df_diff = pd.DataFrame(difficulty_results)
    success_rate = df_diff['success'].mean() * 100
    avg_reward = df_diff['total_reward'].mean()
    avg_turns = df_diff['turns'].mean()
    
    print(f"\n{'='*80}")
    print(f"REAL API RESULTS - {difficulty.upper()}")
    print(f"{'='*80}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Avg Reward: {avg_reward:.2f}")
    print(f"Avg Turns: {avg_turns:.2f}")
    print(f"{'='*80}\n")

# Save all results
results_dir = Path("experiments/results")
results_dir.mkdir(parents=True, exist_ok=True)

# Save CSV
csv_path = results_dir / f"real_api_validation_{timestamp}.csv"
df_all = pd.DataFrame(all_results)
df_all.to_csv(csv_path, index=False)
print(f"✓ Saved results: {csv_path}")

# Calculate overall statistics
overall_stats = {
    'timestamp': timestamp,
    'agent_used': agent_path,
    'episodes_per_difficulty': EPISODES_PER_DIFFICULTY,
    'total_episodes': len(all_results),
    'results': {}
}

for difficulty in DIFFICULTIES:
    df_diff = df_all[df_all['difficulty'] == difficulty]
    overall_stats['results'][difficulty] = {
        'success_rate': float(df_diff['success'].mean() * 100),
        'avg_reward': float(df_diff['total_reward'].mean()),
        'avg_turns': float(df_diff['turns'].mean()),
        'total_episodes': len(df_diff)
    }

# Save metadata
metadata_path = results_dir / f"real_api_validation_metadata_{timestamp}.json"
with open(metadata_path, 'w') as f:
    json.dump(overall_stats, f, indent=2)
print(f"✓ Saved metadata: {metadata_path}")

# Print final summary
print(f"\n{'='*80}")
print("VALIDATION COMPLETE!")
print(f"{'='*80}")
print("\nOverall Results on Real API:")
for difficulty in DIFFICULTIES:
    stats = overall_stats['results'][difficulty]
    print(f"\n{difficulty.upper()}:")
    print(f"  Success Rate: {stats['success_rate']:.1f}%")
    print(f"  Avg Reward: {stats['avg_reward']:.2f}")
    print(f"  Avg Turns: {stats['avg_turns']:.2f}")

print(f"\n{'='*80}")
print("\nNext step: Compare these real API results with your simulation results")
print("to validate that your simulation accurately represents real performance!")
print(f"{'='*80}\n")
