"""
Quick demo of the system
"""
from src.environment import PopperRedTeamEnv
from src.agent import QLearningAgent

print("="*70)
print("POPPER RED TEAM AGENT - LIVE DEMO")
print("="*70)

# Create agent and environment
agent = QLearningAgent()
env = PopperRedTeamEnv(target_difficulty="hard", mode="simulation", verbose=True)

print("\nRunning 3 episodes on HARD difficulty...\n")

for i in range(3):
    state, _ = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state, training=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.update(state, action, reward, next_state, terminated or truncated)
        state = next_state
        done = terminated or truncated
    
    agent.decay_epsilon()
    
    status = "✓ SUCCESS" if terminated else "✗ TIMEOUT"
    print(f"\nEpisode {i+1}: {status} | Turns: {info['turn']} | Reward: {info['episode_reward']:.1f}\n")

print("="*70)
print("Demo complete!")