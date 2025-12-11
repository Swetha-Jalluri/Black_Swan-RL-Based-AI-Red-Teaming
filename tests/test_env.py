"""
Test integrated environment with all real components
"""
from src.environment import PopperRedTeamEnv
from src.agent import QLearningAgent

print("="*70)
print("TESTING INTEGRATED ENVIRONMENT")
print("="*70)

# Create environment with real components
env = PopperRedTeamEnv(target_difficulty="easy", verbose=True)
agent = QLearningAgent()

# Run 3 episodes
for ep in range(3):
    state, info = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state, training=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.update(state, action, reward, next_state, terminated or truncated)
        state = next_state
        done = terminated or truncated
    
    agent.decay_epsilon()
    
    status = "SUCCESS" if terminated else "TIMEOUT"
    print(f"\nEpisode {ep+1}: {status} | Reward: {info['episode_reward']:.2f}\n")

print("="*70)
print("Integration test complete!")