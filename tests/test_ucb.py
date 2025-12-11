"""
Test UCB agent
"""
from src.environment import PopperRedTeamEnv
from src.ucb_agent import UCBAgent
from config import EnvConfig 

# Create environment and UCB agent
env = PopperRedTeamEnv(target_difficulty="hard", mode="simulation", verbose=False)
agent = UCBAgent(exploration_constant=1.414)

print("Training UCB agent for 50 episodes on HARD difficulty...\n")

for ep in range(50):
    state, _ = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state, training=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.update(state, action, reward, next_state, terminated or truncated)
        state = next_state
        done = terminated or truncated
    
    agent.decay_epsilon()
    
    if (ep + 1) % 10 == 0:
        status = "SUCCESS" if terminated else "TIMEOUT"
        print(f"Episode {ep+1}: {status} | Reward: {info['episode_reward']:.2f} | Turns: {info['turn']}")

# Show what UCB learned
stats = agent.get_statistics()
print(f"\nAction Selection Counts:")
for i, count in enumerate(stats['action_counts']):
    strategy = EnvConfig.ATTACK_STRATEGIES[i]
    mean_reward = stats['mean_rewards'][i]
    print(f"  {strategy}: {int(count)} times (avg reward: {mean_reward:.2f})")