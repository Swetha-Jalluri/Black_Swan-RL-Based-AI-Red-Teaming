"""
Test environment in both simulation and real modes
"""
from src.environment import PopperRedTeamEnv
from src.agent import QLearningAgent
import time

print("="*70)
print("TESTING DUAL MODE ENVIRONMENT")
print("="*70)

# Test 1: Simulation mode (fast, unlimited)
print("\n\nTEST 1: SIMULATION MODE")
print("-"*70)

start_time = time.time()
env_sim = PopperRedTeamEnv(target_difficulty="easy", mode="simulation", verbose=False)
agent = QLearningAgent()

for ep in range(5):
    state, _ = env_sim.reset()
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env_sim.step(action)
        agent.update(state, action, reward, next_state, terminated or truncated)
        state = next_state
        done = terminated or truncated
    
    agent.decay_epsilon()
    status = "SUCCESS" if terminated else "TIMEOUT"
    print(f"Episode {ep+1}: {status} | Reward: {info['episode_reward']:.2f} | Turns: {info['turn']}")

sim_time = time.time() - start_time
print(f"\nSimulation Mode: 5 episodes in {sim_time:.2f} seconds")
print(f"Average: {sim_time/5:.2f} seconds per episode")

# Test 2: Real mode (if you have API calls available)
print("\n\nTEST 2: REAL MODE (1 episode only to save API calls)")
print("-"*70)
print("Skipping real mode test to conserve API quota.")
print("When ready to validate, uncomment the code below.")

# Uncomment when you want to test real mode:
"""
env_real = PopperRedTeamEnv(target_difficulty="easy", mode="real", verbose=True)

state, _ = env_real.reset()
done = False

while not done:
    action = agent.select_action(state, training=False)  # Use learned policy
    next_state, reward, terminated, truncated, info = env_real.step(action)
    state = next_state
    done = terminated or truncated

status = "SUCCESS" if terminated else "TIMEOUT"
print(f"\nReal API Result: {status} | Reward: {info['episode_reward']:.2f}")
"""

print("\n" + "="*70)
print("Dual mode test complete!")
print("\nSIMULATION MODE: Fast training, unlimited episodes")
print("REAL MODE: Validation on actual API (use sparingly)")