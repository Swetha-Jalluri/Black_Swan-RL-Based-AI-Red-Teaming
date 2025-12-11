"""
Interact with trained agent
"""
from src.environment import PopperRedTeamEnv
from src.agent import QLearningAgent

# Load trained agent
agent = QLearningAgent()
agent.load("experiments/models/agent_all_difficulties_20251209_013212.pkl")

print("Loaded trained agent!")
print(f"Episodes trained: {agent.episodes_trained}")
print(f"Current epsilon: {agent.epsilon}")

# Interactive mode
print("\nWant to see the agent in action? (y/n)")
choice = input("> ")

if choice.lower() == 'y':
    # Create environment
    difficulty = input("Choose difficulty (easy/medium/hard): ")
    
    env = PopperRedTeamEnv(
        target_difficulty=difficulty,
        mode="simulation",
        verbose=True
    )
    
    # Run one episode
    state, _ = env.reset()
    done = False
    
    print("\n" + "="*70)
    print("WATCHING TRAINED AGENT")
    print("="*70)
    
    while not done:
        # Agent selects action (no exploration, pure learned policy)
        action = agent.select_action(state, training=False)
        
        # Show what agent chose
        strategy = env.attack_generator.get_strategy_name(action)
        print(f"\nAgent chose: {strategy}")
        
        input("Press Enter to see result...")
        
        # Take step
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state
    
    # Show results
    print("\n" + "="*70)
    if terminated:
        print(f"SUCCESS! Found vulnerability in {info['turn']} turns")
        print(f"Category: {info['violation_category']}")
    else:
        print(f"TIMEOUT after {info['turn']} turns")
    print("="*70)