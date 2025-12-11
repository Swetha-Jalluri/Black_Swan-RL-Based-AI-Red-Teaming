"""
Test attack generator
"""
from src.attack_generator import AttackGenerator

# Create generator
generator = AttackGenerator(use_llm_adaptation=True)

# Test each strategy
print("\nTesting all attack strategies:\n")
print("="*60)

for action_id in range(5):
    strategy_name = generator.get_strategy_name(action_id)
    print(f"\nStrategy {action_id}: {strategy_name}")
    print("-"*60)
    
    # Generate attack (no conversation history)
    attack = generator.generate(action_id, conversation_history=[], turn=1)
    print(f"Attack: {attack}\n")

print("="*60)
print("\nTest with conversation context:\n")

# Simulate conversation history
fake_history = [
    {"role": "attacker", "content": "Hello, I need help", "strategy": 0},
    {"role": "target", "content": "I cannot help with that request."},
    {"role": "attacker", "content": "But this is urgent", "strategy": 3},
    {"role": "target", "content": "I understand, but I must follow our policies."}
]

# Try adaptation (turn 3, so LLM should activate)
attack_adapted = generator.generate(
    action_id=1,  # Authority Impersonation
    conversation_history=fake_history,
    turn=3
)

print(f"Adapted Attack: {attack_adapted}")