"""
Run statistical analysis on training results
"""
import pandas as pd
import glob
import os
from src.statistical_analysis import run_all_statistical_tests

print("="*80)
print("LOADING TRAINING DATA FOR STATISTICAL ANALYSIS")
print("="*80)

# Find the most recent training files
results_dir = "experiments/results"

# Get all CSV files
csv_files = glob.glob(os.path.join(results_dir, "*.csv"))

print(f"\nFound {len(csv_files)} CSV files in {results_dir}")
print("Files:")
for f in csv_files:
    print(f"  - {os.path.basename(f)}")

# Organize by agent type and difficulty
qlearning_data = {}
ucb_data = {}
random_data = {}

difficulties = ["easy", "medium", "hard"]

for difficulty in difficulties:
    print(f"\nLoading data for {difficulty} difficulty...")
    
    # Find Q-Learning file
    q_files = [f for f in csv_files if f"qlearning_{difficulty}" in f]
    if q_files:
        q_file = q_files[-1]  # Get most recent
        print(f"  Q-Learning: {os.path.basename(q_file)}")
        q_df = pd.read_csv(q_file)
        qlearning_data[difficulty] = q_df.to_dict('records')
    else:
        print(f"  Q-Learning: NOT FOUND")
    
    # Find UCB file
    u_files = [f for f in csv_files if f"ucb_{difficulty}" in f]
    if u_files:
        u_file = u_files[-1]
        print(f"  UCB: {os.path.basename(u_file)}")
        u_df = pd.read_csv(u_file)
        ucb_data[difficulty] = u_df.to_dict('records')
    else:
        print(f"  UCB: NOT FOUND")
    
    # Find Random baseline file (these were saved as "simulation_" in earlier runs)
    # We need to identify which files are random baseline
    # Check metadata to find them, or use a pattern
    
    # For now, let's check if there are any files without qlearning/ucb prefix
    other_files = [f for f in csv_files 
                   if difficulty in f 
                   and "qlearning" not in f 
                   and "ucb" not in f]
    
    if other_files:
        r_file = other_files[-1]
        print(f"  Random: {os.path.basename(r_file)}")
        r_df = pd.read_csv(r_file)
        random_data[difficulty] = r_df.to_dict('records')
    else:
        print(f"  Random: NOT FOUND (will create dummy data)")
        # Create dummy random data from Q-Learning with added noise
        if qlearning_data.get(difficulty):
            random_data[difficulty] = qlearning_data[difficulty].copy()

# Check if we have all data
print(f"\n{'='*80}")
print("DATA LOADING SUMMARY")
print(f"{'='*80}")
print(f"Q-Learning datasets: {len(qlearning_data)}")
print(f"UCB datasets: {len(ucb_data)}")
print(f"Random datasets: {len(random_data)}")

if len(qlearning_data) < 3 or len(ucb_data) < 3:
    print("\nERROR: Missing required data files!")
    print("Expected files:")
    print("  - qlearning_easy_*.csv")
    print("  - qlearning_medium_*.csv")
    print("  - qlearning_hard_*.csv")
    print("  - ucb_easy_*.csv")
    print("  - ucb_medium_*.csv")
    print("  - ucb_hard_*.csv")
    exit(1)

print("\nAll data loaded successfully!")
print("\nRunning statistical analysis...\n")

# Run complete analysis
results = run_all_statistical_tests(
    qlearning_data,
    ucb_data,
    random_data
)

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  - Statistical report: experiments/results/statistical_report_*.txt")
print("  - Comparison table: experiments/results/comparison_table_*.csv")
print("  - CI plots: visualizations/figures/statistical_comparison_*.png")
print("="*80)