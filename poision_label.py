import sys
import pandas as pd
import numpy as np

# Check if the poisoning percentage is provided
if len(sys.argv) < 2:
    print("Usage: python3 poision_label.py <percentage>")
    sys.exit(1)

# Load the dataset labels from samples.csv
try:
    samples = pd.read_csv('samples.csv')
except FileNotFoundError:
    print("Error: samples.csv not found. Make sure you are in the correct directory.")
    sys.exit(1)

# --- This is a good place for a quick debug line if you face more errors ---
# print("Columns found in samples.csv:", samples.columns.tolist())

# Get the poisoning percentage from the command line argument
poison_level = int(sys.argv[1])

# Calculate the number of samples to poison
num_poison = int(len(samples) * poison_level / 100)

# Randomly select unique indices to poison
np.random.seed(42) # for reproducibility
indices_to_poison = np.random.choice(samples.index, num_poison, replace=False)

# -------------------------
target_column_name = 'species' 

# Flip the labels for the selected indices
unique_labels = sorted(samples[target_column_name].unique())
for idx in indices_to_poison:
    current_label = samples.at[idx, target_column_name]
    current_label_index = unique_labels.index(current_label)
    new_label = unique_labels[(current_label_index + 1) % len(unique_labels)]
    samples.at[idx, target_column_name] = new_label

# Save the poisoned data back to samples.csv
samples.to_csv('samples.csv', index=False)

print(f"Poisoned {num_poison} labels ({poison_level}%) in column '{target_column_name}' out of {len(samples)} total samples.")
