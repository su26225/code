import re
import matplotlib.pyplot as plt
import numpy as np

# Define the file list
files = ['both2\\01', 'both2\\02', 'both2\\03']
key = 3

# Initialize data lists
redundancy_strategy_data = []
disk_scrubbing_freq_data = []

# Iterate over each file
for file in files:
    # Initialize data
    redundancy_strategy = []
    disk_scrubbing_freq = []

    # Read the file content
    with open(file, 'r') as f:
        content = f.read()

        # Use regular expressions to extract data
        matches = re.findall(r"'redundancy_strategy': \((\d+), (\d+)\), 'disk_scrubbing_freq': (\d+)", content)
        for match in matches:
            redundancy_strategy.append(int(match[0]))
            disk_scrubbing_freq.append(int(match[2]))

    # Add the data to the data lists
    redundancy_strategy_data.append(redundancy_strategy)
    disk_scrubbing_freq_data.append(disk_scrubbing_freq)

# Extract the last 200 rounds of disk_scrubbing_freq if needed
from sklearn.metrics import mean_squared_error

# Calculate the mean squared error between Experiment 1 and Experiment 2
mse_exp1_exp2 = mean_squared_error(redundancy_strategy_data[0], redundancy_strategy_data[1])

# Calculate the mean squared error between Experiment 1 and Experiment 3
mse_exp1_exp3 = mean_squared_error(redundancy_strategy_data[0], redundancy_strategy_data[2])

# Calculate the mean squared error between Experiment 2 and Experiment 3
mse_exp2_exp3 = mean_squared_error(redundancy_strategy_data[1], redundancy_strategy_data[2])

# Calculate the average of the three mean squared errors
average_mse = (mse_exp1_exp2 + mse_exp1_exp3 + mse_exp2_exp3) / 3
print('average_mse:', average_mse)
