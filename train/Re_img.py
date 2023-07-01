import re
import matplotlib.pyplot as plt
import numpy as np
import statistics

def moving_average(data, window_size):
    """Smooth the data using the moving average method"""
    ret = np.cumsum(data, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1:] / window_size

# Define the file list
files = ['both2\\A','both2\\B','both2\\C','both2\\D']
key = 3

# Define color list
colors = ['r', 'y', 'b', 'g']
label = ['Level_A','Level_B','Level_C','Level_D']

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

    # Smooth the data and add it to the data lists
    redundancy_strategy1 = moving_average(redundancy_strategy, window_size=10)
    disk_scrubbing_freq1 = moving_average(disk_scrubbing_freq, window_size=10)
    redundancy_strategy_data.append(redundancy_strategy1)
    disk_scrubbing_freq_data.append(disk_scrubbing_freq1)

# Extract the last 200 rounds of redundancy_strategy
last_200_rounds = redundancy_strategy[-200:]

# Calculate the standard deviation of the last 200 rounds
stdev_last_200_rounds = statistics.stdev(last_200_rounds)

# Print the result
print(f"Standard deviation of last 200 rounds: {stdev_last_200_rounds}")

if key == 3:
    # Plot the redundancy strategy variation
    plt.figure()
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14
    for i in range(len(files)):
        plt.plot(redundancy_strategy_data[i], color=colors[i], label=label[i])
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Redundancy Scheme')
    plt.title('Adjusting Redundancy Scheme Only')
    plt.savefig('Scrub\Redundancy.pdf', format='pdf')

    # Plot the disk scrubbing frequency variation
    plt.figure()
    for i in range(len(files)):
        plt.plot(disk_scrubbing_freq_data[i], color=colors[i], label=label[i])
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Scrubbing Rate')
    plt.title('Adjusting Scrubbing Rate Only')
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14
    plt.savefig('Scrub\S_Scrubbing.pdf', format='pdf')
    plt.savefig('S_Scrubbing.pdf', format='pdf')

    # Show the plots
    plt.show()
