import pandas as pd

# Read CSV files, assuming the second column is serial_number
# Store the file names in a list
filenames = ["Train-WDC-WD30EFRX.csv", "Train-Hitachi-HDS723030ALA640.csv", "BalckDaUsare.csv", "HGST-HMS5C4040BLE640-100-label1.csv"]
# Store the corresponding separators for each file in a dictionary
separators = {"Train-WDC-WD30EFRX.csv": ",", "Train-Hitachi-HDS723030ALA640.csv": ",", "BalckDaUsare.csv": ";", "HGST-HMS5C4040BLE640-100-label1.csv": ","}
# Store the names for each file in a list
names = ["WDC", "Hit", "Seg", "hgst"]

# Initialize an empty list to store the dataframes
dfs = []

# Iterate over the filenames and names using a for loop
for filename, name in zip(filenames, names):
    # Get the separator for the current file
    sep = separators[filename]
    # Read the CSV file and append the dataframe to the dfs list
    df = pd.read_csv(filename, sep=sep)
    dfs.append(df)

# Initialize empty dictionaries to store the counts, label counts, and ratios
counts = {}
label_counts = {}
ratios = {}

# Iterate over the dfs and names using a for loop
for df, name in zip(dfs, names):
    # Calculate the count of unique serial_numbers and store it in the counts dictionary with the name as the key
    counts[name] = df["serial_number"].nunique()
    # Filter the dataframe for Label = 1, calculate the count of unique serial_numbers, and store it in the label_counts dictionary with the name as the key
    label_counts[name] = df[df["Label"] == 1]["serial_number"].nunique()
    # Calculate the ratio of label counts to total counts and store it in the ratios dictionary with the name as the key
    ratios[name] = label_counts[name] / counts[name]

# Print the counts dictionary
print("counts:", counts)
# Print the label_counts dictionary
print("label_counts:", label_counts)

# Iterate over the ratios dictionary using the items() function
for name, ratio in ratios.items():
    # Print the result using the format() function to format the output
    print("{}: {:.2f}%".format(name, ratio * 100))


# df1 = pd.read_csv("Train-WDC-WD30EFRX.csv")
# df2 = pd.read_csv("Train-Hitachi-HDS723030ALA640.csv")
# df3 = pd.read_csv("BalckDaUsare.csv",sep=';')
# df4 = pd.read_csv("HGST-HMS5C4040BLE640-100-label1.csv")
#

# count1 = df1["serial_number"].nunique()
# count2 = df2["serial_number"].nunique()
# count3 = df3["serial_number"].nunique()
# count4 = df3["serial_number"].nunique()
#
#

# label_count1 = df1[df1["Label"] == 1]["serial_number"].nunique()
# label_count2 = df2[df2["Label"] == 1]["serial_number"].nunique()
# label_count3 = df3[df3["Label"] == 1]["serial_number"].nunique()
# label_count4 = df4[df4["Label"] == 1]["serial_number"].nunique()
#

# ratio1 = label_count1 / count1
# ratio2 = label_count2 / count2
# ratio3 = label_count3 / count3
# ratio4 = label_count4 / count4
#
#

# print('WDC: ' , count1)
# print('Hit: ' , count2)
# print('Seg: ' , count3)
# print('hgst: ' , count4)