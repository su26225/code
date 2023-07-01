# Import pandas and csv libraries
import pandas as pd
import csv

# Read the file and convert it into a DataFrame object
# df = pd.read_csv('1monthdiskset.csv')
df = pd.read_csv('1monthdiskset.csv', names=['serial_number','model','capacity_bytes'])

# Process the first column by extracting the first two characters for rows starting with "S3", "Z3", or "W3"
# Add the extracted values as a new column to the DataFrame object
df['prefix'] = df['serial_number'].apply(lambda x: x[:2] if x.startswith(('S3', 'Z3', 'W3')) else None)

# Group the new column and the second column together and count the number of occurrences in each group
result = df.groupby(['prefix', 'model']).size()

# Write the counting results into a new CSV file named "result.csv"
with open('startnumber.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['prefix', 'string', 'count'])
    for (prefix, string), count in result.items():
        writer.writerow([prefix, string, count])
