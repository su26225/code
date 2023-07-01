import csv

# Read the serial_number values from file 'a' and store them in a dictionary
serial_numbers = {}
with open("HGST-HMS5C4040BLE640-label1.csv") as a_file:
    reader = csv.reader(a_file)
    next(reader) # Skip the header
    for row_number, row in enumerate(reader):
        serial_numbers[row[1]] = row_number # Assuming serial_number is in the first column

# Read file 'b' and modify the values in the label column
modified_rows = []
with open("HGST-HMS5C4040BLE640-100.csv") as b_file:
    reader = csv.reader(b_file)
    header = next(reader) # Read the header
    modified_rows.append(header) # Add the header to the modified rows list
    for row in reader:
        if row[1] in serial_numbers: # Assuming serial_number is in the first column
            row[-1] = "1" # Assuming label is in the last column
        modified_rows.append(row) # Add the modified row to the list

# Write to a new CSV file
with open("Train-HGST-HMS5C4040BLE640.csv", "w") as new_b_file:
    writer = csv.writer(new_b_file)
    writer.writerows(modified_rows) # Write all the modified rows
