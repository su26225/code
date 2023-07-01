# import pandas as pd
#
# def SelsctRow():
#     df = pd.read_csv('15disk.csv', usecols=['date', 'serial_number', 'model', 'capacity_bytes', 'smart_187_normalized',
#                             'smart_189_normalized',
#                             'smart_194_normalized',
#                             'smart_197_normalized',
#                             'smart_197_raw',
#                             'smart_1_normalized',
#                             'smart_3_normalized',
#                             'smart_5_normalized',
#                             'smart_5_raw',
#                             'smart_7_normalized',
#     'failure',
#                             'smart_9_normalized'])
#

#     df = df.rename(columns={'smart_187_normalized': 'ReportedUncorrectableErrors',
#                             'smart_189_normalized': 'HighFlyWrites',
#                             'smart_194_normalized': 'TemperatureCelsius',
#                             'smart_197_normalized': 'CurrentPendingSectorCount',
#                             'smart_197_raw': 'RawCurrentPendingSectorCount',
#                             'smart_1_normalized': 'RawReadErrorRate',
#                             'smart_3_normalized': 'SpinUpTime',
#                             'smart_5_normalized': 'ReallocatedSectorsCount',
#                             'smart_5_raw': 'RawReallocatedSectorsCount',
#                             'smart_7_normalized': 'SeekErrorRate',
#     'failure': 'Label',
#                             'smart_9_normalized': 'PowerOnHours'                       })

#     df = df.loc[:, ['date','serial_number','model','capacity_bytes',
#                     'ReportedUncorrectableErrors', 'HighFlyWrites',
#                     'TemperatureCelsius','CurrentPendingSectorCount', 'RawCurrentPendingSectorCount',
#                     'RawReadErrorRate','SpinUpTime', 'ReallocatedSectorsCount',
#                     'RawReallocatedSectorsCount','SeekErrorRate', 'PowerOnHours','Label']]
#     df.to_csv('selectrow15disk.csv')
#



import csv



def read_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row

files = {}
for row in read_csv('15disk.csv'):
    key = row['model']
    if key not in files:
        file = open('15allrows-model\\' + key + '.csv', 'w')
        writer = csv.DictWriter(file, fieldnames=row.keys())
        writer.writeheader()
        files[key] = (file, writer)
    files[key][1].writerow(row)

for file, writer in files.values():
    file.close()



    # with open('selectrow15disk.csv', 'r') as f:  # Open the input file
    #     reader = csv.DictReader(f)  # Create a DictReader object
    #     rows = list(reader)  # Store all rows of data in a list
    #
    # files = {}  # A dictionary to store file objects and writers
    # for row in rows:  # Iterate over each row of data
    #     key = row['model']  # Get the value of the "model" column
    #     if key not in files:  # Check if this value is new
    #         file = open('15model/' + key + '.csv', 'w')  # Create a new file object with this value as the filename
    #         writer = csv.DictWriter(file,
    #                                 fieldnames=reader.fieldnames)  # Create a new writer object with the same fieldnames as the reader
    #         files[key] = (file, writer)  # Add the file object and writer to the dictionary with this value as the key
    #     files[key][1].writerow(
    #         row)  # Write this row of data to the corresponding file using the writer from the dictionary
    #
    # for file, writer in files.values():  # Iterate over each file object and writer in the dictionary
    #     file.close()  # Close each file object

# SelsctRow()
# SelsctModel()