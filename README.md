# code

SelectRow.py:
Classify the data for a span of 15 years based on different values in the 'model' column, and save the classified data into separate CSV files with filenames corresponding to the 'model' values.

cal2number.py:
Process and tally the serial number data of various disk models, and store the statistical results in a new CSV file.

number_count.py:
Read multiple CSV files and store them in distinct data frames. Calculate the count of unique sequence numbers in each data frame. Determine the count of unique serial numbers labeled as faults. Compute the ratio of the count of serial numbers labeled as 1 to the total count.

changelabel1.py:
In the original dataset, only the last recorded moment of a failed disk is considered as a fault, and the label of the failed disk for that particular year is changed to 1.

null100.py:
Data processing task involving the removal of empty lines and replacement of blank space values with a fixed value.

Train_GBDT_WDC.py et al.:
Health prediction model encompassing steps such as disk labeling, feature selection, training, and validation.

torch-Reinforcementv2.py:
Training model for reinforcement learning. Health level is calculated based on brand classification. Disk failures are simulated according to the health level, and the optimal strategy during failures is learned.
