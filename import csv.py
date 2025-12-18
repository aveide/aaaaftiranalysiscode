import csv
print("hello1")
input_file = 'Book.csv'
output_file = 'cleaned_data.csv'
print("hello")
# The index of the column you want to fix (0 is the first column, 1 is the second, etc.)
column_index_to_fix = 0

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        # Get the specific cell
        cell_value = row[column_index_to_fix]

        # Check if the cell contains the arrow "->"
        if '->' in str(cell_value):
            # Split the string and keep only the part after the arrow
            clean_value = cell_value.split('->')[1].strip()
            if float(clean_value) <= 10:
                print(float(clean_value))
                row[column_index_to_fix] = clean_value


        #print(row)

        # Write the row (modified or original) to the new file
        writer.writerow(row)

print("Done!")