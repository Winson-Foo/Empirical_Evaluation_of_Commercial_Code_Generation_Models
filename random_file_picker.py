import os
import random

sample_size = 187
# Set the path to your directory
directory_path = r'C:\Users\tan weijin\Desktop\FYP_sonarcube\Empirical_Evaluation_of_Commercial_Code_Generation_Models\calculate_sample_size'

# Get a list of all files in the directory
file_list = os.listdir(directory_path)

# Filter out directories and keep only files
file_list = [file for file in file_list if os.path.isfile(os.path.join(directory_path, file))]

# Choose 10 random files from the list
random_files = random.sample(file_list, sample_size)

# Write the randomly selected file names to a text file
output_file_path = 'random_files.txt'

with open(output_file_path, 'w') as output_file:
    for file in random_files:
        output_file.write(file + '\n')

print("Selected file names written to " + output_file_path)