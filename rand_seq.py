# Author:       B177014
# Title:        Random Sequence Compiler
# Date Created: 23/06/2021 (dd/mm/yyyy)
#   - Implemented selecting an equal amount of sequences from all files and making a new random fasta file with the
#     specified global variables.
# Date Modified: 07/08/2021
#   - Implemented the creation of a key file alongside the random fasta sequences

import os.path          # For checking if files exist.
import sys              # For reading command line arguments.
import random           # For selecting random sequences.
import datetime         # For generating pseudorandom seed.
import pandas as pd     # For reading in csv file.

# Global variables.
num_seq = 50000
num_fasta = 100000
output_file = "rand_seq_3.fasta"

# Seed random number generator.
random.seed(datetime.datetime.now())


def main():
    # Obtains the number of arguments minus 1 to get number of command line parameters.
    num_files = len(sys.argv) - 1

    # Calculates the number of sequences from each file.
    remainder = num_seq % num_files

    # Obtains the number of sequences per file (without the remainder).
    num_list_seq = num_seq // num_files

    # Makes parallel list for random sequences.
    random_seq_list = [[-1] * num_list_seq] * num_files

    # Appends remainder values to random sequences list if it is non-zero.
    for i in range(remainder):
        # Runs for only non-zero values.
        if i != 0:
            random_seq_list[i - 1].append(-1)

    # Populates random sequence list.
    for i in range(num_files):
        # Prompts progress.
        print("Choosing random sequences from " + sys.argv[i + 1])

        # Make modified for loop from while loop so it can rerun duplicate values.
        j = 0
        while j < len(random_seq_list[i]):
            # Obtains a random int in range.
            rand_val = random.randint(0, num_fasta)

            # Checks if random value is in the list already.
            if rand_val not in random_seq_list[i]:
                # Stores the random integer that is not in the list.
                random_seq_list[i][j] = rand_val

                # Increments count to ensure that the right amount of random values are found.
                j = j + 1

    # Makes list of sequences and keys for final files.
    seq_list_final = []
    key_list_final = [[], [], []]

    # Loop through each argument (file).
    for i in range(num_files):
        # Prints that file is being opened.
        print("Opening file:", sys.argv[i + 1])

        # Checks if current file exists.
        if not os.path.isfile(sys.argv[i + 1]):
            print("Error: \"" + sys.argv[i + 1] + "\" was not found")
            return

        # Opens file for reading.
        file_in = open(sys.argv[i + 1])

        # Opens file for key.
        pd_data = pd.read_csv(sys.argv[i + 1].split(".")[0] + ".csv")

        # Makes lists for key values.
        v_label = []
        d_label = []
        j_label = []

        # Loops through all V labels.
        for line in pd_data['V_allele']:
            v_label.append(line.lower())

        # Loops through all D labels.
        for line in pd_data['D_allele']:
            d_label.append(line.lower())

        # Loops through all J labels.
        for line in pd_data['J_allele']:
            j_label.append(line.lower())

        # Make list of sequences from file.
        file_seq_list = []

        # Reads fasta file line by line.
        for line in file_in:
            if ">" not in line:
                # Adds line (sequence) to list.
                file_seq_list.append(line.lower())

        # Closes the current file.
        file_in.close()

        # Loops through list of random values to add sequences to final sequence list.
        for j in range(len(random_seq_list[i])):
            # Adds sequence to list.
            seq_list_final.append(file_seq_list[random_seq_list[i][j]])

            # Stores the labels to a final list for all files.
            key_list_final[0].append(v_label[random_seq_list[i][j]])
            key_list_final[1].append(d_label[random_seq_list[i][j]])
            key_list_final[2].append(j_label[random_seq_list[i][j]])

    # Makes a new file to write sequences and keys to (And empties it if it exists).
    file_out = open(output_file, "w")
    file_out.write("")
    file_out.close()
    file_out = open(output_file.split(".")[0] + ".csv", "w")
    file_out.write("")
    file_out.close()

    # Append data to the new file.
    print("Writing data to " + output_file)
    file_out = open(output_file, "a")
    for i in range(len(seq_list_final)):
        # Writes the header for the sequence.
        file_out.write(">")
        file_out.write(str(i))
        file_out.write("\n")

        # Writes the sequence to the file.
        file_out.write(seq_list_final[i])
    # Closes file.
    file_out.close()

    # Writes the key file.
    print("Writing key data to " + output_file.split(".")[0] + ".csv")
    file_out = open(output_file.split(".")[0] + ".csv", "a")

    # Writes the column names.
    file_out.write("sequence_number,V_allele,D_allele,J_allele\n")
    
    # Writes the key data.
    for i in range(len(key_list_final[0])):
        # Writes sequence number.
        file_out.write(str(i))
        file_out.write(",")
        
        # Writes gene names in VDJ order.
        file_out.write(key_list_final[0][i])
        file_out.write(",")
        file_out.write(key_list_final[1][i])
        file_out.write(",")
        file_out.write(key_list_final[2][i])
        file_out.write("\n")

    # Closes file.
    file_out.close()


# Calls main function.
main()
