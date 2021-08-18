# Author:       B177014
# Title:        VDJ Machine Learning Running Tool
# Date Created: 03/08/2021 (dd/mm/yyyy)
#   - Implemented vdj_run
#       a) Added reading command line parameters to obtain model names and fasta file names.
#       b) Added calling of load model to load in models from files.
#       c) Added reading sequences from fasta file.
#   - Implemented load_model
#       a) Added pickle function to load in the appropriate model.
#   - Implemented read_fasta to read in fasta files and return the sequences and appropriate k-mers.
#   - Implemented run_model to run the saved models on the input fasta sequences.
# Date Modified: 05/08/2021
#   - Replaced using pandas dataframe for validation with a list of values instead (Faster now and no more error for
#     indexing large values of pandas data frame).
#   - Modified run_models to run only one model at a time. This was done to allow preprocessing between each model to
#     run on data that would go into the next one.
#   - Implemented remove_v_seq to remove the V-gene from the sequence to improve accuracy for J and D gene prediction.
#   - Implemented remove_j_seq to remove the J-gene from the sequence to improve accuracy for D-gene prediction.
# Assumptions:
#   - The user has a model for V, D, and J genes stored in separate files.
# Notes:
#   - For more development notes check the file build_vdj_model.py

import sys                                  # Used to read command line arguments.
import pickle                               # Used to read in machine learning model.
from vdj_functions import seq_to_k_mers     # Used to convert sequences to k-mers.
import pandas as pd                         # Used to store data and allow for it to run with other functions.


# Loads a specified model.
def load_model(file_name):
    # Opens and stores the model from a specified binary file.
    return pickle.load(open(file_name, 'rb'))


# Reads in fasta files and returns their sequences and generated k-mers.
def read_fasta(fasta_files):
    # Makes lists to hold sequences and k-mers.
    seq_list = []
    k_mer_list = []

    # Reads in fasta files.
    for file_name in fasta_files:
        # Opens current file.
        file_in = open(file_name, "r")

        # Notifies user of reading fasta file.
        print("Reading sequences from the fasta file: " + file_name)

        # Reads current file line by line.
        for line in file_in:
            # Ensures line is not empty.
            if len(line) > 0:
                # Checks if line is a sequence or not.
                if line[0] != ">":
                    # Adds lowercase line to list.
                    seq_list.append(line.lower())

                    # Adds k-mers to list.
                    k_mer_list.append(seq_to_k_mers(seq_list[-1]))

        # Notifies user of file being read.
        print("Sequences read from the fasta file: " + file_name)

    # Returns the list of sequences and k-mers.
    return seq_list, k_mer_list


# Runs the models that were retrieved.
def run_models(model, vectorizer, name, k_mer_list):
    # Stores k-mers in a pandas dataframe so CountVectorizer will work with the data.
    k_mer_df = pd.DataFrame(k_mer_list, columns=['k_mers'])

    # Prompts user of progress.
    print("Running " + name + " model ")

    # Vectorize data with current vectorizer.
    test_data = vectorizer.transform(k_mer_df['k_mers'])

    # Runs vectorized data through current model.
    predict_list = model.predict(test_data)

    # Prompt user that current model is done.
    print(name + " model complete")

    # Returns the predictions.
    return predict_list


# Reads in and stores the lengths genes.
def read_length_files():
    # Creates dictionary to hold lengths.
    length_dict = {"v": {}, "d": {}, "j": {}}

    # Makes list of the three genes to have files opened for.
    genes = ["v", "d", "j"]

    # Loops through each gene.
    for gene in genes:
        # Opens the appropriate file.
        file_in = open(str(gene) + "_gene_lengths.data", "r")

        # Reads the opened file line by line.
        for line in file_in:
            # Split line on the comma separating terms.
            val_list = line.split(",")

            # Stores the length with the name as the key.
            length_dict[gene][val_list[0]] = val_list[1].split("\n")[0]

    # Returns the populated 2D dictionary.
    return length_dict


# Removes the V Genes from a sequence based on predictions.
def remove_v_seq(predict_list, seq_list, length_dict):
    # Makes a list to hold new sequences and one to hold new k-mers.
    new_seq_list = []
    new_k_mers = []

    # Loops through all predictions.
    for i in range(len(predict_list)):
        # Stores the sequence after the length of the v-gene found.
        new_seq_list.append(seq_list[i][int(length_dict[predict_list[i]]):])

    # Finds the new k-mers from the found sequences.
    for seq in new_seq_list:
        new_k_mers.append(seq_to_k_mers(seq))

    # Returns the new sequences and k-mers.
    return new_seq_list, new_k_mers


# Removes the J Genes from a sequence based on predictions.
def remove_j_seq(predict_list, seq_list, length_dict):
    # Makes a list to hold new sequences and one to hold new k-mers.
    new_seq_list = []
    new_k_mers = []

    # Loops through all predictions.
    for i in range(len(predict_list)):
        # Stores the sequence before the current sequence length minus the length of the found j-gene found.
        new_seq_list.append(seq_list[i][0:len(seq_list[i]) - int(length_dict[predict_list[i]])])

    # Finds the new k-mers from the found sequences.
    for seq in new_seq_list:
        new_k_mers.append(seq_to_k_mers(seq))

    # Returns the new sequences and k-mers.
    return new_seq_list, new_k_mers


# Runs the models from file.
def vdj_run():
    # Retrieves the command line arguments.
    model_file_names = sys.argv[1:4]
    fasta_files = sys.argv[4:]

    # Makes a list of models.
    model_vec_list = []

    # Loops through all file names and opens them.
    for name in model_file_names:
        model_vec_list.append(load_model(name))

    # Reads in fasta files.
    seq_list, k_mer_list = read_fasta(fasta_files)

    # Reads in gene lengths files.
    length_dict = read_length_files()

    # Runs the V-Gene model first
    predict_v_list = run_models(model_vec_list[0][0], model_vec_list[0][1], "V-Gene", k_mer_list)

    # Calculates new sequences and k-mers without predicted V Gene.
    seq_no_v, k_mer_no_v = remove_v_seq(predict_v_list, seq_list, length_dict["v"])

    # Runs the J-Gene model second.
    predict_j_list = run_models(model_vec_list[2][0], model_vec_list[2][1], "J-Gene", k_mer_no_v)

    # Calculates new sequences without V and J genes.
    seq_no_v_j, k_mer_no_v_j = remove_j_seq(predict_j_list, seq_no_v, length_dict["j"])

    # Runs the D-Gene model last.
    predict_d_list = run_models(model_vec_list[1][0], model_vec_list[1][1], "D-Gene", k_mer_no_v_j)

    # Adds V, D, and J predictions to a larger list.
    predict_list = [predict_v_list, predict_d_list, predict_j_list]

    # Makes list to hold csv names.
    csv_names = []

    # Loops through names of fasta to get their csv file names.
    for name in fasta_files:
        # Gets the name of the file before the "." and adds a ".csv" to that result.
        csv_names.append(name.split(".")[0] + ".csv")

    # Makes list of pandas dataframes.
    csv_data = []

    # Loops through csv files and combines them.
    for name in csv_names:
        csv_data.append(pd.read_csv(name))

    # Makes a dictionary to hold the gene labels.
    gene_true = {"v": [], "d": [], "j": []}

    # Loops through each dataframe loaded.
    for data in csv_data:
        # Adds labels to lists.
        for i in range(len(data['V_allele'])):
            gene_true["v"].append(data['V_allele'][i])
            gene_true["d"].append(data['D_allele'][i])
            gene_true["j"].append(data['J_allele'][i])

    # Makes counters.
    count_v = count_d = count_j = count_all = count_v_fam = count_d_fam = count_j_fam = count_all_fam = 0

    # Separator and end values for csv.
    sep = ","
    end = "\n"

    # Opens output file.
    out_file = open("vdj_classifications.csv", "w")
    out_file.write("")
    out_file.close()
    out_file = open("vdj_classifications.csv", "a")

    # Prints the columns for the data.
    out_file.write("Sequence" + sep + "V_Gene" + sep + "V_Actual" + sep + "D_Gene" + sep + "D_Actual" + sep + "J_gene" +
                   sep + "J_Actual" + sep + "V_Correct" + sep + "D_Correct" + sep + "J_correct" + sep + "All_Correct" +
                   sep + "V-Family" + sep + "D-Family" + sep + "J-Family" + end)

    # Print Results by looping through all predictions.
    for i in range(len(predict_list[0])):
        # Checks if V, D, and J genes are correct.
        v_correct = int(predict_list[0][i].lower() == str(gene_true['v'][i]).lower())
        d_correct = int(predict_list[1][i].lower() == str(gene_true['d'][i]).lower())
        j_correct = int(predict_list[2][i].lower() == str(gene_true['j'][i]).lower())

        v_fam_correct = int(predict_list[0][i].lower().split("-")[0] == str(gene_true['v'][i]).lower().split("-")[0])
        d_fam_correct = int(predict_list[1][i].lower().split("-")[0] == str(gene_true['d'][i]).lower().split("-")[0])
        j_fam_correct = int(predict_list[2][i].lower().split("*")[0] == str(gene_true['j'][i]).lower().split("*")[0])

        all_correct = int(1 == v_correct == d_correct == j_correct)
        all_fam_correct = int(1 == v_fam_correct == d_fam_correct == j_fam_correct)

        # Prints the resulting predicted genes.
        out_file.write(str(i) + sep + str(predict_list[0][i]) + sep + str(gene_true['v'][i]) + sep +
                       str(predict_list[1][i]) + sep + str(gene_true['d'][i]) + sep + str(predict_list[2][i]) + sep +
                       str(gene_true['j'][i]) + sep + str(v_correct) + sep + str(d_correct) + sep + str(j_correct) +
                       sep + str(all_correct) + sep + str(v_fam_correct) + sep + str(d_fam_correct) + sep +
                       str(j_fam_correct) + sep + str(all_fam_correct) + end)

        # Increments counts
        count_v = count_v + v_correct
        count_d = count_d + d_correct
        count_j = count_j + j_correct

        count_v_fam = count_v_fam + v_fam_correct
        count_d_fam = count_d_fam + d_fam_correct
        count_j_fam = count_j_fam + j_fam_correct

        count_all = count_all + all_correct
        count_all_fam = count_all_fam + all_fam_correct

    # Writes results to a new file.
    out_file.close()
    out_file = open("vdj_summary.txt", "w")
    out_file.write("")
    out_file.close()
    out_file = open("vdj_summary.txt", "a")

    # Prints count data.
    print("V-Gene:\t" + str(count_v))
    out_file.write("V-Gene:\t" + str(count_v) + "\n")
    print("D-Gene:\t" + str(count_d))
    out_file.write("D-Gene:\t" + str(count_d) + "\n")
    print("J-Gene:\t" + str(count_j))
    out_file.write("J-Gene:\t" + str(count_j) + "\n")

    print("All:\t" + str(count_all))
    out_file.write("All:\t" + str(count_all) + "\n")

    print("V-Family:\t" + str(count_v_fam))
    out_file.write("V-Family:\t" + str(count_v_fam) + "\n")
    print("D-Family:\t" + str(count_d_fam))
    out_file.write("D-Family:\t" + str(count_d_fam) + "\n")
    print("J-Family:\t" + str(count_j_fam))
    out_file.write("J-Family:\t" + str(count_j_fam) + "\n")

    print("All-Family:\t" + str(count_all_fam))
    out_file.write("All-Family:\t" + str(count_all_fam) + "\n")

    print("Total:\t" + str(len(predict_list[0])))
    out_file.write("Total:\t" + str(len(predict_list[0])) + "\n")


# Runs the script.
vdj_run()
