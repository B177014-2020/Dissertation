# Author:       B177014
# Title:        VDJ Machine Learning Running Tool
# Date Created: 03/08/2021 (dd/mm/yyyy)
#   - Moved existing functions from build_vdj_model.py to a new file.
#   - Modified build_model, nat_lang_proc, and save_model to save the vectorizer with the model.
# Date Modified: 05/08/2021
#   - Modified print_times to print the save_ref time.
#   - Implemented save_ref
#       a) Tracks runtime of itself.
#       b) Calls write_gene for each gene to do the actual writing.
#   - Implemented write_gene to store gene names and lengths to a file for each gene.

# Libraries used for reading and storing data.
import sys              # For use of command line arguments.
import os.path          # For checking if files exist.
import timeit           # For checking script runtime.
import pandas as pd     # For pandas dataframe.
import pickle           # For storing the created models.

# Libraries used for making machine learning model.
from sklearn.feature_extraction.text import CountVectorizer     # Used to generate Bag of Words.
from sklearn.model_selection import train_test_split            # Used to split data for model training and testing.
from sklearn.naive_bayes import MultinomialNB                   # Used to make classifier that is fit to data.
from sklearn.metrics import accuracy_score, f1_score,\
                            precision_score, recall_score       # Functions used to analyse ML model's predictions.

# Libraries used for graphing.
import matplotlib.pyplot as plt                                 # Used to make heatmap.
import seaborn                                                  # Also used for making the heatmap.

# Global variables for script
max_num_files = 20              # Arbitrarily selected value to ensure too many files are not run at once.
k_mer_length = 6                # Do not make smaller than 2 or larger than the longest junction length.
n_gram_min = n_gram_max = 1     # Controls the n-gram used by the bag of words model.
alpha_val = 0.1                 # Alpha value used by the Naive Bayes Multinomial model for building the classifier.


# Opens input files and store data into a dictionary.
def open_files(num_files, file_names):
    # Empty dictionary implementation to make returns consistent.
    data_dict = {}

    # Checks if num_files exceeds maximum number allowed.
    if num_files > max_num_files:
        # Tells user they are trying to use too many files.
        print("The maximum number of files allowed is " + str(max_num_files) + " you attempted to use " + num_files +
              " files")

        # Returns from function early.
        return False, data_dict

    # Loops through each file.
    for i in range(num_files):
        # Prints the name of the file from the file list that is being opened.
        print("Opening file:", file_names[i])

        # Checks if the file entered exists.
        if not os.path.isfile(file_names[i]):
            print("Error: \"" + file_names[i] + "\" was not found.")
            print("Try using full path if this error persists.")
            return False, data_dict

        # Opens file for reading.
        file_in = open(file_names[i], "r")

        # Reads labels from first lines.
        labels = file_in.readline().lower().split(",")

        # Makes dictionary for the first loop only.
        if i == 0:
            # Makes dictionary with labels as keys
            for key in labels:
                # Makes an empty list for each dictionary key.
                data_dict[key] = []

        # Loops through all remaining lines.
        for line in file_in:
            # Stores the data from the current line in the form of a list.
            data_list = line.lower().split(",")

            # Loops through data list with an index.
            for j in range(len(data_list)):
                # Appends new data to list at the appropriate key (Assumes all data follows the initial order and no
                # missing data).
                data_dict[labels[j]].append(data_list[j])

        # Notifies user of finished file.
        print("Data read from:", file_names[i])

        # Closes input file.
        file_in.close()

    # Returns True with completed data.
    return True, data_dict


# Times the open_files function.
def time_file_run(num_files, file_names):
    # Starts timer.
    start_time = timeit.default_timer()

    # Runs the open_file function and stores the result.
    file_pair = open_files(num_files, file_names)

    # Ends timer.
    stop_time = timeit.default_timer()

    # Calculates runtime.
    file_runtime = stop_time - start_time

    # Ensures open file ran successfully.
    if not file_pair[0]:
        # returns to main early due to open file not successfully running.
        return False, file_pair[1], file_runtime

    # Returns values.
    return True, file_pair[1], file_runtime


# Prints a message and time with minutes if applicable otherwise prints with seconds.
def print_mins(string, time):
    # Checks if value is greater than 60 (1 minute).
    if time > 60:
        # Calculates minutes and remaining seconds.
        secs = time % 60
        mins = int(time / 60)

        # Prints message with minutes and seconds
        print(str(string) + str(mins) + " minutes & " + str(secs) + " seconds")
    else:
        # Prints message with seconds.
        print(str(string) + str(time) + " seconds")


# Prints the runtimes of the script.
def print_times(file_runtime, pandas_time, k_mer_time, ml_time, ref_time):
    # Calls print_mins with an appropriate message for each time.
    print_mins("Reading data runtime: ", file_runtime)
    print_mins("Pandas data fame building time: ", pandas_time)
    print_mins("K-mer creation time: ", k_mer_time)
    print_mins("Training and testing of model time: ", ml_time)
    print_mins("Saving gene name and position time: ", ref_time)


# Menu for running script (also handles command line arguments).
def prompt_user():
    # Declares empty list for file names.
    file_names = []

    # Checks if command line arguments were used.
    if len(sys.argv) > 1:
        # Stores number of files and file names from command line.
        num_files = len(sys.argv) - 1
        file_names = sys.argv[1:]
    else:
        # Prompts user for number of files.
        num_files = input("Enter the number of input files: ")
        num_files = int(num_files)

        # Loops for the number of selected files.
        for i in range(int(num_files)):
            # Prompts user for file name and stores it in a list.
            file_name = input("Please enter the name/path of file " + str(i + 1) + ": ")
            file_names.append(file_name)

    # Runs open files with set variables.
    return time_file_run(num_files, file_names)


# Splits the input string into k-mers of the specified length. The return value is a single space delimited string of
# k-mers.
def seq_to_k_mers(seq, length=k_mer_length):
    # Make string to hold all k-mers.
    k_mer_str = ''

    # Loops through sequence and is subtracted by k-mer length to ensure read ends when k-mers reach end of sequence
    # and plus 1 to account for
    # starting at 0.
    for i in range(len(seq) - length + 1):
        # Adds the k-mer at the current position with the length specified.
        k_mer_str = k_mer_str + ' ' + seq[i:i + length]

    # Returns space delimited string of k-mers.
    return k_mer_str


# Makes a list of k-mers from a source column in a pandas dataframe.
def list_k_mers(pd_data, source, length=k_mer_length):
    # Notify user of k-mer creation for specified source.
    print("Creating k-mers of length " + str(length) + " for " + source + "...")

    # Makes new list to hold k-mers and eventually become new column of data.
    k_mer_list = []

    # Loops through all sequences and makes k-mers.
    for seq in pd_data[source]:
        # Add list of new k-mers to k-mer list.
        k_mer_list.append(seq_to_k_mers(seq))

    # Notify user of k-mers being made.
    print("K-mers created for " + source + ".")

    # Returns the populated list of k-mers.
    return k_mer_list


# Finds the v allele within the whole sequence.
def find_gene_seq(pd_data, gene):
    # Makes list that will become the new column.
    seq_list = []

    # Makes the start and end category vars for finding gene of interest.
    pos_start_col = str(gene) + "_sequence_start"
    pos_end_col = str(gene) + "_sequence_end"

    # Loop through all mutated sequences.
    for i in range(len(pd_data['mutated_sequence'])):
        # Stores the current sequence.
        seq = pd_data['mutated_sequence'][i]

        # Retrieves the current start and end positions. Specifies type to avoid errors.
        pos_start = int(pd_data[pos_start_col][i])
        pos_end = int(pd_data[pos_end_col][i])

        # Finds the gene of interest (+1 is needed for pythons exclusive behavior of the last value in array parsing).
        gene = seq[pos_start:(pos_end + 1)]

        # Adds sequence to list.
        seq_list.append(gene)

    # Returns the v sequences list.
    return seq_list


# Finds the family name of a gene.
def find_fam(pd_data, col):
    # Makes empty list to hold family names.
    fam_list = []

    # Loops through each gene.
    for gene in pd_data[col]:
        # Parses the family from the gene name and stores it.
        fam_list.append(gene[0:gene.find("-")])

    # Returns the populated list of family names.
    return fam_list


# Converts dictionary of data into pandas dataframe and times this process.
def time_pandas(dict_in):
    # Prompt user of pandas data frame creation.
    print("Creating pandas data frame...")

    # Starts timer.
    start_time = timeit.default_timer()

    # Runs the open_file function and stores the result.
    pd_data = pd.DataFrame(data=dict_in)

    # Ends timer.
    stop_time = timeit.default_timer()

    # Prompt user of data frame creation.
    print("Pandas data frame successfully created.")

    # Create time variable to store dataframe creation time.
    pandas_time = stop_time - start_time

    # Starts timer.
    start_time = timeit.default_timer()

    # Make column of v, d, & j gene sequences.
    pd_data['v_sequence'] = find_gene_seq(pd_data, gene="v")
    pd_data['d_sequence'] = find_gene_seq(pd_data, gene="d")
    pd_data['j_sequence'] = find_gene_seq(pd_data, gene="j")

    # Make column of v, d, & j k-mers.
    pd_data['v_k_mers'] = list_k_mers(pd_data, 'v_sequence')
    pd_data['d_k_mers'] = list_k_mers(pd_data, 'd_sequence')
    pd_data['j_k_mers'] = list_k_mers(pd_data, 'j_sequence')

    # Make column for v, d, & j allele family.
    pd_data['v_family'] = find_fam(pd_data, 'v_allele')
    pd_data['d_family'] = find_fam(pd_data, 'd_allele')
    pd_data['j_family'] = find_fam(pd_data, 'j_allele')

    # Ends timer.
    stop_time = timeit.default_timer()

    # Returns dataframe and runtime.
    return pd_data, pandas_time, stop_time - start_time


# Consolidates metrics calls into one function.
def ml_metrics(test, predict):
    # Calls and stores various sklearn functions.
    conf_matrix = pd.crosstab(pd.Series(test, name='Actual'), pd.Series(predict, name='Predicted'))
    acc = accuracy_score(test, predict)
    prec = precision_score(test, predict, average='weighted')
    recall = recall_score(test, predict, average='weighted')
    f1 = f1_score(test, predict, average='weighted')

    # Creates lists for calculating family accuracy.
    test_fam = []
    predict_fam = []

    # Populates lists of family names from the test data and then prediction data.
    for test_val in test:
        test_fam.append(test_val.split("*")[0].split("-")[0])
    for pred_val in predict:
        predict_fam.append(pred_val.split("*")[0].split("-")[0])

    # Makes counter to hold number of correct families predicted and a total number of predictions variable.
    count = 0
    total = len(test_fam)

    # Loops through each family prediction.
    for i in range(len(test_fam)):
        # Checks if the family prediction is correct.
        if test_fam[i] == predict_fam[i]:
            # Increments count if the family prediction is correct.
            count = count + 1

    # Calculates percent accuracy.
    fam_acc = count / total

    # Returns the resulting values.
    return conf_matrix, acc, prec, recall, f1, fam_acc


# Prints the results from ml_metrics.
def print_ml_metrics(test, predict):
    # Calculates values.
    matrix, acc, prec, recall, f1, fam_acc = ml_metrics(test, predict)

    # Prints the matrix and values.
    print("Confusion Matrix:")
    print(matrix)
    print("Allele Accuracy: " + str(acc))
    print("Family Accuracy: " + str(fam_acc))
    print("Precision: " + str(prec))
    print("Recall: " + str(recall))
    print("F1: " + str(f1))

    # Returns the matrix.
    return matrix


# Prints a heatmap with the given parameters.
def print_heatmap(matrix, title):
    # Modifies matplotlib.pyplot data for heatmap.
    plt.title(title)
    plt.subplots_adjust(bottom=0.2, left=0.21)
    seaborn.heatmap(matrix)

    # Display heatmap
    plt.show()


# Writes the confusion matrix to a file.
def write_matrix(file_name, matrix):
    # Ensures file is empty.
    file_out = open(file_name, "w")
    file_out.write("")
    file_out.close()

    # Opens file to append data.
    file_out = open(file_name, "a")
    matrix.to_csv(file_out)
    file_out.close()


# Builds a model for the specified source and label
def build_model(source, labels, matrix_name, file_name):
    # Make a vectorizer for the data.
    vectorizer = CountVectorizer(ngram_range=(n_gram_min, n_gram_max)).fit(source)

    # Make a bag of words object from the source data.
    words = vectorizer.transform(source)

    # Split data into training, and testing.
    source_train, source_test, label_train, label_test = train_test_split(words, labels, test_size=0.25)

    # Makes and fits classifiers for each gene label.
    classifier = MultinomialNB(alpha=alpha_val).fit(source_train, label_train)

    # Calculate prediction values.
    predictions = classifier.predict(source_test)

    # Prints results.
    confusion_matrix = print_ml_metrics(label_test, predictions)

    # Prints heatmaps.
    print_heatmap(confusion_matrix, matrix_name)

    # Writes matrix to a file.
    write_matrix(file_name, confusion_matrix)

    # Returns the built and tested model with the vectorizer.
    return classifier, vectorizer


# Saves a built model to a file.
def save_model(model, vectorizer, file_name):
    # Notifies user of saving starting.
    print("Saving model to: " + str(file_name))

    # Saves the model in the specified binary file using pickle.
    pickle.dump((model, vectorizer), open(file_name, 'wb'))

    # Notifies user of save completing.
    print("Model successfully saved to: ", str(file_name))


# Creates a bag of words k-mer counting model, trains it, and tests it.
def nat_lang_proc(pd_data):
    # Notifies user on progress.
    print("Training and Testing Bag of Words k-mer counting model...")

    # Stores start time.
    start_time = timeit.default_timer()

    # Builds all models.
    v_gene_model, v_vectorizer = build_model(pd_data['v_k_mers'], pd_data['v_allele'], "V Gene Matrix", "v_gene_matrix.csv")
    d_gene_model, d_vectorizer = build_model(pd_data['d_k_mers'], pd_data['d_allele'], "D Gene Matrix", "d_gene_matrix.csv")
    j_gene_model, j_vectorizer = build_model(pd_data['j_k_mers'], pd_data['j_allele'], "J Gene Matrix", "j_gene_matrix.csv")

    # Stores stop time.
    stop_time = timeit.default_timer()

    # Notifies user of progress.
    print("Training and Testing finished.")

    # Notifies user of models being saved.
    print("Saving models to files...")
    save_model(v_gene_model, v_vectorizer, "v_gene_model.dat")
    save_model(d_gene_model, d_vectorizer, "d_gene_model.dat")
    save_model(j_gene_model, j_vectorizer, "j_gene_model.dat")

    # Returns the runtime.
    return stop_time - start_time


def write_gene(pd_data, gene):
    # Notifies user of writing data.
    print("Writing " + str(gene) + " gene data...")

    # Makes dictionary to save positions with gene name.
    gene_dict = {}

    # Assigns variables for current gene.
    gene_name = str(gene) + "_allele"
    start_pos = str(gene) + "_sequence_start"
    end_pos = str(gene) + "_sequence_end"

    # Loops through all gene names.
    for i in range(len(pd_data[gene_name])):
        # Stores the gene name as the key and gene length.
        gene_dict[pd_data[gene_name][i]] = int(pd_data[end_pos][i]) - int(pd_data[start_pos][i])

    # Opens file to write data to.
    file_out = open(str(gene) + "_gene_lengths.data", "w")

    # Loops through made dictionary to populate file.
    for key in gene_dict:
        file_out.write(str(key) + "," + str(gene_dict[key]) + "\n")

    # Notifies user that writing is complete.
    print("Successfully wrote " + str(gene) + " gene data")


# Saves references to genes and their size.
def save_ref(pd_data):
    # Stores start time.
    start_time = timeit.default_timer()

    # Writes all three genes to a file.
    write_gene(pd_data, "v")
    write_gene(pd_data, "d")
    write_gene(pd_data, "j")

    # Stores the stop time.
    stop_time = timeit.default_timer()

    # Returns the runtime.
    return stop_time - start_time
