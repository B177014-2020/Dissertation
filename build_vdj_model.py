# Author:       B177014
# Title:        Machine Learning VDJ Classification Tool
# Date Created: 10/06/2021 (dd/mm/yyyy)
#   - Implemented command line argument checking.
#   - Implemented checking if file exists.
#   - Implemented opening file and storing data in a dictionary.
# Date Modified 16/06/2021
#   - Made it so all data stored was stored in lower case.
#   - Moved functionality of timing the open_file function to its own function.
#   - Implemented new function, print_times, to print runtimes.
#   - Miscellaneous formatting and commenting changes.
# Date Modified 17/06/2021
#   - Implemented a prompt_user function.
#       a) Allows for user input if no command line parameters are used.
#   - Added global variable for number of files that can be used.
#   - Modified open_file function
#       a) Renamed function to open_files.
#       b) Implemented use of multiple file.
#       c) Implemented checking if number of files exceeds the global maximum.
# Date Modified 29/06/2021
#   - Implemented max_list_len function.
#       a) Finds the max length of a list.
#   - Added global variable for padding strings.
#   - Implemented pad_seq function.
#       a) Pads function with the set global character to make all values max length.
# Date Modified 30/06/2021
#   - Implemented time_pandas function.
#       a) Creates a data frame for training ML model.
#       b) Times the creation of dataframe for final output.
#   - Removed padding misstep (Commented out).
#   - Updated print runtimes to print the dataframe building time.
# Date Modified 01/07/2021
#   - Added k-mer length global variable.
#   - Implemented seq_to_k_mers function.
#       a) Splits a string sequence into a list of strings of specified k-mer length.
#       b) Global variable controls k-mer size.
#   - Updated time_pandas to print progress to the screen.
# Date Modified 15/07/2021
#   - Modified seq_to_k_mers so that it has another parameter for k-mer length that defaults to the global variable.
#   - Modified print_times function to print minutes and seconds for k-mer time if applicable.
#   - Implemented nat_lang_proc function
#       a) Transforms k-mers into a single string of space delimited k-mers per sequence.
#       b) Stores the classification data in a separate variable.
#   - Replaced using "mutated_sequence" for k-mer creation with the use of the "junction" data.
#   - Improved nat_lang_proc
#       a) Implemented Bag of Words for k-mers.
#       b) Implemented splitting data into train and test sets.
# Date Modified 17/07/2021
#   - Implemented new global variables to easily modify nat_lang_proc.
#       a) n_gram_min and n_gram_max was added to change the n-gram value used by CountVectorizer (final values should
#          be determined experimentally).
#       b) alpha_val used by the Naive Bayes Classifier (optimizing this value may be done through a grid search)
# Date Modified 18/07/2021
#   - Modified nat_lang_proc:
#       a) Implemented use of MultiLabelBinarizer in an attempt to avoid a "legacy type" error with the function
#          classifier.fit.
#       b) Multiple list and array manipulations done to attempt to resolve errors with input data dimensions.
# Date Modified 19/07/2021
#   - Removed use of sklearn library.
#       a) Memory use was high and there were problems getting classifier.fit to work.
#   - Implemented numpy arrays for list of k-mers.
# Date Modified 20/07/2021
#   - Implemented bag of words generation for each junction separately.
#       a) May change later.
#       b) Done with intent to be less memory intensive and more computationally intensive.
#   - Attempted to implement bag of words for all k-mers generated.
# Date Modified 21/07/2021
#   - Modified time_pandas to add k-mers to their own new column instead of replacing junction.
#       a) Went with for-loop implementation over the previous lambda expression for easier readability.
#   - Modified seq_to_k_mers so that it returns a space delimited string of k-mers instead of a list.
#       a) This was done to follow the formatting in the sklearn documentation for the function CountVectorizer.
#   - Modified nat_lang_proc:
#       a) Implemented bag of words for all k-mers generated.
#       b) Removed bag of words for each individual row in dataframe.
#       c) Split gene labels into 3 separate classifiers.
# Date Modified 22/07/2021
#   - Modified nat_lang_proc:
#       a) Fixed how classifiers are called.
#       b) Added predicting genes from bag of words.
#       c) Added calling of new metrics functions.
#   - Implemented ml_metrics function.
#       a) Organises all metric functions from sklearn into one.
#   - Implemented print_ml_metrics function.
#       a) Prints the data from ml_metrics function.
# Date Modified 23/07/2021
#   - Attempted to use junctions as labels for nat_lang_proc function.
#       a) Not a feasible solution. Too many unique junction sites making this approach too naive.
#   - Modified nat_lang_proc:
#       a) Added variables to track the time of the training and predicting.
#       b) Added prompts to notify user of script's progress.
#   - Modified print_times:
#       a) now prints the time it takes to train and test the model.
# Date Modified 26/07/2021
#   - Cleaned up various comments.
#   - Implemented print_mins function.
#       a) Used to remove repetitive code in print_times by calculating time in minutes if needed.
#   - Modified print_times to use the newly implemented print_mins function.
# Date Modified 27/07/2021
#   - Modified print_ml_metrics to return the confusion matrix.
#   - Implemented print_heatmap function.
#       a) Displays a heatmap with the given matrix and title.
#   - Modified nat_lang_proc to call print_heatmap and make heatmaps of confusion matrices.
# Date Modified 28/07/2021
#   - Implemented list_k_mers function to streamline making k-mers in multiple instances.
#   - Modified time_pandas to use the new list_k_mers function instead of doing the operations itself.
# Date Modified 29/07/2021
#   - Implemented find_v_seq to find the section of the mutated sequence that is the v allele.
#   - Modified time_pandas
#       a) Now creates a column of v allele sequences using find_v_seq.
#       b) Uses list_k_mers to make a column of v allele k-mers.
#   - Modified nat_lang_proc to use v allele k-mers instead of the previous junction ones for classifying v alleles.
# Date Modified 30/07/2021
#   - Implemented find_fam function to find the family name for an allele.
#   - Modified time_pandas:
#       a) Now uses find_fam to make a column with the family name.
#       b) No longer makes k-mers for junction.
#   - Modified nat_lang_proc to run for the family as well as gene for v.
# Date Modified 31/07/2021
#   - Implemented build_model function to streamline the process that must be done within nat_lang_proc for all genes
#     and families of interest.
# Date Modified 01/08/2021
#   - Implemented find_d_seq function to find the d gene for k-mer creation.
#       a) Came across problem with being unable to find exact position with current input data.
#       b) Currently has beginning of J gene with it.
# Date Modified 02/08/2021
#   - Generalised find_v_seq to find_gene_seq so that it can be used for all 3 genes.
#       a) Removed find_d_seq.
#       b) Made possible by new input data format.
#       c) Ensures better accuracy than the original "11 - v_3_trim" of find_v_seq.
#   - Modified time_pandas to generate data for D and J genes similar to the existing V data generated.
#   - Modified nat_lang_proc to build models for D and J genes.
# Date Modified 03/08/2021
#   - Added the use of pickle library to store models.
#   - Modified build_model to return the built model.
#   - Moved all functions to separate file.
# Date Modified 05/08/2021
#   - Added the save_ref function.
# Assumptions:
#   - Files used for input are formatted correctly.
#   - Files all use the same labels in the same order as the first file.
#   - Global variable k_mer_length is an integer value larger than 1.

import vdj_functions


# Main function that calls others.
def build_vdj_model():
    # Runs menu function.
    file_results = vdj_functions.prompt_user()

    # Checks file results for successful run.
    if not file_results[0]:
        # Ends main early due to an error with the data retrieval.
        return

    # Stores data in a pandas dataframe.
    pd_data, pandas_time, k_mer_time = vdj_functions.time_pandas(file_results[1])

    # Runs data through natural language processing.
    ml_time = vdj_functions.nat_lang_proc(pd_data)

    # Saves reference files for V, D, and J genes.
    ref_time = vdj_functions.save_ref(pd_data)

    # Prints the runtimes.
    vdj_functions.print_times(file_results[2], pandas_time, k_mer_time, ml_time, ref_time)


# Main function call to run the script.
build_vdj_model()
