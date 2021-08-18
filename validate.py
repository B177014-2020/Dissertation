import pandas as pd
from math import isnan

# Stored the key file name.
key_file_name = "rand_seq_3.csv"
# seq_file_name = "rand_seq_3.fasta"
# ig_blast_file_name = "igblast_rand_3.data"
# mixcr_file_name = "mixcr_results_3.txt"
imgt_file_name = "summary_3.txt"


def main():
    # Opens the key file.
    key_data = pd.read_csv(key_file_name)

#    # Opens the IgBLAST file.
#    ig_blast_data = open(ig_blast_file_name, "r")

    # # Opens the MiXCR file.
    # mixcr_data = pd.read_csv(mixcr_file_name, sep="\t")

    # # Sets boolean for genes to false.
    # get_gene = False

    # # Reads in the FASTA file.
    # fasta_file = open(seq_file_name, "r")

    # Sequence list
    seq_list = []

    # # Loops through FASTA file and adds each sequence to a list.
    # for line in fasta_file:
    #     # Ignores comment lines.
    #     if line[0] != ">":
    #         seq_list.append(line.split("\n")[0])

    # Creates dictionary for genes
#    predict_dict = {"v": [], "d": [], "j": [], "seq": []}
    predict_dict = {"v": [], "d": [], "j": []}

    # # Loops through IgBLAST results line by line.
    # for line in ig_blast_data:
    #     # Gets the genes
    #     if get_gene:
    #         # Sets the genes.
    #         predict_dict["v"].append(line.split("\t")[0].split(",")[0].lower())
    #         predict_dict["d"].append(line.split("\t")[1].split(",")[0].lower())
    #         predict_dict["j"].append(line.split("\t")[2].split(",")[0].lower())
    #
    #         # Sets the boolean back to False.
    #         get_gene = False
    #
    #     # Checks for line preceding gene prediction.
    #     if "V-(D)-J rearrangement summary for query sequence" in line:
    #         get_gene = True

    # # Loops through all pandas data.
    # for i in range(len(mixcr_data["allVHitsWithScore"])):
    #     # Checks for string.
    #     if isinstance(mixcr_data["allVHitsWithScore"][i], str):
    #         # Populates dictionary with current values.
    #         predict_dict["v"].append(str(mixcr_data["allVHitsWithScore"][i]).split("(")[0].lower())
    #     else:
    #         predict_dict["v"].append("NaN")
    #
    #     # Checks for string.
    #     if isinstance(mixcr_data["allDHitsWithScore"][i], str):
    #         # Populates dictionary with current values.
    #         predict_dict["d"].append(str(mixcr_data["allDHitsWithScore"][i]).split("(")[0].lower())
    #     else:
    #         predict_dict["d"].append("NaN")
    #
    #     # Checks for NaN.
    #     if isinstance(mixcr_data["allVHitsWithScore"][i], str):
    #         predict_dict["j"].append(str(mixcr_data["allJHitsWithScore"][i]).split("(")[0].lower())
    #     else:
    #         predict_dict["j"].append("NaN")
    #
    #     # Adds target sequence to dictionary.
    #     predict_dict["seq"].append(mixcr_data["targetSequences"][i].lower())

    # Opens IMGT data.
    imgt_data = pd.read_csv(imgt_file_name, sep="\t")

    # Loops through IMGT data.
    for i in range(len(imgt_data["V-GENE and allele"])):
        # Populates dictionary with alleles.
        if isinstance(imgt_data["V-GENE and allele"][i], str):
            predict_dict["v"].append(str(imgt_data["V-GENE and allele"][i]).split(" ")[1].lower())
        else:
            predict_dict["v"].append("NaN")
        if isinstance(imgt_data["D-GENE and allele"][i], str):
            predict_dict["d"].append(str(imgt_data["D-GENE and allele"][i]).split(" ")[1].lower())
        else:
            predict_dict["d"].append("NaN")
        if isinstance(imgt_data["J-GENE and allele"][i], str):
            predict_dict["j"].append(str(imgt_data["J-GENE and allele"][i]).split(" ")[1].lower())
        else:
            predict_dict["j"].append("NaN")

    # Makes counters.
    count_v = count_d = count_j = count_all = count_v_fam = count_d_fam = count_j_fam = count_all_fam = 0

    # Print Results by looping through all predictions.
    for i in range(len(predict_dict["v"])):
        # Checks if V, D, and J genes are correct.
        v_correct = int(key_data["V_allele"][i] == str(predict_dict['v'][i]))
        d_correct = int(key_data["D_allele"][i] == str(predict_dict['d'][i]))
        j_correct = int(key_data["J_allele"][i] == str(predict_dict['j'][i]))

        v_fam_correct = int(key_data["V_allele"][i].split("-")[0] == str(predict_dict['v'][i]).split("-")[0])
        d_fam_correct = int(key_data["D_allele"][i].split("-")[0] == str(predict_dict['d'][i]).split("-")[0])
        j_fam_correct = int(key_data["J_allele"][i].split("*")[0] == str(predict_dict['j'][i]).split("*")[0])

        # if predict_dict['seq'][i] in seq_list:
        #     v_correct = int(key_data["V_allele"][seq_list.index(predict_dict['seq'][i])] == str(predict_dict['v'][i]))
        #     d_correct = int(key_data["D_allele"][seq_list.index(predict_dict['seq'][i])] == str(predict_dict['d'][i]))
        #     j_correct = int(key_data["J_allele"][seq_list.index(predict_dict['seq'][i])] == str(predict_dict['j'][i]))
        #
        #     v_fam_correct = int(key_data["V_allele"][seq_list.index(predict_dict['seq'][i])].split("-")[0] == str(predict_dict['v'][i]).split("-")[0])
        #     d_fam_correct = int(key_data["D_allele"][seq_list.index(predict_dict['seq'][i])].split("-")[0] == str(predict_dict['d'][i]).split("-")[0])
        #     j_fam_correct = int(key_data["J_allele"][seq_list.index(predict_dict['seq'][i])].split("*")[0] == str(predict_dict['j'][i]).split("*")[0])
        # else:
        #     v_correct = d_correct = j_correct = v_fam_correct = d_fam_correct = j_fam_correct = 0

        all_correct = int(1 == v_correct == d_correct == j_correct)
        all_fam_correct = int(1 == v_fam_correct == d_fam_correct == j_fam_correct)

        # Increments counts
        count_v = count_v + v_correct
        count_d = count_d + d_correct
        count_j = count_j + j_correct

        count_v_fam = count_v_fam + v_fam_correct
        count_d_fam = count_d_fam + d_fam_correct
        count_j_fam = count_j_fam + j_fam_correct

        count_all = count_all + all_correct
        count_all_fam = count_all_fam + all_fam_correct

    # Prints count data.
    print("V-Gene:\t" + str(count_v))
    print("D-Gene:\t" + str(count_d))
    print("J-Gene:\t" + str(count_j))

    print("All:\t" + str(count_all))

    print("V-Family:\t" + str(count_v_fam))
    print("D-Family:\t" + str(count_d_fam))
    print("J-Family:\t" + str(count_j_fam))

    print("All-Family:\t" + str(count_all_fam))

    print("Total:\t" + str(len(predict_dict["v"])))


main()
