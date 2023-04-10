import csv
from collections import defaultdict
import json
import pandas as pd
from utils import add_additional_info
# def add_additional_info():


def read_and_transform_into_csv(org_data_path='./data/all_items_train.txt',
                                csv_path='./data'):
    # NEVER CHANGE THIS.
    org_data_path='./data/all_items_train.txt'
    csv_path='./data'

    qid_list = ["VH134067", "VH266015", "VH302907", "VH507804", "VH139380", "VH266510", "VH269384", "VH271613", "VH304954", "VH525628"]
    qid_dict = {}
    with open(org_data_path, 'r') as org_txt_file:
        txt_data = org_txt_file.read()
        ## Using <SEP> because some data contains comma. 
        txt_data = txt_data.replace('\t', "<SEP>")
        txt_data = txt_data.replace('\ufeff', '')
        txt_data = txt_data.replace('"','')
        ## Split into rows. 
        txt_in_lines = txt_data.split('\n')       

    header = txt_in_lines[0].split("<SEP>")

    for row in txt_in_lines[1:]:
        row = row.split("<SEP>")
        assert len(row) == len(header), "Length of the row and the header should be the same."
        qid = row[1]
        if qid == "VH266510":
            if row[4] == "2019":
                qid = "VH266510_2019"
            else:
                assert row[4] == "2017", "It is either 2019 or 2017."
                qid = "VH266510_2017"
        if qid_dict.get(qid) is None:
            ## We divide qid VH266510 into two seperate case because of different vars.
            valid_idx_list = [i for i, x in enumerate(row) if x != "NA"]
            valid_cols = [header[i] for i in valid_idx_list]
            qid_dict[qid] = {}
            ## Store variables
            qid_dict[qid]["vars"] = valid_cols # NEED THIS?
            qid_dict[qid]["var_idxs"] = valid_idx_list
            ## Store student responses & scores (data)
            ## NOTE: Do we need all the data in the row? For example, do we need student id?
            qid_dict[qid]["data"] = []
            ## Add questions and other infos.
            add_additional_info(qid, qid_dict)
        qid_dict[qid]["data"].append([row[i] for i in qid_dict[qid]["var_idxs"]])

    print(qid_dict.keys())

    for key, value in qid_dict.items():
        file_name = f"{key}_train_data.csv"
        f = open(f"./data/{file_name}", 'w')
        writer = csv.writer(f)
        num_of_responses = len(value["data"])

        writer.writerow(value["vars"])
        for row in value["data"]:
            writer.writerow(row)

    ## Remove data info for saving meta info only.
    for key in qid_dict.keys():
        del qid_dict[key]["data"]    
    with open("./data/meta_info.json", 'w') as f:
        json.dump(qid_dict, f)    


if __name__ == '__main__':

    read_and_transform_into_csv()
    #construct_useful_fields()
