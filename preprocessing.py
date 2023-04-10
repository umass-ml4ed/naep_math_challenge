import csv
from collections import defaultdict
import json
import pandas as pd
score_list = ['rater_1','pta_rtr1','ptb_rtr1','ptc_rtr1','score','assigned_score','score_to_predict']

def read_and_transfor_into_csv(train_path='./data/all_items_train.txt',
                               data_dict='./data', sep='<SEP>'):
    with open(train_path,'r') as train_file:
        file_content = train_file.read()
        file_content = file_content.replace('\t',sep)
        file_content = file_content.replace('\ufeff', '')
        file_content = file_content.replace('"','')
        file_lines = file_content.split('\n')
    question_list = json.load(open('question.json','r'))

    # create a CSV writer object to write to the output file
    heads = file_lines[0]
    #question_dict = defaultdict(lambda : [heads])
    question_dict = {q: [heads] for q in question_list.keys()}
    correct_formate_dict = {}
    number_of_field = len(heads.split(sep))
    for line in file_lines[1:]:
        try:
            question = line.split(sep)[1].replace('"','')
            assert len(line.split('<SEP>')) == number_of_field, print(
                '{} and {} doesnt match'.format(len(line.split('<SEP>')), number_of_field))
            assert question in question_dict, print(question)
            question_dict[question].append(line)
        except:
            print('here')


    for key in question_dict.keys():
        cols_to_include = question_list[key]['var'] + score_list
        split_lines = [line.split('<SEP>') for line in question_dict[key]]
        # Convert the list of lists into a Pandas DataFrame
        df = pd.DataFrame(split_lines[1:], columns=split_lines[0])
        # Only keep the columns that are specified in cols_to_include
        df = df[cols_to_include]
        # Save the resulting DataFrame to a CSV file
        # df.to_csv( data_dict + 'train_' + key + '.csv', index=False)
        df.to_csv( f"./data/{data_dict+'train_'+key+'.csv'}", index=False)

        #save_csv(data_dict, 'train_'+ key + '.csv', question_dict[key], sep=sep)

    # with open(test_path,'r') as test_file:
    #     file_content = test_file.read()
    #     file_content = file_content.replace('\t',',')
    #     file_lines = file_content.split('\n')
    # save_csv(data_dict, 'test_' + '.csv', file_lines)

def construct_useful_fields(path='./data/all_items_train.txt',sep='<SEP>'):
    with open(path,'r') as file:
        file_content = file.read()
        file_content = file_content.replace('\t',sep)
        file_content = file_content.replace('"','')
        file_content = file_content.replace('﻿','')
        file_lines = file_content.split('\n')
    question_list = json.load(open('question.json','r'))
    heads = file_lines[0].split(sep)
    data_point = file_lines[1:]
    for q in data_point[:-1]:
        q = q.split(sep)
        pair_list = list(zip(heads,q))
        filtered_pairs = [pair[0] for pair in pair_list if pair[1] != 'NA']
        #index = filtered_pairs.index('parsed_xml_v1')
        #unique_list = filtered_pairs[index:]
        unique_list = filtered_pairs
        question_list[q[0]].update({'var': unique_list})
    with open('question.json','w') as f:
        json.dump(question_list,f,indent=4)

def save_csv(data_dict, name, data, sep='<SEP>'):
    with open(data_dict + name, 'w', newline='') as output_file:
        csv_writer = csv.writer(output_file)

        # write each line as a row in the CSV file
        for line in data:
            # split the line by commas
            row = line.split(sep)
            # write the row to the CSV file
            csv_writer.writerow(row)

def main():
    pass

if __name__ == '__main__':
    read_and_transfor_into_csv()
    #construct_useful_fields()