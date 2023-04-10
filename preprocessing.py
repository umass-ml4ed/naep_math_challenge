import csv
from collections import defaultdict
import json
import pandas as pd
import collections
score_list = ['rater_1', 'pta_rtr1','ptb_rtr1','ptc_rtr1','score','assigned_score','score_to_predict']


def preprocessing_each_question_var(path='/home/mengxue/Downloads/Math_scoring_challenge/train.csv',
                           data_dict='/home/mengxue/Downloads/Math_scoring_challenge/', sep='<SEP>', analysis=True):
    flag_mapping = {1: 'incorrect', 2: 'correct', 0: 'empty'}
    question_list = json.load(open('question.json', 'r'))
    df = pd.read_csv(path)
    df.fillna(0, inplace=True)
    type1 = ['VH266015', "VH302907","VH507804", "VH139380"]
    type2 = ["VH139380"]

    for key in type1:
        #type 1
        qdf = df[df['accession'] == key]
        columns = question_list[key]['context_var']
        if key == 'VH266015':
            # 1. for question VH266015, div
            cols_to_include = question_list[key]['var'] + score_list
            for part_name, column_list in columns.items():
                qdf['context_' + part_name] = qdf[column_list].iloc[:, 1:].values.tolist()
                qdf['context_' + part_name] = qdf['context_' + part_name].apply(_list_to_string)
                qdf['context_' + 'all'] = qdf.apply(lambda row: "{} {}:{}".format(part_name, flag_mapping[row[column_list[0]]], row['context_' + part_name]), axis=1)
                # analysis
                test = qdf[qdf[column_list[0]] == 2]
                values = collections.Counter(list(test['context_' + part_name]))
                # unique_values = test['context_'+ part_name].unique()
        if key == "VH302907": #geometry
            if analysis:
                col = columns['B']
                #part B
                test = qdf[qdf[col[0]] == 2]
                values = collections.Counter(list(test[col[1]]))

                col=columns['C']
                test = qdf[qdf[col[0]] == 2]
                values = collections.Counter(list(test[col[1]]))
            col = columns['ALL']
            #qdf['context_all'] = qdf.apply(lambda row: "B is {}: [{}], C is {}: [{}]".format(flag_mapping[row[col[0]]], row[col[1]], flag_mapping[row[col[2]]], row[col[3]]), axis=1)
            qdf['context_all'] = qdf[col].values.tolist()
            qdf['context_all'] = qdf['context_all'].apply(lambda row: _list_to_string(row,ver='geo'))
        if key == "VH507804":
            for part_name, column_list in columns.items():
                qdf['context_' + part_name] = qdf[column_list].iloc[:, 1:].values.tolist()
                qdf['context_' + part_name] = qdf['context_' + part_name].apply(lambda row: _list_to_string(row, ver='4card'))
                qdf['context_' + 'all'] = qdf.apply(lambda row: "{} {}:{}".format(part_name, flag_mapping[row[column_list[0]]], row['context_' + part_name]), axis=1)
                # analysis
                if analysis:
                    test = qdf[qdf[column_list[0]] == 2]
                    values = collections.Counter(list(test['context_' + part_name]))


        #Type 2

    for key in type2:
        if key == "VH139380":
            qdf = df[df['accession'] == key]
            columns = question_list[key]['context_var']







def read_and_transfor_into_csv(train_path='/home/mengxue/Downloads/Math_scoring_challenge/all_items_train.txt',
                           data_dict='/home/mengxue/Downloads/Math_scoring_challenge/', sep='<SEP>'):
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
        df.to_csv( data_dict + 'train_' + key + '.csv', index=False)

    #also save a uniform file named train.csv
    split_lines = [line.split('<SEP>') for line in file_lines]
    df = pd.DataFrame(split_lines[1:], columns=split_lines[0])
    df.to_csv(data_dict + 'train.csv', index=False)


        #save_csv(data_dict, 'train_'+ key + '.csv', question_dict[key], sep=sep)

    # with open(test_path,'r') as test_file:
    #     file_content = test_file.read()
    #     file_content = file_content.replace('\t',',')
    #     file_lines = file_content.split('\n')
    # save_csv(data_dict, 'test_' + '.csv', file_lines)

def construct_useful_fields(path='/home/mengxue/Downloads/Math_scoring_challenge/all_items_train.txt',sep='<SEP>'):
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


#HELP
def _list_to_string(lst, ver='div'):
    if ver == 'div':
        number = {1: 3, 2: 4, 3: 6, 4: 7,0:'nan'}
        n1, p1, n2, p2, n3, p3, n4, p4 = lst
        try:
            orders = {p1: number[n1], p2: number[n2], p3: number[n3], p4: number[n4]}
            n1, n2, n3, n4 = [orders[i + 1] for i in range(4)]
            result = 'nan1/nan2 * nan3/nan4'
            for p, n in orders.items():
                result = result.replace('nan' + str(int(p)), str(n))
            #for p, n in orders.items():
            #result = '{}/{} * {}/{}'.format(n1, n2, n3, n4)
        except:
            pass
    if ver == 'geo':
        flag_mapping = {1: 'incorrect', 2: 'correct', 0: 'empty'}
        result = 'A is {}: {}; B is {}: {}'.format(flag_mapping[int(lst[0])], lst[1], flag_mapping[int(lst[2])], lst[3])
    if ver == '4card':
        number = {1: 17, 2: 27, 3: 54, 4:62, 0:'nan'}
        n1, p1, n2, p2, n3, p3 = lst
        try:
            orders = {}
            result = 'nan1 * nan2 - nan3'
            orders.update({p1: number[n1], p2: number[n2], p3: number[n3]})
            for p,n in orders.items():
                result = result.replace('nan' + str(int(p)), str(n))

            #n1, n2, n3 = [orders[i + 1] for i in range(3)]
            #result = '{} * {} - {}'.format(n1, n2, n3)
        except:
            pass


    return result
    # return '[' + ','.join([str(int(elem)) for elem in lst]) + ']'

def main():
    pass

if __name__ == '__main__':
    read_and_transfor_into_csv()
    #construct_useful_fields()
    #preprocessing_each_question_var()