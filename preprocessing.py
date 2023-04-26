import csv
from collections import defaultdict
import json
import pandas as pd
import collections
score_list = ['rater_1', 'pta_rtr1', 'ptb_rtr1', 'ptc_rtr1', 'score', 'score_to_predict']

def preprocessing_each_question_var(path='data/train.csv',
                           data_dict='data/', sep='<SEP>', analysis=True):
    """
    Review each question and merge some variables
    :param path:
    :param data_dict:
    :param sep:
    :param analysis: True for running extra code to analyze the data
    :return:
    """

    flag_mapping = {1: 'incorrect', 2: 'correct', 0: 'empty'}
    question_list = json.load(open('question.json', 'r'))
    df = pd.read_csv(path)
    df.fillna(0, inplace=True)

    df_list = []

    type1 = []
    type1 = ["VH134067", 'VH266015', "VH302907","VH507804"]
    type2 = ["VH139380","VH304954","VH525628","VH266510_2017", "VH266510_2019"]
    type3 = ["VH269384","VH271613"]
    type_all = type1 + type2 + type3

    for key in type1:
        #type 1
        qdf = df[df['accession'] == key]
        columns = question_list[key]['context_var']
        score = question_list[key]['score']
        if key == "VH134067":
            pass
            #qdf['context_all'] = ''
        if key == 'VH266015':
            #for question VH266015, div
            cols_to_include = question_list[key]['var'] + score_list
            for part_name, column_list in columns.items():
                qdf['context_' + part_name] = qdf[column_list].iloc[:, 1:].values.tolist()
                qdf['context_' + part_name] = qdf['context_' + part_name].apply(_list_to_string)
                qdf['context_' + 'all'] = qdf.apply(lambda row: "{} {}:{}".format(part_name, flag_mapping[row[column_list[0]]], row['context_' + part_name]), axis=1)
                if analysis:
                    test = qdf[qdf[column_list[0]] == 2]
                    values = collections.Counter(list(test['context_' + part_name]))
        if key == "VH302907": #geometry
            if analysis:
                col = columns['B']
                #part B
                test = qdf[qdf[col[0]] == 2]
                values = collections.Counter(list(test[col[1]]))
                col=columns['C']
                test = qdf[qdf[col[0]] == 2]
                values = collections.Counter(list(test[col[1]]))
            col = columns['all']
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

        qdf['label'] = qdf[score]
        df_list.append(qdf)

        #Type 2

    for key in type2:
        qdf = df[df['accession'] == key]
        columns = question_list[key]['context_var']
        correct_scores = question_list[key]['correct_score']
        score = question_list[key]['score']
        if key == "VH139380":
            col = columns['A']
            if analysis:
                correct_A = correct_scores['A']
                test = qdf[qdf[score].isin(correct_A)]
                values = collections.Counter(list(test[col[0]]))
            qdf['context_all'] = qdf[col[0]]
        if key == "VH304954": #sub
            col = columns['B']
            if analysis:
                correct_B = correct_scores['B']
                test = qdf[qdf[score].isin(correct_B)]
                values = collections.Counter(list(test[col[0]]))
            qdf['context_all'] = qdf[col[0]]
        if key == "VH525628":
            col = columns['A']
            for part_name, column_list in columns.items():
                qdf['context_' + part_name] = qdf[column_list].values.tolist()
                qdf['context_' + part_name] = qdf['context_' + part_name].apply(lambda row: _list_to_string(row, ver='least'))
            qdf['context_all'] = qdf['context_A']
            if analysis:
                correct_A = correct_scores['A']
                test = qdf[qdf[score].isin(correct_A)]
                values = collections.Counter(list(test['context_all']))
        if key == "VH266510_2017":
            qdf['context_A'] = qdf[columns['A']]
            qdf['context_all'] = qdf['context_A']
            if analysis:
                correct_A = correct_scores['A']
                test = qdf[qdf[score].isin(correct_A)]
                values = collections.Counter(list(test['context_all']))
        if key == "VH266510_2019":
            qdf['context_A'] = qdf[columns['A']].values.tolist()
            qdf['context_A'] = qdf['context_A'].apply(
                lambda row: _list_to_string(row, ver='slop_2019'))
            qdf['context_all'] = qdf['context_A']
            if analysis:
                correct_A = correct_scores['A']
                test = qdf[qdf[score].isin(correct_A)]
                values = collections.Counter(list(test['context_all']))
        qdf['label'] = qdf[score]
        df_list.append(qdf)


    #type 3 means a combine of type 2 and type 1
    for key in type3:
        qdf = df[df['accession'] == key]
        columns = question_list[key]['context_var']
        correct_scores = question_list[key]['correct_score']
        score = question_list[key]['score']
        if key == 'VH269384':
            for part_name, column_list in columns.items():
                qdf['context_' + part_name] = qdf[column_list].values.tolist()
                qdf['context_' + part_name] = qdf['context_'+part_name].apply(lambda row: _list_to_string(row, ver='8card_'+part_name))
            if analysis:
                    col = columns['A']
                    correct_A = correct_scores['A']
                    test = qdf[qdf[score].isin(correct_A)]
                    values = collections.Counter(list(test['context_all']))
        if key == 'VH271613':
            for part_name, column_list in columns.items():
                qdf['context_' + part_name] = qdf[column_list].values.tolist()
            qdf['context_all'] = qdf['context_all'].apply(lambda row: _list_to_string(row, ver='age'))
            if analysis:
                    col = columns['A']
                    correct_A = correct_scores['A']
                    test = qdf[qdf[score].isin(correct_A)]
                    values = collections.Counter(list(test['context_all']))
        qdf['label'] = qdf[score]
        df_list.append(qdf)

    merged_df = pd.concat(df_list, axis=0, sort=False)
    merged_df['label'] = merged_df['label'].replace({'1.0': '1', '2.0': '2', 1.0: '1', 2.0: '2', 3.0: '3', 1:'1', 2:'2',3:'3'})
    merged_df['label'] = merged_df['label'].astype(str)
    merged_df.to_csv(data_dict + 'train.csv', index=False)

    df = merged_df
    question_list = construct_useful_fields()
    extra = ['context_A','context_B','context_all','label']
    for key in type_all:
        qdf = df[df['accession'] == key]
        if "VH266510" in key:
            cols_to_include = question_list["VH266510"]['var'] + extra
        else:
            cols_to_include = question_list[key]['var'] + extra
        qdf = qdf[cols_to_include]
        # Save the resulting DataFrame to a CSV file
        qdf.to_csv(data_dict + 'train_' + key + '.csv', index=False)



def read_and_transfor_into_csv(train_path='data/all_items_train.txt',
                           data_dict='data/', sep='<SEP>'):
    with open(train_path,'r') as train_file:
        file_content = train_file.read()
        file_content = file_content.replace('\t',sep)
        file_content = file_content.replace('\ufeff', '')
        file_content = file_content.replace('"','')
        file_lines = file_content.split('\n')
    question_list = json.load(open('question.json','r'))
    question_list = construct_useful_fields()

    # create a CSV writer object to write to the output file
    heads = file_lines[0]
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
        df.to_csv(data_dict + 'train_' + key + '.csv', index=False)

    #also save a uniform file named train.csv
    split_lines = [line.split('<SEP>') for line in file_lines]
    df = pd.DataFrame(split_lines[1:], columns=split_lines[0])

    # Define a function to modify the question_id based on the year value
    def modify_question_id(row):
        if row['accession'] == "VH266510":
            return f"VH266510_{str(int(row['year']))}"
        else:
            return row['accession']
    # Apply the modify_question_id function to the question_id column
    df['accession'] = df.apply(modify_question_id, axis=1)
    df.to_csv(data_dict + 'train.csv', index=False)

def construct_useful_fields(path='data/all_items_train.txt',sep='<SEP>'):
    with open(path,'r') as file:
        file_content = file.read()
        file_content = file_content.replace('\t',sep)
        file_content = file_content.replace('"','')
        file_content = file_content.replace('﻿','')
        file_lines = file_content.split('\n')
    question_list = {"VH134067":{}, "VH266015":{}, "VH302907":{}, "VH507804":{}, "VH139380":{}, "VH266510":{}, "VH269384":{}, "VH271613":{}, "VH304954":{}, "VH525628":{}}
    heads = file_lines[0].split(sep)
    data_point = file_lines[1:]
    for q in data_point[:-1]:
        q = q.split(sep)
        assert len(q) == len(heads)
        if len(question_list[q[1]]) > 0:
            continue
        if q[1] == "VH507804":
            pass
        pair_list = list(zip(heads,q))
        filtered_pairs = [pair[0] for pair in pair_list if pair[1] != 'NA']
        unique_list = filtered_pairs
        question_list[q[1]].update({'var': unique_list})
    return question_list
    #with open('question.json','w') as f:
    #    json.dump(question_list, f, indent=4)

def save_csv(data_dict, name, data, sep='<SEP>'):
    with open(data_dict + name, 'w', newline='') as output_file:
        csv_writer = csv.writer(output_file)

        # write each line as a row in the CSV file
        for line in data:
            # split the line by commas
            row = line.split(sep)
            # write the row to the CSV file
            csv_writer.writerow(row)


#HELPER function
def _list_to_string(lst, ver='div'):
    flag_mapping = {1: 'incorrect', 2: 'correct', 0: 'empty'}
    if ver == 'div':
        number = {1: 3, 2: 4, 3: 6, 4: 7,0:'nan'}
        n1, p1, n2, p2, n3, p3, n4, p4 = lst
        result = 'nan1/nan2 * nan3/nan4'
        try:
            orders = {p1: number[n1], p2: number[n2], p3: number[n3], p4: number[n4]}
            n1, n2, n3, n4 = [orders[i + 1] for i in range(4)]
            for p, n in orders.items():
                result = result.replace('nan' + str(int(p)), str(n))
            #for p, n in orders.items():
            #result = '{}/{} * {}/{}'.format(n1, n2, n3, n4)
        except:
            pass
    if ver == 'geo':
        result = 'A is {}: {}; B is {}: {}'.format(flag_mapping[int(lst[0])], lst[1], flag_mapping[int(lst[2])], lst[3])
    if ver == '4card':
        number = {1: 17, 2: 27, 3: 54, 4:62, 0:'nan'}
        n1, p1, n2, p2, n3, p3 = lst
        result = 'nan1 * nan2 - nan3'
        try:
            orders = {}
            orders.update({p1: number[n1], p2: number[n2], p3: number[n3]})
            for p, n in orders.items():
                result = result.replace('nan' + str(int(p)), str(n))
        except:
            pass
    if ver == "8card_all":
        index_list = [4, 8, 10]
        score = lst[0]
        result = 'A is ' + flag_mapping[score] + ': '
        lst = lst[1:]
        mean_list = ['s: ','e: ','; B: s: ','e: ']
        # Use list comprehension to create list of sublists
        lst_sep = [str(lst[i:j]) for i, j in zip([0] + index_list, index_list + [len(lst)])]
        for i, name in enumerate(mean_list):
            result += name + lst_sep[i] + ' '
    if ver == "8card_A" or ver == '8card_B':
        split = int(len(lst)/2)
        result = "s: {}, e: {}".format(str(lst[0:split]),str(lst[split:]))

    if ver == 'age':
        index_list = [1,2]
        score = lst[0]
        lst = lst[1:]
        mean_list = ['', '; B: r: ', 'e: ']
        lst_sep = [str(lst[i:j]) for i, j in zip([0] + index_list, index_list + [len(lst)])]
        result = 'A is ' + flag_mapping[score] + ': '
        for i, name in enumerate(mean_list):
            result += name + lst_sep[i] + ' '
    if ver == 'least':
        number = {1: 'w', 2: 'x', 3: 'y', 4:'z', 0:'nan'}
        n1, p1, n2, p2, n3, p3, n4,p4 = lst
        result = '(nan1 * nan2) - (nan3 + nan4)'
        try:
            orders = {}
            orders.update({p1: number[n1], p2: number[n2], p3: number[n3], p4: number[n4]})
            for p, n in orders.items():
                result = result.replace('nan' + str(int(p)), str(n))
        except:
            pass

    if ver == 'slop_2019':
        result = 's: [{}], e: [{}]'.format(lst[0],lst[1])
    return result

def main():
    pass

if __name__ == '__main__':
    """
    Run the code to generate csv file for data
    Saved in data/train.csv
    """
    read_and_transfor_into_csv()

    """
    Futher process the train.csv file to merge some vars
    Create two new vars called: 'context_all' and 'label'
    """
    preprocessing_each_question_var()