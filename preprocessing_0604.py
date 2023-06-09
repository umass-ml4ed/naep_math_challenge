import csv
from collections import defaultdict
import json
import pandas as pd
import collections
import math
from sklearn.model_selection import StratifiedKFold
from gingerit.gingerit import GingerIt
from tqdm import tqdm, tqdm_notebook
import pandas as pd
import dask.dataframe as dd
from dask.multiprocessing import get
import multiprocessing
import time
import json
from collections import defaultdict
import hashlib
import os
import pathlib
from neuspell import BertChecker
from textblob import TextBlob
sentence = TextBlob('A sentencee to checkk!')
sentence.correct()


#import swifter
score_list = ['rater_1', 'pta_rtr1', 'ptb_rtr1', 'ptc_rtr1', 'score', 'score_to_predict']

def preprocessing_each_question_var(path='data/train_0.csv',
                           data_dict='data/', sep='<SEP>', analysis=True):
    """
    Review each question and merge some variables
    :param path:
    :param data_dict:
    :param sep:
    :param analysis: True for running extra code to analyze the data
    :return:
    """

    def float_to_int(x):
        if isinstance(x, float) and not math.isnan(x):
            return int(x)
        return x

    if 'test' in path:
        name = 'test'
    else:
        name = 'train'

    flag_mapping = {1: 'incorrect', 2: 'correct', 0: 'empty'}
    question_list = json.load(open('question.json', 'r'))
    df = pd.read_csv(path)
    df.fillna(0, inplace=True)


    df_list = []

    type1, type2, type3 = [],[],[]
    type1 = ["VH134067", 'VH266015', "VH302907","VH507804"]
    type2 = ["VH139380","VH304954","VH525628","VH266510_2017", "VH266510_2019"]
    type3 = ["VH269384","VH271613"]
    #type3=["VH269384"]
    #type3=["VH271613"]
    #type2 = ['VH266510_2019']
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
            #qdf['context_A'] = qdf[columns['A'] + ['predict_from']].values.tolist()
            qdf['context_A'] = qdf[columns['A']].values.tolist()
            qdf['context_A'] = qdf['context_A'].apply(
                lambda row: _list_to_string(row, ver='slop_2019'))
            qdf['context_all'] = qdf['context_A']
            #qdf['predict_from'] = qdf['context_all']
            #qdf['context_all'] = ''
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
            qdf['label'] = qdf[score]
            columns_all = columns['all'] + ['predict_from']
            qdf['context_all'] = qdf[columns_all].values.tolist()
            qdf['text1'] = qdf['context_all'].apply(lambda row: _list_to_string(row, ver='8card', parta=True))
            qdf['predict_from'] = qdf['context_all'].apply(lambda row: _list_to_string(row, ver='8card', partb=True))
            qdf['text2'] = qdf['predict_from']
            qdf['context_all'] = qdf['text1'] #qdf['context_all'].apply(lambda row: _list_to_string(row, ver='8card'))
            # for part_name, column_list in columns.items():
            #     qdf['context_' + part_name] = qdf[column_list].values.tolist()
            #     qdf['context_' + part_name] = qdf['context_'+part_name].apply(lambda row: _list_to_string(row, ver='8card_'+part_name))
            if analysis:
                    col = columns['A']
                    correct_A = correct_scores['A']
                    test = qdf[qdf[score].isin(correct_A)]
                    values = collections.Counter(list(test['context_all']))
        if key == 'VH271613':
            for part_name, column_list in columns.items():
                qdf['context_' + part_name] = qdf[column_list].values.tolist()
            qdf['label'] = qdf[score]
            if 'test' in path:
                qdf['label'] = '1A'
            reduced_label = question_list[key]['reduce_label']
            reverse_label_dict = _reverse_label_dict(reduced_label)
            qdf['r_label'] = qdf['label'].apply(lambda row: reverse_label_dict[row])
            qdf['est_score'] = qdf['context_all'].apply(lambda row: _list_to_string(row, ver='age', est=True))
            #qdf['full_response'] = qdf['context_all'].apply(lambda row: _list_to_string(row, ver='age', full=True))
            qdf['text1'] = qdf['context_all'].apply(lambda row: _list_to_string(row, ver='age', full=True, extra=False))
            qdf['text2'] = qdf['predict_from']
            qdf['predict_from'] = qdf['text1'].astype(str) + '. ' +  qdf['text2'].astype(str)
            if analysis:
                values = collections.Counter(list(qdf['partA_response_val']))
                values = collections.Counter(list(qdf['partB_response_val'] + ' e:' + qdf['partB_eliminations']))
            qdf['context_all'] = qdf['context_all'].apply(lambda row: _list_to_string(row, ver='age', parta=True, extra=True))
            if analysis:
                    col = columns['A']
                    correct_A = correct_scores['A']
                    test = qdf[qdf[score].isin(correct_A)]
                    values = collections.Counter(list(test['context_all']))

        df_list.append(qdf)

    merged_df = pd.concat(df_list, axis=0, sort=False)
    merged_df['label'] = merged_df['label'].replace({'1.0': '1', '2.0': '2', 1.0: '1', 2.0: '2', 3.0: '3', 1:'1', 2:'2',3:'3'})
    merged_df['label'] = merged_df['label'].astype(str)


    if 'test' in path:
        import random
        merged_df['score_to_predict'] = random.choices(['1','2','3'], k=len(df))
        merged_df['label'] = merged_df['score_to_predict']
    df = merged_df
    # Apply the function to convert float values to int in the dataframe
    df = df.applymap(float_to_int)
    # add id for each example
    if 'test' not in path:
        df = _split_fold(df, type_all=type_all)
    else:
        df['fold'] = 10

    if 'test' in path:
        train_df = pd.read_csv('data/train2.csv')
        df = pd.concat([df, train_df])
    df['id'] = df['student_id']

    df = df.applymap(float_to_int)
    df.to_csv(data_dict + 'train2' + '.csv', index=False)
    # question_list = construct_useful_fields()
    # extra = ['text1', 'text2', 'context_A','context_B','context_all','label', 'r_label','est_score', 'fold', 'id']
    # for key in type_all:
    #     qdf = df[df['accession'] == key]
    #     if "VH266510" in key:
    #         cols_to_include = question_list["VH266510"]['var'] + extra
    #     else:
    #         cols_to_include = question_list[key]['var'] + extra
    #     qdf = qdf[cols_to_include]
    #     # Save the resulting DataFrame to a CSV file
    #     qdf.to_csv(data_dict + 'train_' + key + '.csv', index=False)

def grammarly(df):
    checker = BertChecker()
    checker.from_pretrained()
    parse_test_csv(df, spell_check=True, checker=checker)
    return df


def parse_test_csv(df, spell_check=False, checker=None):
    if (spell_check):
        txt = df['predict_from']
        txt_spell_checked = checker.correct_strings(txt)
        df['predict_from'] = txt_spell_checked
    return df

def grammaly_check(row):
    #parser = GingerIt()
    #if row=='NA': return row
    #try:
    #    text = parser.parse(row)
    #except:
        # print('Error with {}'.format(row))
        # print('Pause 1 minutes')
        # time.sleep(60)
        # try:
        #     text = parser.parse(row)
        # except:
        #     print('Still error')
        #     time.sleep(60)
        #     return row
        # return text['result']
    #return text['result']
    if row=='NA': return row
    sentence = TextBlob(str(row))
    text = sentence.correct()
    return text.raw


def read_and_transfor_into_csv(train_path='data/all_items_train.txt', test_path = 'data/all_items_test.txt',
                           data_dict='data/', sep='<SEP>'):
    parser = GingerIt()
    def load_df(path):
        with open(path, 'r') as train_file:
            file_content = train_file.read()
            file_content = file_content.replace('\t', sep)
            file_content = file_content.replace('\ufeff', '')
            file_content = file_content.replace('"', '')
            file_lines = file_content.split('\n')
        # question_list = json.load(open('question.json','r'))
        question_list = construct_useful_fields()
        # create a CSV writer object to write to the output file
        heads = file_lines[0]
        question_dict = {q: [heads] for q in question_list.keys()}
        number_of_field = len(heads.split(sep))
        for line in file_lines[1:]:
            try:
                question = line.split(sep)[1].replace('"', '')
                assert len(line.split('<SEP>')) == number_of_field, print(
                    '{} and {} doesnt match'.format(len(line.split('<SEP>')), number_of_field))
                assert question in question_dict, print(question)
                question_dict[question].append(line)
            except:
                print('here')

        split_lines = [line.split('<SEP>') for line in file_lines]
        if split_lines[-1][0] == '':
            split_lines = split_lines[:-1]
        df = pd.DataFrame(split_lines[1:], columns=split_lines[0])
        return df
    # Define a function to modify the question_id based on the year value
    def modify_question_id(row):
        if row['accession'] == "VH266510":
            return f"VH266510_{str(int(row['year']))}"
        else:
            return row['accession']

    df = load_df(test_path)
    df['accession'] = df.apply(modify_question_id, axis=1)
    #df = df.iloc[range(10)]
    df.to_csv(data_dict + 'test_0.csv', index=False)
    #apply grammaly check
    # print('Start')
    #df = grammarly(df)
    # pool = multiprocessing.Pool()
    # result = pool.map(grammaly_check, df['predict_from'].to_list())
    # df['predict_from'] = result
    #df['predict_from'] = df['predict_from'].apply(grammaly_check)
    df.to_csv(data_dict + 'test_0.csv', index=False)
    #ddata = dd.from_pandas(df, npartitions=30)
    #df['predict_from'] = ddata.map_partitions(lambda df: df['predict_from'].apply((lambda row: grammaly_check(*row)), axis=1)).compute(get=get)
    # with mp.Pool(mp.cpu_count()) as pool:
    #     df['predict_from1'] = pool.map(grammaly_check, df['predict_from'])
    #     df['parsed_xml_v11'] = pool.map(grammaly_check, df['parsed_xml_v1'])
    #     df['parsed_xml_v21'] = pool.map(grammaly_check, df['parsed_xml_v2'])
    #     df['parsed_xml_v31'] = pool.map(grammaly_check, df['parsed_xml_v3'])

    list_all = ['predict_from', 'parsed_xml_v1', 'parsed_xml_v2', 'parsed_xml_v3']
    # for l in tqdm(list_all, total=len(list_all)):
    #     #df[l] = df[l].apply(grammaly_check)
    #     temp = []
    #     for d in tqdm(df[l].tolist(), total=len(df), position=0):
    #         temp.append(grammaly_check(d))
    #     df[l] = temp
    print('Done testing')
    # print('finish one')
    # df.to_csv(data_dict + 'test_0.csv', index=False)
    # df['parsed_xml_v1'] = df['parsed_xml_v1'].apply(grammaly_check)
    # print('finish two')
    # df.to_csv(data_dict + 'test_0.csv', index=False)
    # df['parsed_xml_v2'] = df['parsed_xml_v2'].apply(grammaly_check)
    # print('finish three')
    # df.to_csv(data_dict + 'test_0.csv', index=False)
    # df['parsed_xml_v3'] = df['parsed_xml_v3'].apply(grammaly_check)
    # print('finish four')
    # df.to_csv(data_dict + 'test_0.csv', index=False)


    print('start training dataset')
    df = load_df(train_path)
    df['accession'] = df.apply(modify_question_id, axis=1)
    df.to_csv(data_dict + 'train_0.csv', index=False)
    # pool = multiprocessing.Pool()
    # result = pool.map(grammaly_check, df['predict_from'].to_list())
    # df['predict_from'] = result
    # for l in tqdm(list_all, total=len(list_all)):
    #     #df[l] = df[l].apply(grammaly_check)
    #     temp = []
    #     for d in tqdm(df[l].tolist(), total=len(df),position=0):
    #         temp.append(grammaly_check(d))
    #     df[l] = temp
    # df = grammarly(df)
    # df['predict_from'] = df['predict_from'].apply(grammaly_check)
    # df.to_csv(data_dict + 'train_0.csv', index=False)
    # Apply the modify_question_id function to the question_id column
    # df['accession'] = df.apply(modify_question_id, axis=1)
    #df.to_csv(data_dict + 'train_0.csv', index=False)
    #apply grammaly check
    # print('Start')
    # df['predict_from'] = df['predict_from'].apply(grammaly_check)
    # print('finish one')
    # df.to_csv(data_dict + 'train_0.csv', index=False)
    # df['parsed_xml_v1'] = df['parsed_xml_v1'].apply(grammaly_check)
    # print('finish two')
    # df.to_csv(data_dict + 'train_0.csv', index=False)
    # df['parsed_xml_v2'] = df['parsed_xml_v2'].apply(grammaly_check)
    # print('finish three')
    # df.to_csv(data_dict + 'train_0.csv', index=False)
    # df['parsed_xml_v3'] = df['parsed_xml_v3'].apply(grammaly_check)
    # print('finish four')
    # df.to_csv(data_dict + 'train_0.csv', index=False)



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
def _list_to_string(lst, ver='div', est=False, full=False, extra=False, parta=False, partb=False):
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
    if ver == "8card":
        predict_str = lst[-1]
        lst = lst[:-1]
        a = {0:'Null', 1:'1/8', 2:'3/8', 3:'5/8', 4:'6/8'}
        #a2 = {0:'Null',1:'younger', 2:'older'}
        if extra:
            b = {0: 'No Idea. ', 1:'Replacing the card will change Trent probability of wining', 2:'Replacing the card wont change Trent probability of wining' }
        else:
            b = {0: 'No Idea. ', 1: 'Yes, the probability will change', 2: 'No, the probability won\'t change'}
        def process_a(lst):
            score = lst[0]
            result = 'Part A is ' + flag_mapping[score] + ': '
            choose = 0
            for i, c in enumerate(lst[1:5]):
                if c:
                    choose += 1
                    result += a[i]
            if choose == 0:
                result += 'No answer'
            return result
        def process_b(lst):
            lst = lst[-4:]
            result = ''
            if lst[0] and not lst[1]:
                result = b[1]
            elif not lst[0] and lst[1]:
                result = b[2]
            elif lst[0] and lst[1]:
                result = 'Not sure '
            if result == '':
                select = {1: 'Yes, the probability will change', 2: 'No, the probability won\'t change'}
                if lst[0]:
                    select.pop(1)
                if lst[1]:
                    select.pop(2)
                if len(select) == 0 or len(select) == 2:
                    result = 'Not Sure '
                if len(select) == 1:
                    result = list(select.values())[0]
            return result
        partA = process_a(lst)
        partB_c = process_b(lst)
        if parta:
            result = partA #context_all
        elif partb:
            if predict_str == 0:
                predict_str = 'I don\'t know'
            result = partB_c + ', ' + str(predict_str) #score_to_predict
        #mean_list = ['s: ','e: ','; B: s: ','e: ']
        # Use list comprehension to create list of sublists
        #lst_sep = [str(lst[i:j]) for i, j in zip([0] + index_list, index_list + [len(lst)])]
        #for i, name in enumerate(mean_list):
        #    result += name + lst_sep[i] + ' '
    if ver == "8card_A" or ver == '8card_B':
        split = int(len(lst)/2)
        result = "s: {}, e: {}".format(str(lst[0:split]),str(lst[split:]))

    if ver == 'age':
        a1 = {0:'Null', 1:'4', 2:'8'}
        a2 = {0:'Null',1:'younger', 2:'older'}
        if extra:
            b = {0: 'No Idea. ', 1:'Phil age is not 3 times of Alex in 10 year', 2:'Phil is not 2 years older than Zach in ten year'}
        else:
            b = {0: 'No Idea. ', 1: 'A', 2: 'B'}

        index_list = [1,2]
        score = lst[0]
        lst = lst[1:]
        lst_sep = [str(lst[i:j]) for i, j in zip([0] + index_list, index_list + [len(lst)])]
        result = 'Part A is ' + flag_mapping[score]
        if extra:
            result += ': '
        def process_a(y):
            if y == 0:
                return 'No answer'
            x = y.strip('[]')
            x = x.replace("'","")
            x = x.replace('c(','')
            x = x.replace(')','')
            x = x.replace(',',' ')
            x = x.replace('  ',' ')
            x = x.split(' ')
            if len(x) == 1:
                result = a1[int(x[0])] + ' Null'
            else:
                try:
                    a, b = x
                except:
                    print(x)
                if len(a) == 0 or a == 'NA':
                    a = 0
                if len(b) == 0:
                    b = 0
                result = a1[int(a)] + ' ' + a2[int(b)]
            return result
        def process_b(x, y):
            answer = {}
            if 'TRUE' in x or 'FALSE' in x:
                pass
            if '1' in x or 'TRUE ' in x:
                answer.update({1:b[1]})
            if '2' in x or ' TRUE' in x:
                answer.update({2:b[2]})
            if ('1' in y or 'TRUE ' in y) and 1 in answer:
                answer.pop(1)
            if ('2' in y or ' TRUE' in y) and 2 in answer:
                answer.pop(2)
            if len(answer) == 0:
                answer.update({0:b[0]})
            return answer
        if extra:
            result += process_a(lst_sep[0])
        if parta:
            return result


        part_b = process_b(lst_sep[1], lst_sep[2])
        if full:
            part_b = 'and '.join(list(part_b.values()))
            if extra:
                part_b = 'and '.join(list(part_b.values()))
            else:
                result = 'I choose ' + part_b
        elif not full and not est:
            part_b = ', '.join(list(part_b.values()))
            result = '' + part_b + '. ' + result
        elif est:
            if (1 in part_b) and (2 not in part_b):
                result = 1
            else:
                result = 0
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
        choose = lst[0].split(' ')
        eliminate = lst[1].split(' ')
        #predict_str = lst[2]
        def parta(choose, eliminate):
            a = {0:'A', 1: 'B', 2: 'C', 3: 'D'}
            b = {0:'The slope of the lines must be equal.', 1: 'The y-intercepts of the lines must be equal.',
                 2: 'The slopes of the lines cannot be equal.', 3: 'The y-intercepts of the lines cannot be equal.'}
            result = []
            for i, c in enumerate(choose):
                if c =='TRUE':
                    result.append(b[i])
            assert len(result) <= 1, 'more chice made'

            if len(result) == 1:
                result = result[0]
                return result
            for i, c in enumerate(eliminate):
                if c == 'TRUE':
                    a.pop(i)
            if len(a) == 0:
                return 'Not sure.'
            else:
                return 'I choose ' + ' and '.join(list(a.values()))
        # if predict_str == 0:
        #     predict_str = 'No idea.'
        #result = parta(choose, eliminate) + ' ' + predict_str
        result = 's: [{}], e: [{}]'.format(lst[0],lst[1])
    return result


def _reduce_label(lst, d):
    pass

def _reverse_label_dict(d):
    reverse_dict = {}
    for key, values in d.items():
        for value in values:
            reverse_dict[value] = key
    return reverse_dict

def _split_fold(df, type_all = [], n_splits=10):
    df['fold'] = 0
    alls = []
    skf = StratifiedKFold(n_splits=n_splits)
    for key in type_all:
        if '384' in key:
            print('here?')
        qdf = df[df['accession'] == key]
        qdf.reset_index(drop=True, inplace=True)
        for fold_id, (_, test_index) in enumerate(skf.split(qdf, qdf['label'])):
            qdf.loc[test_index, 'fold'] = fold_id
        alls.append(qdf)
    def _sanity_check(qdf):
        print('Check fold algorithm')
        a = list(qdf.groupby('fold'))
        for k, i in a:
            print(k)
            print(i['label'].value_counts())


    alls = pd.concat(alls, ignore_index=True)
    _sanity_check(alls)
    return alls



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
    #preprocessing_each_question_var(analysis=False)
    preprocessing_each_question_var(analysis=False)
    preprocessing_each_question_var(path='data/test_0.csv', analysis=False)