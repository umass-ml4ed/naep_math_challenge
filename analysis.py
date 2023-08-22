import csv
from collections import defaultdict
import json
import pandas as pd
import collections
from sklearn.metrics import cohen_kappa_score
import numpy as np
from utils.utils import itemwise_avg_kappa
from sklearn.metrics import accuracy_score
import tqdm
from collections import defaultdict
from utils.metric import bias
from utils import  var
from prettytable import PrettyTable

def analysis_bias(test , train, path1 = '../data/train.csv', path2 = 'test_predict.csv', label = 'avg'):
    #l = input('enter prefix')
    #train = pd.read_csv(path1)
    #train = train.drop_duplicates(keep='first')
    #test = pd.read_csv(l + path2)
    #test = test.drop_duplicates(keep='first')
    type = ['srace10', 'dsex', 'accom2', 'iep', 'lep']
    for t in type:
        try:
            test = test.drop(columns=[t])
        except:
            print('no ', t)
    #test, _ = itemwise_avg_kappa(test)
    test[label + 't'] = test[label].apply(round)
    label = label + 't'
    result = pd.merge(test,train, on='id', how='left')
    #result['qid'] = 'all same'
    #step one calculate each group std, mean and population
    type = ['srace10', 'dsex', 'accom2', 'iep', 'lep']
    type_n = {}
    groupby_qid = {}
    info_list = {}
    overall = []
    for qid, qdf in list(result.groupby('qid')):
        info = defaultdict(dict)
        for t in type:
            type_n[t] = list(set(qdf[t].tolist()))
            for name, tdf in list(qdf.groupby(t)):
                predict = tdf[label]
                avg = predict.mean()
                std = predict.std()
                pop = predict.count()
                info[t][name] = [avg, std, pop]
        final_result = {i: defaultdict(dict) for i in type}
        for t in type:
            info_temp = info[t]
            item_list = list(info_temp.keys())
            for key1 in item_list:
                for key2 in item_list:
                    if key1 != key2:
                        value = bias(info_temp[key1], info_temp[key2])
                        final_result[t][key1][key2] = value
                        overall.append(abs(value))
                    else:
                        final_result[t][key1][key2] = 0


            try:
                print('For item ', t, ' ',  qid, ' ', var.QUESTION_TO_NAME[qid])
            except:
                print('For item ', t, ' ', qid, ' ALL')
            draw_table(final_result[t], name=t)
        groupby_qid[qid] = final_result
        info_list[qid] = info
        overall1 = np.mean( list(set(overall)))
        print('Overall ', overall1)

    print('done')


def bias_calculate(test, type=['accom2'], label='avg'):
    test[label + 't'] = test[label].apply(round)
    label = label + 't'
    type_n = {}
    groupby_qid = {}
    info_list = {}
    overall = []
    for qid, qdf in list(test.groupby('qid')):
        info = defaultdict(dict)
        for t in type:
            type_n[t] = list(set(qdf[t].tolist()))
            for name, tdf in list(qdf.groupby(t)):
                predict = tdf[label]
                avg = predict.mean()
                std = predict.std()
                pop = predict.count()
                info[t][name] = [avg, std, pop]
        final_result = {i: defaultdict(dict) for i in type}
        for t in type:
            info_temp = info[t]
            item_list = list(info_temp.keys())
            for key1 in item_list:
                for key2 in item_list:
                    if key1 != key2:
                        value = bias(info_temp[key1], info_temp[key2])
                        final_result[t][key1][key2] = value
                        overall.append(abs(value))
                    else:
                        final_result[t][key1][key2] = 0
            draw_table(final_result[t], name=t)
        groupby_qid[qid] = final_result
        info_list[qid] = info
        overall1 = np.mean( list(set(overall)))
        print('Overall ', overall1)
    return info_list, groupby_qid

def bias_treatement(test, type='accom2', label='avg'):
    path1 = '../data/train.csv'
    path2 = 'val_predict.csv'
    train = pd.read_csv(path1)
    train = train.drop_duplicates(keep='first')
    test = pd.read_csv(path2)
    key = 'VH271613'
    test = test[test['qid']==key]
    type2 = ['srace10', 'dsex', 'accom2', 'iep', 'lep']
    for t in type2:
        try:
            test = test.drop(columns=[t])
        except:
            print('no ', t)
    test = test.drop_duplicates(keep='first')
    test = pd.merge(test,train, on='id', how='left')

    print('True')
    info_t, resul_t = bias_calculate(test, label='label_str')
    print(info_t)

    print('before')
    _, metric = itemwise_avg_kappa(test, start=-1)
    print(metric)
    info1, resul1 =  bias_calculate(test)
    print(info1)

    print('after')
    test.loc[test[type] == 1, label] += 0.2
    info2, resul2 = bias_calculate(test)
    _, metric = itemwise_avg_kappa(test, start=-1)
    print(metric)


    print('here')




def draw_table(data, name='race'):
    import json
    from prettytable import PrettyTable
    # Create a PrettyTable object
    table = PrettyTable()
    # Add headers
    headers = [""] + [f"{name} {i}" for i in data.keys()]
    table.field_names = headers
    table.float_format = '.3'
    # Add rows with scores
    for i in data.keys():
        row = [f"{name} {i}"] + list(data[i].values())
        table.add_row(row)

    # Print the table
    print(table)


def analysis_item_slop(path='data/train.csv'):
    df = pd.read_csv(path)
    df = df.drop_duplicates(keep='first')
    df = df[df["accession"] == "VH266510_2017"]
    df = df[['score', 'predict_from', 'eliminations']]
    socres = ['3','2A','2B','1A','1B']
    for s in socres:
        print('\nscore ', s, '\n')
        test = df[ df['score'] == s]
        values = collections.Counter(list(test['eliminations']))
        values2 = collections.Counter(list(test['predict_from']))
        correct = 0
        wrong = 0
        partial = 0
        for v in values:
            length = len(test)
            if v == '0':
                print('{}/{} = {:.2f}% data with no selection information\n'.format(values[v], length, values[v]/length*100))
            elif ('2' in v and '1' in v and '4' in v):
                correct += values[v]
            else:
                wrong += values[v]
            if '3' not in v and v != '0':
                partial += values[v]
        print('{}/{} = {:.2f}% data with selected response correct (eliminate 1,2,4)\n'.format(correct, length,
                                                                  correct / length * 100, v))
        print('{}/{} = {:.2f}% data with selected response incorrect\n'.format(wrong, length,
                                                                  wrong / length * 100, v))
    df = pd.read_csv(path)
    df = df.drop_duplicates(keep='first')
    df = df[df["accession"] == "VH266510_2019"]
    #df = df[['score', 'predict_from', 'context_all']]
    for s in socres:
        print('\nscore ', s, '\n')
        test = df[df['score'] == s]
        values = collections.Counter(list(test['context_all']))
        values2 = collections.Counter(list(test['predict_from']))
        correct = 0
        wrong = 0
        partial = 0
        for v in values:
            length = len(test)
            if v == '0':
                print('{}/{} = {:.2f}% data with no selection information\n'.format(values[v], length,
                                                                                    values[v] / length * 100))
            elif ('2' in v and '1' in v and '4' in v):
                correct += values[v]
            else:
                wrong += values[v]
            if '3' not in v and v != '0':
                partial += values[v]
        print('{}/{} = {:.2f}% data with selected response correct (eliminate 1,2,4)\n'.format(correct, length,
                                                                                               correct / length * 100,
                                                                                               v))
        print('{}/{} = {:.2f}% data with selected response incorrect\n'.format(wrong, length,
                                                                               wrong / length * 100, v))
# def sample_avg_value(x):
#     x['avg']
#     return x['predict'+str(index)]
def directly_evluation(path='predict.csv', start = 2, epoch=None):
    df = pd.read_csv(path)
    df = df.drop_duplicates(keep='first')
    labels = np.array(df['label_str'].tolist())
    predict_name = df.columns
    if epoch is None:
        epoch = 0
        for p in predict_name:
            if 'predict' in p:
                i = p.split('predict')
                i = i[1]
                if i != '' and int(i) > epoch:
                    epoch = int(i)
    epoch += 1
    column = ['predict'+str(i) for i in range(start,epoch)]
    df['avg'] = df[column].mean(axis=1)
    df1 = df['avg'].apply(round)
    predictions = np.array(df1.tolist())
    kappa = float(cohen_kappa_score(labels, predictions, weights='quadratic'))
    accuracy = float(accuracy_score(labels, predictions))
    print('round kappa is {}, acc is {}'.format(kappa, accuracy))
    return itemwise_kappa(df)
    # df1 = df.apply(sample_one_value, axis=1)
    # predictions = np.array(df1.tolist())
    # kappa = float(cohen_kappa_score(labels, predictions, weights= 'quadratic'))
    # print('ave kappa is {}'.format(kappa))
    #
    # predictions = np.array(df['predict'].tolist())
    # labels = np.array(df['label_str'].tolist())
    # kappa = float(cohen_kappa_score(labels, predictions, weights= 'quadratic'))
    # print('last kappa is {}'.format(kappa))

def itemwise_kappa(df, text='avg'):
    qdf_temp = []
    all_qdf = []
    for key, qdf in df.groupby('qid'):
        qdf[text] = qdf[text].apply(round)
        all_qdf.append(qdf)
        if '2017' in key or '2019' in key:
            qdf_temp.append(qdf)
        predictions = np.array(qdf[text].tolist())
        labels = np.array(qdf['label_str'].tolist())
        kappa = float(cohen_kappa_score(labels, predictions, weights='quadratic'))
        accuracy = float(accuracy_score(labels, predictions))
        print(f'{key} kappa is {kappa}, acc is {accuracy}')
    if len(qdf_temp) > 0:
        qdf = pd.concat(qdf_temp)
        qdf[text] = qdf[text].apply(round)
        predictions = np.array(qdf[text].tolist())
        labels = np.array(qdf['label_str'].tolist())
        kappa = float(cohen_kappa_score(labels, predictions, weights='quadratic'))
        accuracy = float(accuracy_score(labels, predictions))
        print(f'VH266510_1719 kappa is {kappa}, acc is {accuracy}')
        #all_qdf.append(qdf)
    return pd.concat(all_qdf)

def merge(q, rq):
    rq_id = rq['id'].tolist()
    q_rest = q[~q['id'].isin(rq_id)]
    q_merged = pd.concat([q_rest, rq])
    itemwise_kappa(q_merged)

def _check_precentage(test):
    test_not_sure = test[~test['avg0'].isin([1, 2, 3])]
    reduced_test_id = test_not_sure['id'].to_list()
    # sanity check
    test_wrong = test[test['label_str'] != test['avg']]
    r = test_wrong[test_wrong['id'].isin(reduced_test_id)]
    reduced_test = test[test['id'].isin(reduced_test_id)]
    print('Size is {}. The select testing example {:.2f} % cover true '
          'miss-labelling example, there are {:.2f} % selected '
          'reduced examples are true wrong'.format(len(reduced_test_id), len(r) / len(test_wrong) * 100,
                                                   100 * len(r) / len(reduced_test)))

def _relabel(df:pd.DataFrame):
    my_label = []
    j = 0
    for i, row in df.iterrows():
        print(f'{j}/{len(df)}: !!!===========')
        response = row['text']
        model_predict = row['avg0']
        true_answer = row['label_str']

        print(f'Student response: \n ==>{response} <==')
        print(f'Model predict: {model_predict}')
        score = input("Enter your score, must be 1 , 2 or 3")
        print(f'True answer is {true_answer}\n')
        my_label.append(int(score))
        j+=1
    df['modified'] = my_label
    print('Done!')
    return df


def self_grade():
    key = 'VH271613'
    path_t = 'test_predict.csv'
    path_v =  'val_predict.csv'
    test = pd.read_csv(path_v)
    test = test.drop_duplicates(keep='first')
    val = pd.read_csv(path_v)
    val = val.drop_duplicates(keep='first')
    val, _ = itemwise_avg_kappa(val)
    test, _ = itemwise_avg_kappa(test)
    val_not_sure = val[~val['avg0'].isin([1, 2, 3])]
    test_not_sure = test[~test['avg0'].isin([1, 2, 3])]
    #test_not_sure = test_not_sure.sample(n=5)
    _check_precentage(val)

    test_new = _relabel(test_not_sure)
    test['modified'] = test['avg']
    test_old = test[~test['id'].isin(test_new['id'].to_list())]
    merged_test = pd.concat([test_old, test_new])
    merged_test['avg'] = merged_test['avg'].astype(int)
    #evaluatate
    print('======MODIFIED======')
    a = itemwise_kappa(merged_test, 'modified')
    print('======old======')
    b = itemwise_kappa(test)
    #merged_test.to_csv(path_t)
    print('Done')

def gold_check():
    def d(i):
        return di[i]
    di = {'1A': 1, '1B': 1, '2A': 2, '2B': 2, '3': 3}
    error = pd.read_csv('error.csv')
    val = pd.read_csv('val_predict_9.csv')
    error = error.merge(val, on='id')
    error['grade'] = error['grade'].apply(d)
    human_new = error['grade'].tolist()
    labels = error['label_str'].astype(int).tolist()
    model_predict = error['predict'].astype(int).tolist()
    kappa1 = cohen_kappa_score(human_new, labels)
    kappa2 = cohen_kappa_score(human_new, model_predict)

def overall_best_result(root_path=''):
    def evaluation(df, start=1, epoch=None):
        column = ['predict' + str(i) for i in range(start, epoch)]
        df['avg'] = df[column].mean(axis=1)
        df_result, result = itemwise_avg_kappa(df, start, epoch)
        return df_result, result
    def loop_evaluation(df):
        best_score = defaultdict(float)
        best_epoch = {}
        epoch = 0
        predict_name = df.columns
        #val_df = {}
        for p in predict_name:
            if 'predict' in p:
                i = p.split('predict')
                i = i[1]
                if i != '' and int(i) > epoch:
                    epoch = int(i)
        for i in range(epoch+1):
            start = i + 1
            for j in range(start+2, epoch+1):
                df_result, result = evaluation(df, start,j)
                for qid, kappa in result.items():
                    if best_score[qid] < kappa:
                        best_score[qid] = kappa
                        best_epoch[qid]= [start, j]
                        #val_df[qid] = df_result[df_result['qid']==qid]

        #val_df = pd.concat(val_df)
        return best_score, best_epoch

    import os
    from collections import defaultdict
    #give the path, check all of the predict.csv file under the path and filter out the best result
    best_score = defaultdict(float)
    best_path = {}
    val_df = {}
    test_df = {}
    root_path = os.path.abspath(root_path)
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if 'val_predict.csv' in file:
                file_path = os.path.join(root, file)
                val_path = file_path
                file_path = file_path.replace('val_predict.csv', '')
                test_path = file_path + 'test_predict.csv'

                val = pd.read_csv(val_path)
                test = pd.read_csv(test_path)
                val = val.drop_duplicates(keep='first')
                test = test.drop_duplicates(keep='first')

                best_val_score, best_epoch = loop_evaluation(val)
                for qid, kappa in best_val_score.items():
                    if best_score[qid]< kappa:
                        temp_epoch = best_epoch[qid]
                        best_score[qid] = kappa
                        best_path[qid] = {'path': file_path, 'epoch': best_epoch[qid]}

                        temp_val, m = evaluation(val, start=temp_epoch[0], epoch=temp_epoch[1])
                        assert m[qid] == kappa
                        temp_test, m = evaluation(test, start=temp_epoch[0], epoch=temp_epoch[1])
                        qid = qid.replace('avg_', '')
                        qid = qid.replace('_kappa', '')
                        if qid == 'slope':
                            continue
                        qid = var.NAME_TO_QUESTION[qid]
                        temp_val = temp_val[temp_val['qid'] == qid]
                        temp_test = temp_test[temp_test['qid']==qid]
                        #list_needed=['id','qid','label_str','avg', '']
                        #temp_val = temp_val[list_needed]
                        #temp_test = temp_test[list_needed]
                        val_df[qid] = temp_val[temp_val['qid'] == qid]
                        test_df[qid] = temp_test[temp_test['qid']==qid]
    val_df = pd.concat(val_df)
    test_df = pd.concat(test_df)

    val_df.to_csv(os.path.join(root_path, 'val_result.csv'))
    test_df.to_csv(os.path.join(root_path, 'test_result.csv'))
    with open(os.path.join(root_path, 'best_score.json'), 'w') as f:
        json.dump(best_score, f, indent=2)
    with open(os.path.join(root_path, 'best_epoch.json'), 'w') as f:
        json.dump(best_path, f, indent=2)
    print('done')
    print(best_score)



def get_avg_score():
    path = 'saved_models/bert_meanP_random_c_e2_l0_8card_fold_9_8_loss0/'
    path = 'saved_models/bert_c_e2_l0_slope_2019_fold_9_8_loss0/'
    patht = path + 'test_predict.csv'
    pathv = path + 'val_predict.csv'
    path_r = path + 'r_test_predict.csv'
    #analysis_item_slop()
    user_input = input("enter predict path, q to exit")
    if user_input.lower() == 'q':
        exit()
    elif user_input.lower() == 'r':
        should_reload = True
    if user_input == 'avg':
        patht = 'test_predict.csv'
        pathv = 'val_predict.csv'

        df = pd.read_csv(pathv)
        df = df.drop_duplicates(keep='first')
        labels = np.array(df['label_str'].tolist())
        predictions = np.array(df['avg'].tolist())
        kappa = float(cohen_kappa_score(labels, predictions, weights='quadratic'))
        accuracy = float(accuracy_score(labels, predictions))
        print('VAL round kappa is {}, acc is {}'.format(kappa, accuracy))
        itemwise_kappa(df)

        df = pd.read_csv(patht)
        df = df.drop_duplicates(keep='first')
        labels = np.array(df['label_str'].tolist())
        predictions = np.array(df['avg'].tolist())
        kappa = float(cohen_kappa_score(labels, predictions, weights='quadratic'))
        accuracy = float(accuracy_score(labels, predictions))
        print('TEST round kappa is {}, acc is {}'.format(kappa, accuracy))
        itemwise_kappa(df)

        exit()


    start = input("enter start epoch")
    if start == '': start = 1
    start = int(start)

    end = input("enter end epoch")
    if end == '': end = None
    else:
        end = int(end)
    patht = user_input + 'test_predict.csv'
    pathv = user_input + 'val_predict.csv'
    path_r = user_input + 'r_test_predict.csv'
    print('VAL========================')
    valqd = directly_evluation(pathv, start, end)
    print('TEST: ==============')
    testqd = directly_evluation(patht, start, end)
    # print('REDUCED======================')
    # r_qd = directly_evluation(path_r, start, end)
    # merge(testqd, r_qd)

def correct_type_2(text='avg'):
    def correct(df, name='avg'):
        text = df['text'].tolist()
        label = df[name].tolist()
        est_score = []
        for i, t in enumerate(text):
            if  not 'Phil is not 2 years older than Zach in ten year' in t:
                est_score.append(label[i])
            elif label[i] == '3':
                est_score.append('2')
            else:
                est_score.append(label[i])
        df['est'] = est_score
    path_t = 'test_predict.csv'
    path_v =  'val_predict.csv'
    test = pd.read_csv(path_t)
    val = pd.read_csv(path_v)
    val, _ = itemwise_avg_kappa(val)
    test, _ = itemwise_avg_kappa(test)

    correct(val)
    correct(test)

    print('val')
    itemwise_kappa(val)
    itemwise_kappa(val, 'est')
    print('test')
    itemwise_kappa(test)
    itemwise_kappa(test,'est')


if __name__ == '__main__':
    i = 0
    score = input("a: self grade \nb: correct type 2, \nc: avg score \n d analysisi bias \n f overall best \n g bias treatment")
    while True:
        if score == "a":
            self_grade()
        if score == 'b':
            correct_type_2()
        if score == 'c':
            get_avg_score()
        if score == 'd':
            l = input('choose label: 1 avg 2 label_str')
            if l == '' or l == '1':
                l = 'avg'
            elif l == '2':
                l='label_str'

            l2 = input('enter prefix')
            path1 = '../data/train.csv'
            path2 = 'val_predict.csv'
            train = pd.read_csv(path1)
            train = train.drop_duplicates(keep='first')
            test = pd.read_csv(l2 + path2)
            test = test.drop_duplicates(keep='first')
            analysis_bias(test, train, label=l)
        if score == 'g':
            bias_treatement('')
        if score == 'f':
            path = input('Enter root path')
            overall_best_result(root_path=path)
        score = input("\n Try another one? \n a: self grade \nb: correct type 2, \nc: avg score \n d analysisi bias")
