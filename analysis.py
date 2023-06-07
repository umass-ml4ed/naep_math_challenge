import csv
from collections import defaultdict
import json
import pandas as pd
import collections
from sklearn.metrics import cohen_kappa_score
import numpy as np
from utils.utils import itemwise_avg_kappa
import tqdm
def analysis_item_slop(path='data/train.csv'):
    df = pd.read_csv(path)
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
    print('round kappa is {}'.format(kappa))
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
        print(f'{key} kappa is {kappa}')
    if len(qdf_temp) > 0:
        qdf = pd.concat(qdf_temp)
        qdf[text] = qdf[text].apply(round)
        predictions = np.array(qdf[text].tolist())
        labels = np.array(qdf['label_str'].tolist())
        kappa = float(cohen_kappa_score(labels, predictions, weights='quadratic'))
        print(f'VH266510_1719 kappa is {kappa}')
        all_qdf.append(qdf)
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
        my_label.append(score)
        j+=1
    df['modified'] = my_label
    print('Done!')
    return df


def self_grade():
    path_t = 'test_predict.csv'
    path_v =  'val_predict.csv'
    test = pd.read_csv(path_t)
    val = pd.read_csv(path_v)
    val, _ = itemwise_avg_kappa(val)
    test, _ = itemwise_avg_kappa(test)
    val_not_sure = val[~val['avg0'].isin([1, 2, 3])]
    test_not_sure = test[~test['avg0'].isin([1, 2, 3])]
    _check_precentage(val)

    test_new = _relabel(test_not_sure)
    test['modified'] = test['avg']
    test_old = test[~test['id'].isin(test_new['id'].to_list())]
    merged_test = pd.concat([test_old, test_new])

    #evaluatate
    print('======MODIFIED======')
    a = itemwise_kappa(merged_test, 'modified')
    print('======old======')
    b = itemwise_kappa(test)
    merged_test.to_csv(path_t)
    print('Done')



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
    else:
        user_input = ''
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
    print('TEST: ==============')
    testqd = directly_evluation(patht, start, end)
    print('VAL========================')
    valqd = directly_evluation(pathv, start, end)
    print('REDUCED======================')
    r_qd = directly_evluation(path_r, start, end)
    merge(testqd, r_qd)

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
    score = input("a: self grade b: correct type 2, c: avg score")
    if score == "a":
        self_grade()
    if score == 'b':
        correct_type_2()
    if score == 'c':
        get_avg_score()

