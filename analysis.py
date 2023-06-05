import csv
from collections import defaultdict
import json
import pandas as pd
import collections
from sklearn.metrics import cohen_kappa_score
import numpy as np

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
def directly_evluation(path='predict.csv', start = 1):
    df = pd.read_csv(path)
    labels = np.array(df['label_str'].tolist())
    predict_name = df.columns
    epoch=0
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
    itemwise_kappa(df)

    def sample_one_value(x):
        index = np.random.randint(start, epoch)
        return x['predict' + str(index)]

    # df1 = df.apply(sample_one_value, axis=1)
    # predictions = np.array(df1.tolist())
    # kappa = float(cohen_kappa_score(labels, predictions, weights= 'quadratic'))
    # print('ave kappa is {}'.format(kappa))
    #
    # predictions = np.array(df['predict'].tolist())
    # labels = np.array(df['label_str'].tolist())
    # kappa = float(cohen_kappa_score(labels, predictions, weights= 'quadratic'))
    # print('last kappa is {}'.format(kappa))

def itemwise_kappa(df):
    qdf_temp = []
    for key, qdf in df.groupby('qid'):
        df1 = qdf['avg'].apply(round)
        if '2017' in key or '2019' in key:
            qdf_temp.append(qdf)
        predictions = np.array(df1.tolist())
        labels = np.array(qdf['label_str'].tolist())
        kappa = float(cohen_kappa_score(labels, predictions, weights='quadratic'))
        print(f'{key} kappa is {kappa}')
    if len(qdf_temp) > 0:
        qdf = pd.concat(qdf_temp)
        df1 = qdf['avg'].apply(round)
        predictions = np.array(df1.tolist())
        labels = np.array(qdf['label_str'].tolist())
        kappa = float(cohen_kappa_score(labels, predictions, weights='quadratic'))
        print(f'VH266510_1719 kappa is {kappa}')



if __name__ == '__main__':
    #path = 'saved_models/test_predict.csv'
    #path = 'saved_models/test_predict2.csv'
    #path = 'saved_models/bert_c_e2_l0_8card_fold_9_8_loss0/test_predict.csv'
    #path = 'saved_models/bert_c_e2_l0_8card_fold_9_8_loss0/val_predict.csv'
    path = 'saved_models/bert_meanP_random_c_e2_l0_8card_fold_9_8_loss0/'
    path = 'saved_models/bert_c_e2_l0_slope_2019_fold_9_8_loss0/'
    patht = path + 'test_predict.csv'
    pathv = path + 'val_predict.csv'
    #analysis_item_slop()
    user_input = input("enter predict path, q to exit")
    if user_input.lower() == 'q':
        exit()
    elif user_input.lower() == 'r':
        should_reload = True
    else:
        user_input = ''
    patht = user_input + 'test_predict.csv'
    pathv = user_input + 'val_predict.csv'
    directly_evluation(patht)
    directly_evluation(pathv)