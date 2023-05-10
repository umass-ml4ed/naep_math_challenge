import torch
import random
import numpy as np
import copy

import pandas as pd
from utils.var import QUESTION_LIST
import utils.var as var
from sklearn.model_selection import train_test_split


def split_data_into_TrainValTest(dataset: pd, ratio: list= [8, 1, 1]):
    """
    The split is based on question-wise.
    :param dataset: pandas dataframe
    :param ratio: the ratio to split into train/val/test
    :return: three datagrame
    """
    trains, vals, tests = [], [], []

    for q in QUESTION_LIST:
        try:
            qdf = dataset[dataset['accession'] == q]
        except:
            qdf = dataset[dataset['qid'] == q]
        train, val = train_test_split(qdf, train_size=ratio[0]/sum(ratio))
        if len(ratio) == 3:
            val, test = train_test_split(val, train_size=ratio[1]/sum(ratio[1:]))
        else:
            test = None
        trains.append(train)
        vals.append(val)
        tests.append(test)
    train = pd.concat(trains, ignore_index=True)
    val = pd.concat(vals, ignore_index=True)
    test = pd.concat(tests,ignore_index=True)
    return train, val, test

def test_split_data_into_TrainValTest():
    path = 'data/train.csv'
    data = pd.read_csv(path)
    split_data_into_TrainValTest(data)



def tokenize_function(tokenizer, sentences_1, sentences_2=None):
    if(sentences_2 == None):
        return tokenizer(sentences_1, padding=True, truncation=True, return_tensors="pt")
    else:
        return tokenizer(sentences_1, sentences_2, padding=True, truncation=True, return_tensors="pt")

class CollateWraperParent(object):
    def __init__(self, tokenizer, min_label):
        self.tokenizer = tokenizer
        self.min_label = min_label


class CollateWraper(CollateWraperParent):
    def __init__(self, tokenizer, min_label):
        super().__init__(tokenizer, min_label)

    def __call__(self, batch):
        # construct features
        features = [d['txt'] for d in batch]
        inputs = tokenize_function(self.tokenizer, features)

        # construct labels
        labels = torch.tensor([d['l1'] if d['l1'] >= 0 else d['l2'] for d in batch]).long() - self.min_label
        inputs['labels'] = labels
        return {"inputs": inputs}

def prepare_dataset(data, args):
    data['label'] = data['label'].astype(str)
    data['label'] = data['label'].replace(
        {'1.0': '1', '2.0': '2', '3.0': '3', 1.0: '1', 2.0: '2', 3.0: '3', 1: '1', 2: '2', 3: '3'})
    # unify labels' names
    data = data.rename(columns=var.COLS_RENAME)
    if args.label == 0:
        """
        For label = 0, we use "score_to_predict" column as labels
        """
        data['label'] = data[var.LABEL0]
    elif args.label == 1:
        pass
    else:
        raise 'no definition'

    if args.base:  # the basic classification problem
        """
        The base case: 
        input: sutdent response 
        output: label 

        BASE_COLS = ['qid', 'label','text']
        Get corresponding information and rename the column 
        """
        data = data[var.BASE_COLS]
    elif args.in_context:
        useful_cols = copy.deepcopy(var.BASE_COLS)
        if args.closed_form:
            useful_cols += [var.CONTEXT_ALL]
        data = data[useful_cols]

    else:
        raise 'No task information defined'
    data['label'] = data['label'].astype(str)
    return data


if __name__ == '__main__':
    test_split_data_into_TrainValTest()