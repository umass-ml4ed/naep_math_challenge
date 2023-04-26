import torch
import random
import numpy as np
import pandas as pd
from utils.var import QUESTION_LIST
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
        train, test = train_test_split(qdf, train_size=ratio[0]/sum(ratio))
        val, test = train_test_split(test, train_size=ratio[1]/sum(ratio[1:]))
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


if __name__ == '__main__':
    test_split_data_into_TrainValTest()