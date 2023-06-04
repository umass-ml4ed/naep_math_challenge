import random
import copy
import math
#from datasets import Dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import pandas as pd
from utils import var
from collections import defaultdict
from sentence_transformers import SentenceTransformer,util,InputExample

def _append_closed_form(example):
    result = example['text']
    if isinstance(example[var.CONTEXT_ALL], str):
        result += var.SEP + var.PRE_CLOSED + example[var.CONTEXT_ALL]
    return result

def rerange_data(raw_data, args):
    raw_data['label_str'] = raw_data['label']
    data = raw_data
    if args.closed_form:
        data.loc[data[var.CONTEXT_ALL].notna(), 'text'] = data['text'].astype(str) + '. ' + var.PRE_CLOSED + data[
            var.CONTEXT_ALL].astype(str)
        data['text'] = data['text'].replace('..', '.')
    return data

def rerange_examples(examples):
    data_dict = defaultdict(dict)
    freq_dict = defaultdict(dict)
    for name, df in examples.groupby(['qid', 'label']):
        qid, l = name
        data_dict[qid][l] = df
        freq_dict[qid][l] = len(df)
    for q, v in freq_dict.items():
        over_len = sum(v.values())
        freq_dict[q] = {key: value / over_len for key, value in v.items()}

    freq_dict = freq_dict
    examples_dict = data_dict
    return freq_dict, examples_dict





class IncontextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, data: pd.DataFrame, args = None,
                 labels_dict = {}, example=None, question_dict={}, retriever = None, eval=False, **kwargs):
        self.loc = False
        self.num_examples = args.n_examples
        self.tokenizer = tokenizer
        self.raw_data = copy.deepcopy(data)
        self.args = args
        self.labels = list(labels_dict.keys())
        self.labels_dict = labels_dict
        self.question_dict = question_dict
        self.retriever = retriever
        self.eval = eval
        if example is None:
            self.examples = self.raw_data
            self.is_examples_same_as_testing = True
        else:
            self.examples = example
            self.is_examples_same_as_testing = False
        self._rerange_data()

    def _rerange_data(self):
        # self.raw_data['label_str'] = self.raw_data['label']
        # if self.args.closed_form:
        #     data = self.raw_data
        #     data.loc[data[var.CONTEXT_ALL].notna(), 'text'] = data['text'].astype(str) + '. ' + var.PRE_CLOSED + data[var.CONTEXT_ALL].astype(str)
        #     data['text'] = data['text'].replace('..', '.')
        data_dict = defaultdict(dict)
        freq_dict = defaultdict(dict)
        for name, df in self.examples.groupby(['qid', 'label']):
            qid, l = name
            data_dict[qid][l] = df
            freq_dict[qid][l] = len(df)
        for q, v in freq_dict.items():
            over_len = sum(v.values())
            freq_dict[q] = {key: value/over_len for key, value in v.items()}

        self.freq_dict = freq_dict
        self.examples_dict = data_dict

    def __len__(self):
        return len(self.raw_data)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):


        args = self.args
        if isinstance(i, list):
            raise 'Not finished'
        else:
            if self.args.im_balance:
                i = self._modify_index_to_balance_labels(i)
            if self.loc:
                try:
                    item_df = self.raw_data.loc[[i]]
                except:
                    print('error ', i)
            else:
                try:
                    item_df = self.raw_data.iloc[[i]]
                except:
                    print('error')
            item = item_df
            item_df = item_df.to_dict('list')
            item_df = {k: v[0] for k, v in item_df.items()}
            if args.examples:
                item_df['example'] = self._select_example(i)
                if args.ag:
                    item_df['text'] = var.PRE_OVERALL_EXAMPLE + item_df['example'] + var.SEP + item_df['text'] + '. ' + var.PRE_SCORE
                else:
                    item_df['text'] = item_df['text'] + var.SEP + var.PRE_OVERALL_EXAMPLE + item_df['example']
            if not args.ag:
                item_df['text'] = var.PRE_QUERY_GRADE + item_df['text']
            batch_result = self._preprocess_function_base(item_df)
            result = batch_result
        return result

    def _preprocess_function_base(self, examples):
        result = self.tokenizer(examples["text"], truncation=True)
        label_ids = self.labels_dict[str(examples['label'])]
        result['label_ids'] = label_ids
        if self.args.label == 2:
            result[var.EVAL_LABEL] = examples[var.EVAL_LABEL]
            result[var.EST_SCORE] = examples[var.EST_SCORE]
        if self.args.multi_head:
            try:
                result['question_ids'] = self.question_dict[examples['qid']]
            except:
                print('Kye errors, ', self.question_dict)
        return result

    def _select_example(self, i, to_text=True):
        args = self.args
        if args.retriever.name == 'knn' and self.retriever != None:
            try:
                query = self.raw_data.iloc[i]
            except:
                print('error, i is {} len of raw data is {}'.format(i, len(self.raw_data)))
            examples_df = self.retriever.fetch_examples(query)
        else:
            # choose one example for each label class
            if self.loc:
                qid = self.raw_data.loc[i]['qid']
            else:
                qid = self.raw_data.iloc[i]['qid']
            data_dict = self.examples_dict[qid]
            example_index = []
            for key, v in data_dict.items():
                temp = list(v.index)
                if i in temp and self.is_examples_same_as_testing:
                    temp.remove(i)
                if args.sample_seed != -1:
                    random.seed(args.sample_seed)
                if (len(temp) < self.num_examples):
                    choose_index = temp
                else:
                    choose_index = random.sample(temp, self.num_examples)
                example_index += choose_index
            try:
                examples_df = self.examples.loc[example_index][['id', 'text', 'label']]
            except:
                print('Error with example index', example_index)

        if not to_text:
           return examples_df
        examples_df = examples_df.apply(
                lambda x: var.PRE_EXAMPLE + x['text'] + '. ' + var.PRE_SCORE + str(x['label']), axis=1)


        examples_text = var.SEP.join(examples_df)

        return examples_text

    def _modify_index_to_balance_labels(self, i):
        j = -1
        item_df = self.raw_data.iloc[[i]]
        qid = item_df['qid'].to_list()
        qid = qid[0]
        if qid in var.Imbalance and self.eval == False:
            if item_df['label_str'].to_list()[0] in ['1', '1A', '1B']:
                random_num = random.random()
                if random_num > 0.90:
                    data = self.raw_data[self.raw_data['qid'] == qid]
                    data = data[data['label_str'].isin(['2','2A','2B','3'])]
                    temp = list(data.index)
                    j = random.sample(temp, 1)
                    j = j[0]
                    assert self.raw_data.iloc[i]['qid'] == self.raw_data.loc[j]['qid']
                    self.loc = True
        if j != -1:
            i = j
            self.loc = True
        else:
            self.loc = False
        return i

    def to_pandas(self):
        return self.raw_data


class SentenceBertDataset(IncontextDataset):
    def if_rerun_cossim(self, i):
        if hasattr(self,'epoch_count'):
            if self.i == i:
                self.epoch_count += 1
                return True
        else:
            self.epoch_count = 0
            self.i = i
            return False

    def __getitem__(self, i):
        if self.args.im_balance:
            i = self._modify_index_to_balance_labels(i)
        if self.loc:
            try:
                item_df = self.raw_data.loc[[i]]
            except:
                raise 'Error when loading with index {}'.format(i)
        else:
            try:
                item_df = self.raw_data.iloc[[i]]
            except:
                raise 'Error when loading with index {}'.format(i)
        item_df = item_df.to_dict('list')
        item_df = {k: v[0] for k, v in item_df.items()}
        item2_list  = self._select_example(i, to_text=False)
        sentence1 = item_df['text']
        label1 = item_df['label']

        random_num = random.random()
        if random_num > 0.5:
            temp = item2_list[item2_list['label'] == label1]
            temp = temp['id'].tolist()
            if len(temp) == 0:
                temp = item2_list['id'].tolist()
                temp = random.sample(temp,1)
                temp = temp[0]
            else:
                temp = temp[-1]
            item2 = item2_list[item2_list['id'] == temp]
        else:
            temp = item2_list[item2_list['label'] != label1]
            temp = temp['id'].tolist()
            if len(temp) == 0:
                temp = item2_list['id'].tolist()
                temp = random.sample(temp,1)
                temp = temp[0]
            else:
                temp = temp[0]
            item2 = item2_list[item2_list['id'] == temp]
        item2 = item2.iloc[0].to_dict()


        sentence2 = item2['text']
        label2 = item2['label']
        if self.args.loss == 0:
            if label1 == label2:
                score = 1
            else:
                score = 0
            return InputExample(texts=[sentence1, sentence2], label=score)
        elif self.args.loss == 1:
            if random.random() > 0.5:
                a = random.sample(self.label_list[str(label1)], k=1)
                a = a[0]
                label2, sentence2 = self.labels[a], self.histories[a]
            else:
                index = random.randint(0, len(self.steps) - 1)
                label2, sentence2 = self.labels[index], self.histories[index]
            import torch
            label1 = torch.Tensor(label1)
            label2 = torch.Tensor(label2)
            cosine_scores = util.cos_sim(label1, label2)
            cosine_scores = cosine_scores.numpy()

            return InputExample(texts=[sentence1, sentence2], label=cosine_scores[0, 0])








