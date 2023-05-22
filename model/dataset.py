import random
import math
#from datasets import Dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import pandas as pd
from utils import var
from collections import defaultdict

def _append_closed_form(example):
    result = example['text']
    if isinstance(example[var.CONTEXT_ALL], str):
        result += var.SEP + var.PRE_CLOSED + example[var.CONTEXT_ALL]
    return result





class IncontextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, data: pd.DataFrame, args = None, labels_dict = {}, example=None):
        self.num_examples = args.n_examples
        self.tokenizer = tokenizer
        self.raw_data = data
        self.args = args
        self.labels = list(labels_dict.keys())
        self.labels_dict = labels_dict
        if example is None:
            self.examples = self.raw_data
            self.is_examples_same_as_testing = True
        else:
            #merge raw data with examples
            self.examples = example
            self.is_examples_same_as_testing = False
        self._rerange_data()

    def _rerange_data(self):
        self.raw_data['label_str'] = self.raw_data['label']
        if self.args.closed_form:
            data = self.raw_data
            data.loc[data[var.CONTEXT_ALL].notna(), 'text'] = data['text'].astype(str) + var.SEP + var.PRE_CLOSED + data[var.CONTEXT_ALL].astype(str)
            self.raw_data = data

            # if not self.is_examples_same_as_testing:
            #     data = self.examples
            #     data.loc[data[var.CONTEXT_ALL].notna(), 'text'] = data['text'] + var.SEP + var.PRE_CLOSED + data[var.CONTEXT_ALL].astype(str)
            #     self.examples = data
            # Don't need this since if examples is different then it is train dataset, we already applied function on what we expected

        data_dict = defaultdict(dict)
        for name, df in self.examples.groupby(['qid', 'label']):
            qid, l = name
            data_dict[qid][l] = df
        self.examples_dict = data_dict

    def __len__(self):
        return len(self.raw_data)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        def preprocess_function_base(examples):
            result = self.tokenizer(examples["text"], truncation=True)
            other = {}
            label_ids = self.labels_dict[str(examples['label'])]
            result['label_ids'] = label_ids
            #result['label_str'] = examples['label']
            if self.args.label == 2:
                result[var.EVAL_LABEL] = examples[var.EVAL_LABEL]
                result[var.EST_SCORE] = examples[var.EST_SCORE]
            #result['label'] = result['label_ids']
            return result

        args = self.args
        if isinstance(i, list):
            raise 'Not finished'
            item_df = self.raw_data.iloc[i] #.to_dict('list')
            #if args.in_context and args.closed_form:
            #    item_df['text'] = item_df.apply(_append_closed_form, axis = 1)
            if args.examples:
                item_df['example'] = item_df.apply(self._select_example, axis=1)
                item_df['text'] = item_df['text'] + var.SEP + var.PRE_EXAMPLE + item_df['example']

            item_df = item_df.apply(preprocess_function_base, axis=1)
            result = item_df.to_dict('list')
        else:
            item_df = self.raw_data.iloc[[i]]#.to_dict('list')
            item_df = item_df.to_dict('list')
            item_df = {k: v[0] for k, v in item_df.items()}
            #if args.in_context and args.closed_form:
            #    item_df['text'] = _append_closed_form(item_df)
            if args.examples:
                item_df['example'] = self._select_example(i)
                item_df['text'] = item_df['text'] + var.SEP + var.PRE_OVERALL_EXAMPLE + item_df['example']

            item_df['text'] = var.PRE_QUERY_GRADE + item_df['text']
            batch_result = preprocess_function_base(item_df)
            #batch_result.update({'raw':result})
            result = batch_result
        return result

    def _select_example(self, i):
        #i = int(example.index.values)
        args = self.args
        #choose one example for each label class
        qid = self.raw_data.iloc[i]['qid']
        data_dict = self.examples_dict[qid]
        example_index = []
        for key, v in data_dict.items():
            temp = list(v.index)
            if i in temp and self.is_examples_same_as_testing:
                temp.remove(i)
            if args.sample_seed != -1:
                random.seed(args.sample_seed)
            choose_index = random.sample(temp, self.num_examples)
            example_index += choose_index
        try:
            examples_df = self.examples.loc[example_index][['text','label']]
        except:
            print('Error with example index', example_index)
        examples_df = examples_df.apply(lambda x: var.PRE_EXAMPLE + x['text'] + var.SEP + var.PRE_SCORE + str(x['label']), axis=1)
        examples_text = var.SEP.join(examples_df)
        return examples_text

    def to_pandas(self):
        return self.raw_data








