from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import pandas as pd
from tqdm import tqdm
import time
from sklearn.cluster import DBSCAN
from model.ModelFactory import ModelFactory as mf
from collections import defaultdict

class Retriever():
    @abstractmethod
    def fetch_examples(self, query, examples):
        raise NotImplementedError

class KNNRetriever(Retriever):
    def __init__(self, args, model=None, tokenizer=None,num_label=None,
                 id2label=None, label2id=None, pooling=None, model_str=None):
        self.retrieverCfg = args.retriever
        self.test = False
        self.ids_item = None
        retrieverCfg = self.retrieverCfg
        self.args = args
        if retrieverCfg.encoderModel:
            self.model_str = retrieverCfg.encoderModel
        else:
            self.model_str = "sentence-transformers/all-MiniLM-L6-v2"
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        if 'bert' in self.model_str:
            self.pooling = 'bert'
        else:
            self.pooling = 'all'
        if pooling is not None:
            self.pooling = pooling
        if model_str != None:
            self.model_str = model_str

        if model is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            (model, tokenizer) = mf.produce_model_and_tokenizer(args, num_label, id2label, label2id)
            self.model = model #.to(self.device)
            self.tokenizer = tokenizer
            #self.model = AutoModel.from_pretrained(self.model_str, cache_dir="cache").to(self.device)
        self.max_len, self.emb_size = self.get_max_len_and_emb_size(self.model_str)
        self.embedding_examples = None

    def get_max_len_and_emb_size(self, model_name):
        # https://www.sbert.net/docs/pretrained_models.html
        if "all-MiniLM" in model_name:
            return 256, 384
        if "all-mpnet" in model_name:
            return 384, 768
        if "all-distilroberta" in model_name:
            return 512, 768
        if "bert" in model_name or "sbert" in model_name:
            return 512, 768

        raise ValueError(f"Properties not specified for {model_name}")

    def update_model(self, args, model=None, tokenizer=None,num_label=None,
                 id2label=None, label2id=None, pooling=None, model_str=None):
        self.__init__(args, model, tokenizer,num_label,
                 id2label, label2id, pooling, model_str)

    # Mean Pooling - Take attention mask into account for correct averaging
    # Adapted from at https://www.sbert.net/examples/applications/computing-embeddings/README.html
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def create_examples_embedding(self, examples = None, model=None, tokenizer = None,
                                  embed_text_name = 'text', pooling='', test=False, save=True):
        #examples is a dictionary
        self.test = test
        self.model = self.model.to(self.device)
        if self.test:
            self.sample_size = False
        embedding_examples = {}
        if pooling == '':
            pooling = self.pooling
        data_dict = {}

        if examples is None:
            data_dict = self.examples
        else:
            for qid, df in examples.groupby(['qid']):
                data_dict[qid] = df
        if model is None:
            model = self.model
            tokenizer = self.tokenizer
        for key, item in data_dict.items():
            parsed_examples = item[embed_text_name].values.tolist()
            with torch.no_grad():
                dataloader = DataLoader(parsed_examples, batch_size=self.retrieverCfg.batch_size)
                all_embeddings = []
                for _, batch_inputs in enumerate(tqdm(dataloader)):
                    token_batch = tokenizer(batch_inputs, padding='max_length', truncation=True,
                                                 max_length=self.max_len, return_tensors='pt').to(self.device)
                    # Call the forward pass of BERT on the batch inputs and get mean pooled embeddings
                    if self.model_str == 'sbert':
                        batch_outputs = model(input=token_batch)
                    else:
                        batch_outputs = model(input_ids=token_batch['input_ids'],
                                               attention_mask=token_batch['attention_mask'], return_dict = True)
                    if pooling == 'all':
                        all_embeddings.append(
                            self.mean_pooling(batch_outputs["last_hidden_state"],
                                              token_batch['attention_mask']).detach().cpu())
                    elif pooling == 'bert' or pooling == 'cls':
                        all_embeddings.append(batch_outputs['pooler_output'])
                    elif pooling == 'sbert':
                        all_embeddings.append(batch_outputs.data['sentence_embedding'])
                    else:
                        raise 'no definition for pooling option {}'.format(pooling)
                embedding_examples[key] = torch.cat(all_embeddings, dim=0)

        if save:
                self.embedding_examples = embedding_examples
                self.examples=data_dict
                self.raw = examples
        qdf_result =  self.obtain_embedding_as_df(embedding_examples, data_dict)
        self.calculate_cluster()

        self.model = self.model.cpu()
        return qdf_result

    def create_topk_list_for_each_item(self, data: pd = None):
        if data is None:
            data = self.examples
        def create_default_dict():
            return defaultdict(list)
        ids_list = defaultdict(create_default_dict)
        print('Select example in advance and save it to save some time')
        for key, item in data.items():
            for d in tqdm(item.iterrows(), total=len(item), position=0):
                d = d[1]
                e = self.fetch_examples(d, generate=True)
                ids = e['id'].tolist()
                ids_list[key][d['id']] = ids
        self.ids_item = ids_list

    def obtain_embedding_as_df(self, embedding_examples = None, example = None):
        if embedding_examples == None:
            embedding_examples = self.embedding_examples
            example = self.examples

        for key in example.keys():
            qdf = example[key]
            embedding = embedding_examples[key]
            embedding = embedding.tolist()
            qdf['emb'] = embedding
        return example

    def calculate_cluster(self, examples = None):
        if examples is None:
            examples = self.examples
        for key, qdf in examples.items():
            X = np.array(qdf['emb'].tolist())
            db = DBSCAN(eps=0.3, min_samples=10).fit(X)
            labels = db.labels_
            qdf['c_label'] = labels

    # q: question only
    # q_a: question and answer
    # q_a_f: question, answer and feedback
    def fetch_examples(self, query, k=None, tok=None,
                       model=None, pooling=None, sample_size=None, generate=False):
        if k is None:
            k = self.retrieverCfg.k
        if model is None:
            model = self.model
            tok = self.tokenizer
        if pooling is None:
            pooling = self.pooling

        if sample_size is None and not self.test:
            sample_size = self.retrieverCfg.sample_size

        parsed_query = query['text']
        key = query['qid']
        query_id = query['id']
        if hasattr(self, 'ids_item') and self.ids_item is not None and (not generate):
            top_k_id = self.ids_item[key][query_id]
            qdf = self.examples[key]
            top_k = qdf[qdf['id'].isin(top_k_id)]
        else:
            token_query = tok(parsed_query, padding='max_length', truncation=True, max_length=self.max_len,
                              return_tensors='pt').to(self.device)
            with torch.no_grad():
                if 'sbert' in self.model_str:
                    model_output_query = model(input=token_query)
                else:
                    model = model.to(self.device)
                    model_output_query = model(**token_query)
                if pooling == 'all':
                    embedding_query = self.mean_pooling(model_output_query["last_hidden_state"],
                                                        token_query["attention_mask"])
                elif pooling == 'cls' or pooling == 'bert':
                    embedding_query = model_output_query['pooler_output']
                elif pooling == 'sbert':
                    embedding_query = model_output_query.data['sentence_embedding']
                else:
                    raise 'no definition for pooling option {}'.format(pooling)
            examples = self.examples[key]
            embedding_examples = self.embedding_examples[key]
            assert len(examples) == len(embedding_examples)
            # sample examples
            if sample_size is not None:
                # reduced_examples = []
                # for c_label, cdf in examples.groupby('c_label'):
                #     if len(cdf) < sample_size:
                #         reduced_examples.append(cdf)
                #     else:
                #         temp = cdf.sample(n=sample_size)
                #         reduced_examples.append(temp)
                # reduced_examples = pd.concat(reduced_examples)
                # examples = reduced_examples
                # examples = examples[examples['id'] != query_id]
                # embedding_examples = examples['emb'].to_list()
                # embedding_examples = torch.tensor(embedding_examples).to(self.device)
                # assert len(examples) == len(embedding_examples)
                examples['emb'] = embedding_examples.detach().cpu().tolist()
                reduced_examples = []
                for item in list(examples.groupby('label1')):
                    l, ldf = item
                    if len(ldf) < sample_size:
                        reduced_examples.append(ldf)
                    else:
                        reduced_examples.append(ldf.sample(n=sample_size, replace=False))
                reduced_examples = pd.concat(reduced_examples)
                examples = reduced_examples
                embedding_examples = examples['emb'].to_list()
                embedding_examples = torch.tensor(embedding_examples).to(self.device)
                assert len(examples) == len(embedding_examples)
            cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            embedding_examples = embedding_examples.to(self.device)
            cos_sim = cos_sim(embedding_query, embedding_examples)  # (examples)
            sorted_examples = [examples.iloc[i.item()] for i in
                               np.argsort(-cos_sim.cpu())]  # negative for descending order
            # convert back to pandas df for easier manipulation
            sorted_examples_df = pd.concat(sorted_examples, axis=1).T.reset_index(drop=True)
            if self.retrieverCfg.exclude_construct_level:
                # Define a custom function to extract the value from the dictionary
                construct_str = f"construct{self.retrieverCfg.exclude_construct_level}"

                def check_value(row, query, construct_str):
                    return row['construct_info'][construct_str][0] != query["construct_info"][construct_str][0]

                # Use the .apply() method to apply the custom function to each row
                sorted_examples_df = sorted_examples_df[
                    sorted_examples_df.apply(check_value, args=(query, construct_str,), axis=1)]
            assert k <= len(examples), "k must be less than the number of examples"
            # NOTE: We assume the first example is the query itself if it's in the pool of examples
            if sorted_examples_df.iloc[0].equals(query):
                top_k = sorted_examples_df.iloc[1:k + 1]
            else:
                top_k = sorted_examples_df.iloc[:k]
        return top_k