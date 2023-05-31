from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import pandas as pd
from tqdm import tqdm
import time

class Retriever():
    @abstractmethod
    def fetch_examples(self, query, examples):
        raise NotImplementedError

class KNNRetriever(Retriever):
    def __init__(self, retrieverCfg, model=None, tokenizer=None):
        self.retrieverCfg = retrieverCfg
        if retrieverCfg.encoderModel:
            self.model_str = retrieverCfg.encoderModel
        else:
            self.model_str = "sentence-transformers/all-MiniLM-L6-v2"
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        if 'bert' in self.model_str:
            self.pooling = 'bert'
        else:
            self.pooling = 'all'
        if model is not None:
            self.model = model
        else:
            self.model = AutoModel.from_pretrained(self.model_str, cache_dir="cache").to(self.device)
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_str, cache_dir="cache")
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
        if "bert" in model_name:
            return 512, 768
        raise ValueError(f"Properties not specified for {model_name}")

    # Mean Pooling - Take attention mask into account for correct averaging
    # Adapted from at https://www.sbert.net/examples/applications/computing-embeddings/README.html
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def create_examples_embedding(self, examples, model=None, tokenizer = None,
                                  embed_text_name = 'text', pooling=''):
        #examples is a dictionary
        self.embedding_examples = {}
        if pooling == '':
            pooling = self.pooling
        data_dict = {}
        for qid, df in examples.groupby(['qid']):
            data_dict[qid] = df
        self.examples=data_dict
        self.raw = examples
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
                    batch_outputs = model(input_ids=token_batch['input_ids'],
                                               attention_mask=token_batch['attention_mask'], return_dict = True)
                    if pooling == 'all':
                        all_embeddings.append(
                            self.mean_pooling(batch_outputs["last_hidden_state"],
                                              token_batch['attention_mask']).detach().cpu())
                    elif pooling == 'bert' or pooling == 'cls':
                        all_embeddings.append(batch_outputs['pooler_output'])
                    else:
                        raise 'no definition for pooling option {}'.format(pooling)
                embedding_examples = torch.cat(all_embeddings, dim=0)
            self.embedding_examples[key] = embedding_examples
        return self.obtain_embedding_as_df()

    def obtain_embedding_as_df(self):
        embedding_examples = self.embedding_examples
        example = self.examples

        for key in example.keys():
            qdf = example[key]
            embedding = embedding_examples[key]
            embedding = embedding.tolist()
            qdf['emb'] = embedding
        return example

    # q: question only
    # q_a: question and answer
    # q_a_f: question, answer and feedback
    def fetch_examples(self, query, k=None, tok=None, model=None, pooling='all'):
        if k is None:
            k = self.retrieverCfg.k
        if model is None:
            model = self.model
            tok = self.tokenizer
        if pooling is None:
            pooling = self.pooling
        parsed_query = query['text']
        key = query['qid']
        examples = self.examples[key]
        embedding_examples = self.embedding_examples[key]
        assert len(examples) == len(embedding_examples)
        token_query = tok(parsed_query, padding='max_length', truncation=True, max_length=self.max_len,
                                     return_tensors='pt').to(self.device)  # (examples, seq, hidden)
        with torch.no_grad():
            model_output_query = model(**token_query)
            if pooling == 'all':
                embedding_query = self.mean_pooling(model_output_query["last_hidden_state"],
                                                token_query["attention_mask"])
            elif pooling == 'cls' or pooling == 'bert':
                embedding_query = model_output_query['pooler_output']
            else:
                raise 'no definition for pooling option {}'.format(pooling)


        # TODO make this a hyperparameter
        # Compute cosine similarity between query and each example

        start_time = time.time()
        # TODO With a normalized embedding, this is the same as dot product
        cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        embedding_examples = embedding_examples.to(self.device)
        cos_sim = cos_sim(embedding_query, embedding_examples)  # (examples)
        sorted_examples = [examples.iloc[i.item()] for i in np.argsort(-cos_sim.cpu())]  # negative for descending order
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
        end_time = time.time()
        # print(f"similarity computing elapsed: {end_time - start_time}")

        # In case  you wanna try L2
        # l2_sim = torch.sqrt(torch.sum(torch.square(embedding_query - embedding_examples), dim=1))
        # sorted_examples = [examples.iloc[i.item()] for i in np.argsort(l2_sim)] # ascending order

        # Return the n closest examples to the query
        assert k < len(examples), "k must be less than the number of examples"
        # NOTE: We assume the first example is the query itself if it's in the pool of examples
        if sorted_examples_df.iloc[0].equals(query):
            top_k = sorted_examples_df.iloc[1:k + 1]
        else:
            top_k = sorted_examples_df.iloc[:k]
        return top_k