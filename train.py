from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
import numpy as np

import utils
from utils import prepare_dataset
from utils import split_data_into_TrainValTest
import utils.var as var
from utils.metric import outer_computer_metrics
import pandas as pd
import json
import os
import shutil
from datasets import Dataset
import datasets
from transformers import DataCollatorWithPadding
from transformers import Trainer
from model.dataset import IncontextDataset
from ExperimentLogger import ExperimentLogger as el
from model.ModelFactory import ModelFactory as mf
from model.EncoderDecoder import FlanT5encoder
class MyTrainer(Trainer):
    def __init__(self, args, device):
        if 'saved_models' in args.lm:
            model_path = os.path.abspath(args.lm)
            args.lm = model_path
        self.device = device
        self.args = args
        self.input_args = args
        # load question information
        # todo add automatic question information generation code if question.json didn't exist
        with open('question.json', 'r') as f:
            question_info = json.load(f)
        self.question_info = question_info

        """
        1. Prepare label and model information
        """
        model, tokenizer = self.prepare_model()

        """
        2. Prepare dataloader
        """
        dataset_dict = self.prepare_dataloader()

        """
        3. Initilized the trainer 
        """

        #3.1 set up trainner
        data_collator = DataCollatorWithPadding(tokenizer= self.tokenizer)
        #3.2 set up evaluation metrics
        compute_metrics = outer_computer_metrics(args, id2label=self.id2label)




        training_args = TrainingArguments(
            output_dir=args.save_model_dir,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.iters,
            weight_decay=args.decay,
            evaluation_strategy="epoch",
            #save_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model = args.best_metric,
            load_best_model_at_end=True,
            push_to_hub=False,
            report_to="wandb"
            #remove_unused_columns = False,
        )

        super().__init__(
            model=model,
            args=training_args,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["val"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        self.test_dataset = dataset_dict['test']


    def prepare_model(self):
        args = self.args
        training_dataset = pd.read_csv(args.train_path)
        training_dataset = training_dataset.rename(columns=var.COLS_RENAME)
        if args.label == 0:
            """
            For label = 0, we use "score_to_predict" column as labels
            """
            training_dataset['label'] = training_dataset[var.LABEL0]
        elif args.label == 1:
            pass
        else:
            raise 'no definition'



        """ 
        BUILDING LABEL MAP and # of Labels 
        Example here: 
        id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        label2id = {"NEGATIVE": 0, "POSITIVE": 1}
        """
        training_dataset['label'] = training_dataset['label'].replace(
            {'1.0': '1', '2.0': '2', '3.0': '3', 1.0: '1', 2.0: '2', 3.0: '3', 1: '1', 2: '2', 3: '3'})
        training_dataset['label'] = training_dataset['label'].astype(str)
        labels = set(list(training_dataset['label']))
        id2label = {}
        id_count = 0
        for elem in sorted(list(labels)):
            id2label[id_count] = elem
            id_count += 1
        label2id = dict((v, k) for k, v in id2label.items())
        num_label = len(labels)
        self.id2label = id2label
        self.label2id = label2id
        self.num_label = num_label
        #todo could apply other architecture: encoder_decoder, multi-classfication head
        (model, tokenizer) = mf.produce_model_and_tokenizer(args, num_label, id2label, label2id)
        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer

    def prepare_dataloader(self):
        """
        Prepare dataloader
        """
        args = self.args
        tokenizer = self.tokenizer
        label_dict = self.label2id

        def preprocess_function_base(examples):
            # todo write own data collator that can take non-tensor input
            result = tokenizer(examples["text"], truncation=True)
            try:
                label_ids = [label_dict[str(label)] for label in examples['label']]
            except:
                label_ids = [label_dict[str(int(label))] for label in examples['label']]
            result['label_ids'] = label_ids
            result['label_str'] = examples['label']
            result['label'] = result['label_ids']
            return result
        def preprocess_function_in_context(examples):
            if args.question_id:
                temp = []
                for x, y in zip(examples['text'], examples['qid']):
                    if y is None or y == '':
                        temp.append(x)
                    else:
                        temp.append(x + 'Question id:  ' + y)
                examples['text'] = temp

            if args.closed_form:
                temp = []
                for x, y in zip(examples['text'], examples[var.CONTEXT_ALL]):
                    if y is None or y == '':
                        temp.append(x)
                    else:
                        temp.append(x + 'Closed form response: ' + y)
                examples['text'] = temp
            result = preprocess_function_base(examples)
            return result





        if args.base:  # the basic classification problem
            preprocess_function = preprocess_function_base
        elif args.in_context:
            preprocess_function = preprocess_function_in_context

        training_dataset = pd.read_csv(args.train_path)
        training_dataset = prepare_dataset(training_dataset, args)
        if args.split:
            train, val, test = split_data_into_TrainValTest(training_dataset)
        elif args.eval_only:
            train = training_dataset
            val  = prepare_dataset(pd.read_csv(args.test_path), args)
            test = val
        else:
            raise 'not define how to split the data'

        if args.debug:
            train, val, test = train[:100], val[:10], test[:10]
        utils.safe_makedirs(args.save_model_dir)
        test.to_csv(args.save_model_dir + 'test.csv')

        """
        Add question-wise dataset for testing 
        """
        question_wise_test = list(test.groupby('qid'))
        if not args.examples or args.base:
            question_wise_test = {key: Dataset.from_pandas(item) for key, item in question_wise_test}
            train, val, test = Dataset.from_pandas(train), Dataset.from_pandas(val), Dataset.from_pandas(test)
            dataset_dict = datasets.DatasetDict({'train': train, 'val': val, 'test': test})
            dataset_dict.update(question_wise_test)
            dataset_dict = dataset_dict.map(preprocess_function, batched=True)
            self.dataset_dict = dataset_dict
        else:
            train_dataset = IncontextDataset(tokenizer=tokenizer, data=train, args=args,
                                     labels_dict = self.label2id)
            val_dataset = IncontextDataset(tokenizer=tokenizer, data=val, args=args,
                                   labels_dict = self.label2id, example=train)
            test_dataset = IncontextDataset(tokenizer=tokenizer, data=test,args=args,
                                    labels_dict = self.label2id, example=train)
            self.dataset_dict = datasets.DatasetDict({'train': train_dataset, 'val': val_dataset, 'test': test_dataset})
            question_wise_test = {key: IncontextDataset(tokenizer=tokenizer, data=item, args=args,
                                   labels_dict = self.label2id, example=train[train['qid'] == key]) for key, item in question_wise_test}
            self.dataset_dict.update(question_wise_test)
            #raise 'not finished'
        return self.dataset_dict

    def save_best_model_and_remove_the_rest(self):
        """
        Save the best model to saved_models/test_name/best/~
        Remove the rest of checkpoints saved models
        """
        run_dir = self._get_output_dir(trial=None)
        output_dir = os.path.join(run_dir, 'best')
        self.save_model(output_dir, _internal_call=True)
        dir_list = os.listdir(run_dir)
        for directory in dir_list:
            if directory.startswith("checkpoint"):
                # Construct the full path to the directory and remove it
                dir_path = os.path.join(run_dir, directory)
                shutil.rmtree(dir_path)


    def save_metrics(self, metrics, alias = ''):
        el.log(metrics)
        path = os.path.join(self.args.output_dir + alias + 'metrics.json')
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4, sort_keys=True)
        q = pd.DataFrame.from_dict(self.question_info).T
        q = q[['name','type']]
        m = pd.DataFrame.from_dict(metrics).T
        m = m.join(q)
        m.to_csv(self.args.output_dir + alias + 'metrics.csv')

    def predict_to_save(self, data:Dataset):
        """
        :param data: the data to evaluate
        :return: the dataframe with an extra column named "predict"
        """
        predicts = self.predict(data)
        data_df = data.to_pandas()
        pred = np.argmax(predicts.predictions, axis=1)
        pred = list(map(lambda x: self.id2label[x], list(pred)))
        data_df['predict'] = pred
        data_df = data_df[['qid', 'text', 'predict', 'label_str']]
        data_df.to_csv(self.args.output_dir + 'test_predict.csv')
        return data_df




