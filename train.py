from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
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
class MyTrainer(Trainer):
    def __init__(self, cfg, deivce):
        if 'saved_models' in cfg.lm:
            model_path = os.path.abspath(cfg.lm)
            cfg.lm = model_path
        self.deivce = deivce
        self.cfg = cfg
        self.input_cfg = cfg
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
        compute_metrics = outer_computer_metrics(cfg, id2label=self.id2label)




        training_args = TrainingArguments(
            output_dir=cfg.save_model_dir,
            learning_rate=cfg.lr,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            num_train_epochs=cfg.iters,
            weight_decay=cfg.decay,
            evaluation_strategy="epoch",
            #save_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model = cfg.best_metric,
            load_best_model_at_end=True,
            push_to_hub=False,
            report_to="none"
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
        cfg = self.cfg
        training_dataset = pd.read_csv(cfg.train_path)
        training_dataset = training_dataset.rename(columns=var.COLS_RENAME)
        if cfg.label == 0:
            """
            For label = 0, we use "score_to_predict" column as labels
            """
            training_dataset['label'] = training_dataset[var.LABEL0]
        elif cfg.label == 1:
            pass
        else:
            raise 'no definition'



        """ 
        BUILDING LABEL MAP and # of Labels 
        Example here: 
        id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        label2id = {"NEGATIVE": 0, "POSITIVE": 1}
        """
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
        model = AutoModelForSequenceClassification.from_pretrained(cfg.lm, num_labels=num_label,
                                                                   id2label = id2label, label2id = label2id)
        tokenizer = AutoTokenizer.from_pretrained(cfg.lm)
        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer

    def prepare_dataloader(self):
        """
        Prepare dataloader
        """
        cfg = self.cfg
        tokenizer = self.tokenizer
        label_dict = self.label2id

        def preprocess_function_base(examples):
            # todo write own dataset class
            # todo write own data collator that can take non-tensor input
            result = tokenizer(examples["text"], truncation=True)
            label_ids = [label_dict[str(label)] for label in examples['label']]
            result['label_ids'] = label_ids
            result['label_str'] = examples['label']
            result['label'] = result['label_ids']
            return result
        def preprocess_function_in_context(examples):
            if cfg.question_id:
                temp = []
                for x, y in zip(examples['text'], examples['qid']):
                    if y is None or y == '':
                        temp.append(x)
                    else:
                        temp.append(x + 'Question id:  ' + y)
                examples['text'] = temp

            if cfg.closed_form:
                temp = []
                for x, y in zip(examples['text'], examples[var.CONTEXT_ALL]):
                    if y is None or y == '':
                        temp.append(x)
                    else:
                        temp.append(x + 'Closed form response: ' + y)
                examples['text'] = temp


            result = preprocess_function_base(examples)
            return result


        training_dataset = pd.read_csv(cfg.train_path)
        training_dataset['label'] = training_dataset['label'].astype(str)


        #unify labels' names
        training_dataset = training_dataset.rename(columns=var.COLS_RENAME)
        if cfg.label == 0:
            """
            For label = 0, we use "score_to_predict" column as labels
            """
            training_dataset['label'] = training_dataset[var.LABEL0]
        elif cfg.label == 1:
            pass
        else:
            raise 'no definition'

        if cfg.base: #the basic classification problem
            """
            The base case: 
            input: sutdent response 
            output: label 
            
            
            BASE_COLS = ['qid', 'label','text']
            Get corresponding information and rename the column 
            """
            preprocess_function = preprocess_function_base
            training_dataset = training_dataset[var.BASE_COLS]
        elif cfg.in_context:
            useful_cols = var.BASE_COLS
            if cfg.closed_form:
                useful_cols += [var.CONTEXT_ALL]
            preprocess_function = preprocess_function_in_context
            training_dataset = training_dataset[useful_cols]

        else:
            raise 'No task information defined'


        if cfg.split:
            train, val, test = split_data_into_TrainValTest(training_dataset)
        else:
            raise 'not define how to split the data'

        if cfg.debug:
            train, val, test = train[:100], val[:100], test[:100]

        """
        Add question-wise dataset for testing 
        """
        question_wise_test = list(test.groupby('qid'))
        if not cfg.examples:
            question_wise_test = {key: Dataset.from_pandas(item) for key, item in question_wise_test}
            train, val, test = Dataset.from_pandas(train), Dataset.from_pandas(val), Dataset.from_pandas(test)
            dataset_dict = datasets.DatasetDict({'train': train, 'val': val, 'test': test})
            dataset_dict.update(question_wise_test)
            dataset_dict = dataset_dict.map(preprocess_function, batched=True)
            self.dataset_dict = dataset_dict
        else:
            train_dataset = IncontextDataset(tokenizer=tokenizer, data=train, cfg=cfg,
                                     labels_dict = self.label2id)
            val_dataset = IncontextDataset(tokenizer=tokenizer, data=val, cfg=cfg,
                                   labels_dict = self.label2id, example=train)
            test_dataset = IncontextDataset(tokenizer=tokenizer, data=test,cfg=cfg,
                                    labels_dict = self.label2id, example=train)
            self.dataset_dict = datasets.DatasetDict({'train': train_dataset, 'val': val_dataset, 'test': test_dataset})
            question_wise_test = {key: IncontextDataset(tokenizer=tokenizer, data=item, cfg=cfg,
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
        path = os.path.join(self.cfg.output_dir + alias + 'metrics.json')
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4, sort_keys=True)

        q = pd.DataFrame.from_dict(self.question_info).T
        q = q[['name','type']]
        m = pd.DataFrame.from_dict(metrics).T
        m = m.join(q)
        m.to_csv(self.cfg.output_dir + alias + 'metrics.csv')



