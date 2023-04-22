from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from utils import split_data_into_TrainValTest
from utils.globel_var import BASE_COLS
from utils.metric import compute_metrics
import pandas as pd
import json
import os
import shutil
from datasets import Dataset
import datasets
from transformers import DataCollatorWithPadding
from transformers import Trainer
class MyTrainer(Trainer):
    def __init__(self, args, deivce):
        if 'saved_models' in args.lm:
            model_path = os.path.abspath(args.lm)
            args.lm = model_path
        self.deivce = deivce
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
        model, tokenizers = self.prepare_model()

        """
        2. Prepare dataloader
        """
        dataset_dict = self.prepare_dataloader()

        """
        3. Initilized the trainer 
        """
        data_collator = DataCollatorWithPadding(tokenizer= self.tokenizer)
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

            #remove_unused_columns = False,
        )

        super().__init__(
            model=self.model,
            args=training_args,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["val"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        self.test_dataset = dataset_dict['test']


    def prepare_model(self):
        args = self.args
        training_dataset = pd.read_csv(args.train_path)
        if args.label == 0:
            label_name = 'score_to_predict'
        else:
            raise 'no type of label defined'

        """ 
        BUILDING LABEL MAP and # of Labels 
        Example here: 
        id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        label2id = {"NEGATIVE": 0, "POSITIVE": 1}
        """
        labels = set(list(training_dataset[label_name]))
        id2label = {}
        id_count = 0
        for elem in sorted(labels):
            id2label[id_count] = elem
            id_count += 1
        label2id = dict((v, k) for k, v in id2label.items())
        num_label = len(labels)
        self.id2label = id2label
        self.label2id = label2id
        self.num_label = num_label

        model = AutoModelForSequenceClassification.from_pretrained(args.lm, num_labels=num_label,
                                                                   id2label = id2label, label2id = label2id)
        tokenizer = AutoTokenizer.from_pretrained(args.lm)
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
        def preprocess_function(examples):
            #todo write own dataset class 
            result = tokenizer(examples["text"], truncation=True)
            label_ids = [label_dict[label] for label in examples['label']]
            result['label_ids'] = label_ids
            return result
        training_dataset = pd.read_csv(args.train_path)

        if args.base and args.label == 0: #the basic classification problem
            """
            BASE_COLS = {'accession':'qid','score_to_predict':'label','predict_from':'text'}
            Get corresponding information and rename the column 
            """
            training_dataset = training_dataset[BASE_COLS.keys()]
            training_dataset = training_dataset.rename(columns=BASE_COLS)
        else:
            raise 'No task information defined'


        if args.split:
            train, val, test = split_data_into_TrainValTest(training_dataset)
        else:
            raise 'not define how to split the data'

        if args.debug:
            train, val, test = train[:100], val[:100], test[:100]

        train, val, test = Dataset.from_pandas(train), Dataset.from_pandas(val), Dataset.from_pandas(test)
        dataset_dict = datasets.DatasetDict({'train': train, 'val': val, 'test':test})
        dataset_dict = dataset_dict.map(preprocess_function, batched=True)
        self.dataset_dict = dataset_dict
        return dataset_dict

    def save_best_model_and_remove_the_rest(self):
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
        path = os.path.join(self.args.output_dir + alias)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4, sort_keys=True)


