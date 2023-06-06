from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
import numpy as np
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import torch
from torch import nn
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
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
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import (
    is_sagemaker_mp_enabled,
)

from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    can_return_loss,
    find_labels,
    get_full_repo_name,
    is_accelerate_available,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_tpu_available,
    logging,
)
from collections import defaultdict
from model.dataset import rerange_data,rerange_examples
from model.examplesRetriever import KNNRetriever
if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl
from packaging import version
import math
import sys
import contextlib
import functools
import glob
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from collections.abc import Mapping
from distutils.util import strtobool
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.utils import ModelOutput
from tqdm.auto import tqdm
skip_first_batches = None




# Integrations must be imported before ML frameworks:
# isort: off
from transformers.integrations import (
    default_hp_search_backend,
    get_reporting_integration_callbacks,
    hp_params,
    is_fairscale_available,
    is_optuna_available,
    is_ray_tune_available,
    is_sigopt_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
    run_hp_search_sigopt,
    run_hp_search_wandb,
)

# isort: on

import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import Repository, create_repo
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    #from smdistributed.modelparallel import __version__ as SMP_VERSION

    #IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

from transformers import __version__
from transformers.configuration_utils import PretrainedConfig
#from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from model.dataset import DataCollatorWithPadding
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.optimization import Adafactor, get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_10, is_torch_less_than_1_11
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)

from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    can_return_loss,
    find_labels,
    get_full_repo_name,
    is_accelerate_available,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_tpu_available,
    logging,
)

from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from transformers.utils import (
    CONFIG_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    can_return_loss,
    find_labels,
    get_full_repo_name,
    is_accelerate_available,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_tpu_available,
    logging,
)
from transformers.utils.generic import ContextManagers
logger = logging.get_logger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
class MyTrainer(Trainer):
    def __init__(self, args, device):

        if 'saved_models' in args.lm:
            #model_path = os.path.abspath(args.lm)
            args.lm = os.path.abspath(args.lm)

        self.extra_info = {'label2': [var.EVAL_LABEL, var.EST_SCORE]}
        self.device = device
        self.args = args
        self.input_args = args
        self.all_metrics = {}
        # load question information
        with open('question.json', 'r') as f:
            question_info = json.load(f)
        self.question_info = question_info

        """
        1. Prepare label and model information
        """

        training_dataset = pd.read_csv(args.train_path)
        training_dataset = prepare_dataset(training_dataset, args)

        if args.task != 'all':
            if args.task not in var.QUESTION_LIST:
                args.task = var.NAME_TO_QUESTION[args.task]
            training_dataset = training_dataset[training_dataset['qid'] == args.task]

        self.question2id = {value:i for i, value in enumerate(var.QUESTION_LIST)}
        self.num_questions = len(self.question2id)
        self.args.num_questions = self.num_questions
        model, tokenizer = self.prepare_model(training_dataset)

        """
        2. Prepare dataloader
        """
        dataset_dict = self.prepare_dataloader(training_dataset)

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
            report_to="wandb",
            seed=args.seed,
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


    def prepare_model(self, training_dataset):
        args = self.args
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
        if args.num_label:
            num_label = args.num_label
        self.num_label = num_label

        (model, tokenizer) = mf.produce_model_and_tokenizer(args, num_label, id2label, label2id)

        self.model = model
        self.tokenizer = tokenizer
        return model, tokenizer

    def prepare_dataloader(self, training_dataset):
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

        if args.reduce:
            # if 'json' in args.reduce_path:
            #     with open(args.reduce_path, "r") as file:
            #         reduce_list = json.load(file)
            #     train = training_dataset[training_dataset['id'].isin(reduce_list['train'])]
            #     val = training_dataset[training_dataset['id'].isin(reduce_list['val'])]
            #     test = training_dataset[training_dataset['id'].isin(reduce_list['test'])]
            # else:
            train, val, test = split_data_into_TrainValTest(training_dataset, args=args)
            iddf = pd.read_csv(args.reduce_path)
            reduce_list = iddf['id'].tolist()
            train = train[train['id'].isin(reduce_list)]
            val = val[val['id'].isin(reduce_list)]
            test = test[test['id'].isin(reduce_list)]
        elif args.split:
            train, val, test = split_data_into_TrainValTest(training_dataset, args = args)
        elif args.eval_only:
            train = training_dataset
            val  = prepare_dataset(pd.read_csv(args.test_path), args)
            test = val
        else:
            raise 'not define how to split the data'
        rerange_data(train,args)
        rerange_data(val, args)
        rerange_data(test,args)
        _, examples = rerange_examples(train)

        if args.debug:
            if args.prompting:
                train = train.sample(n=50, replace=False)
                test, val = train, train
            elif args.analysis:
                qdf_list = []
                for key, qdf in list(train.groupby('qid')):
                    qdf = qdf.sample(n=1000, replace=False)
                    qdf_list.append(qdf)
                train = pd.concat(qdf_list)
                #val = val.sample(n=100, replace=False)
                #test = test.sample(n=100, replace=False)
            else:
                train = train.sample(n=1000, replace=False)
                test, val = train, train

        utils.safe_makedirs(args.save_model_dir)
        test.to_csv(args.save_model_dir + 'test.csv')


        if args.retriever.name=='knn':
            retriever = KNNRetriever(args, num_label=self.num_label, id2label=self.id2label, label2id=self.label2id)
            if not args.analysis:
                retriever.create_examples_embedding(train)
        elif args.same:
            retriever = KNNRetriever(args, model = self.model,
            pooling='bert', num_label=self.num_label, id2label=self.id2label, label2id=self.label2id)
        else:
            retriever = None
        self.retriever = retriever


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
                                     labels_dict = self.label2id, question_dict = self.question2id, question_info=self.question_info)
            val_dataset = IncontextDataset(tokenizer=tokenizer, data=val, args=args,
                                   labels_dict = self.label2id, example=train,
                                   question_dict = self.question2id, retriever=retriever, eval=True,  question_info=self.question_info)
            test_dataset = IncontextDataset(tokenizer=tokenizer, data=test,args=args,
                                    labels_dict = self.label2id, example=train,
                                    question_dict = self.question2id, retriever=retriever, eval=True, question_info=self.question_info)
            self.dataset_dict = datasets.DatasetDict({'train': train_dataset, 'val': val_dataset, 'test': test_dataset})
            question_wise_test = {key: IncontextDataset(tokenizer=tokenizer, data=item, args=args,
                                   labels_dict = self.label2id, example=train[train['qid'] == key],
                                   question_dict = self.question2id) for key, item in question_wise_test}
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


    def save_metrics(self, metrics, alias=''):
        el.log(metrics)
        path = os.path.join(self.args.output_dir + alias + 'metrics.json')
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4, sort_keys=True)
        q = pd.DataFrame.from_dict(self.question_info).T
        q = q[['name','type']]
        m = pd.DataFrame(metrics, index=[0]).T
        #m = pd.DataFrame.from_dict(metrics).T
        m = m.join(q)
        m.to_csv(self.args.output_dir + alias + 'metrics.csv')


    def save_embedding(self):
        retriever = self.retriever
        raise 'not fihish'

    def predict_to_save(self, data:Dataset, alias=''):
        """
        :param data: the data to evaluate
        :return: the dataframe with an extra column named "predict"
        """
        #todo need to check label 1
        predicts = self.predict(data)
        data_df = data.to_pandas()
        predictions = predicts.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        pred = np.argmax(predictions, axis=1)
        pred = list(map(lambda x: self.id2label[x], list(pred)))
        data_df['predict'] = pred
        all_metrics = self.itemwise_score(data_df)
        #calculate itemwise information
        #data_df = data_df[['id', 'qid', 'text', 'predict', 'label_str', 'label1', 'label']]
        data_df.to_csv(self.args.output_dir + alias + '_predict.csv',index=False)
        self.save_metrics(all_metrics, alias)
        self.save_metrics(self.all_metrics, 'epoch')
        return data_df

    def prompting_predict_to_save(self, data, alias=''):
        prompting_path = 'conf/prompting.txt'
        score_list = ['1', '1A', '1B', '2', '2A', '2B', '3']
        #todo need to write in parallel way to save time
        def generate_completion(prompt, model, tokenizer, device):
            inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
            input_length = inputs.size()[1]
            output = model.generate(inputs, max_length=input_length + 50, num_return_sequences=1, do_sample=True)
            completion = tokenizer.decode(output[:, input_length:][0], skip_special_tokens=True)
            return completion

        def extract_information(string, pattern):
            #pattern = rf"<{pattern}>(.*?)</{pattern}>"
            #match = re.search(pattern, string)
            #if match:
            #    return match.group(1)
            string = string.split("<" +pattern + ">")[1]
            string = string.split("</" + pattern + ">")[0]
            return string

        def extrat_score_out(string):
            for s in score_list:
                if s in string:
                    return s
            return '1'
        id2simplelabel = {k: int(re.sub(r"\D", "", k)) for k in score_list}


        # dataloader = self.get_test_dataloader(data)
        # for step, inputs in enumerate(dataloader):
        with open(prompting_path, 'r') as file:
            prompts = file.read()
        data_df = data.to_pandas()
        predict = []
        full_predict = []
        for qid, qdf in tqdm(list(data_df.groupby('qid')), position=0):
            prompt = extract_information(prompts, qid)
            for text in tqdm(zip(qdf['text1'].values.tolist(), qdf['text2'].values.tolist()), total=len(qdf), position=0):
                text1, text2 = text
                prompt = prompt.replace('TEXT1', text1)
                prompt = prompt.replace('TEXT2', text2)
                result = generate_completion(prompt, self.model, self.tokenizer, self.device)
                score = extrat_score_out(result)
                full_predict.append(result)
                predict.append(score)
        data_df['predict'] = predict
        data_df['full_predict'] = full_predict
        #calculate itemwise information
        data_df = data_df[['id', 'qid', 'text', 'predict', 'label_str','full_predict']]
        data_df.to_csv(self.args.output_dir + alias + 'test_predict.csv')
        preds = np.array(list(map(lambda x: id2simplelabel[x], predict)))
        labels = np.array(list(map(lambda x: id2simplelabel[x], data_df['label_str'].values.tolist())))
        metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=labels))
        metrics = denumpify_detensorize(metrics)
        self.log(metrics)
        self.save_metrics(metrics, alias)

    def itemwise_score(self, data_df, prefix = ''):
        try:
            epoch = str(int(self.state.epoch))
        except:
            epoch = ''
        all_metrics = {}
        for qid, qdf in list(data_df.groupby('qid')):
            qdf['predict'+epoch]  = qdf['predict'+epoch].astype('int')
            qdf['label'] = qdf['label'].astype('int')
            preds = np.array(qdf['predict'+epoch].values.tolist())
            labels = np.array(qdf['label'].values.tolist())
            if self.input_args.label == 2:
                other = {}
                other[var.EVAL_LABEL] = np.array(qdf[var.EVAL_LABEL].values.tolist())
                other[var.EST_SCORE] = np.array(qdf[var.EST_SCORE].values.tolist())
                metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=labels, inputs= other), id=False)
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=labels))
            metrics = denumpify_detensorize(metrics)
            qid = var.QUESTION_TO_NAME[qid]
            all_score = defaultdict(int)
            #prefix += epoch
            for key in list(metrics.keys()):
                if not key.startswith(f"{qid}_"):
                    value = metrics.pop(key)
                    if prefix !='':
                        #metrics[f"{epoch}_{prefix}_{qid}_{key}"] = value
                        metrics[f"{prefix}_{qid}_{key}"] = value
                        all_score[f"{prefix}_{key}"] += value
                    else:
                        #metrics[f"{epoch}_{qid}_{key}"] = value
                        metrics[f"{qid}_{key}"] = value
                        all_score[f"{key}"] += value
            all_metrics.update(metrics)
            all_metrics.update(all_score)
        return all_metrics
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        if self.input_args.label == 2:
            others = {}
            for key in self.extra_info['label2']:
                others.update({key:inputs.pop(key)})
        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.input_args.label == 2:
            outputs = (outputs, others)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        other_info = None
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()
                    if self.input_args.label==2:
                        other_info = outputs[1]
                        outputs = outputs[0]
                    if isinstance(outputs, dict):
                        #logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                        logits = outputs['logits']
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels, other_info)

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        other_hosts = defaultdict(lambda : None)
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels, other_info = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if other_info is not None:
                for key, value in other_info.items():
                    value = self._pad_across_processes(value)
                    value = self._nested_gather(value)
                    other_hosts[key] = value if other_hosts[key] is None else nested_concat(other_hosts[key], value,
                                                                                   padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
        all_others = defaultdict(lambda: None)
        if len(other_hosts) != 0:
            for key, value in other_hosts.items():
                other = nested_numpify(other_hosts[key])
                all_others[key] = other if all_others[key] is None else np.concatenate(all_others[key], other, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)
        if len(all_others) != 0:
            for key, value in all_others.items():
                all_others[key] = nested_truncate(value, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                if self.input_args.label == 2:
                    metrics = self.compute_metrics(EvalPrediction(predictions=all_preds,
                                                                  label_ids=all_labels, inputs=all_others))
                else:
                    metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))


        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
            else:
                epoch = str(int(self.state.epoch))
                outputs = self.evaluate(ignore_keys=ignore_keys_for_eval, full=True)
                data = self.eval_dataset
                data_df = data.to_pandas()
                predictions = outputs.predictions
                if isinstance(predictions, tuple):
                    predictions = predictions[0]
                pred = np.argmax(predictions, axis=1)
                pred = list(map(lambda x: self.id2label[x], list(pred)))
                data_df['predict'+epoch] = pred
                metrics = self.itemwise_score(data_df, prefix= 'eval')

                data = self.test_dataset
                outputs = self.evaluate(data, ignore_keys=ignore_keys_for_eval, full=True, metric_key_prefix='test')
                data_df = data.to_pandas()
                predictions = outputs.predictions
                if isinstance(predictions, tuple):
                    predictions = predictions[0]
                pred = np.argmax(predictions, axis=1)
                pred = list(map(lambda x: self.id2label[x], list(pred)))
                data_df['predict'+epoch] = pred
                metrics2 = self.itemwise_score(data_df, prefix= 'test')
                metrics.update(metrics2)
            self.log(metrics)
            self.all_metrics[epoch] = metrics
            self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval", full = False,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(eval_dataloader, description="Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,)

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        if full:
            return output

        return output.metrics

    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{str(int(self.state.epoch))}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)
        if self.deepspeed:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_16bit_weights_on_model_save` is True
            self.deepspeed.save_checkpoint(output_dir)

        # Save optimizer and scheduler
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            self.optimizer.consolidate_state_dict()

        if self.fsdp:
            # FSDP has a different interface for saving optimizer states.
            # Needs to be called on all ranks to gather all states.
            # full_optim_state_dict will be deprecated after Pytorch 2.2!
            full_osd = self.model.__class__.full_optim_state_dict(self.model, self.optimizer)

        if is_torch_tpu_available():
            xm.rendezvous("saving_optimizer_states")
            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
        elif is_sagemaker_mp_enabled():
            opt_state_dict = self.optimizer.local_state_dict(gather_if_shard=False)
            smp.barrier()
            if smp.rdp_rank() == 0 or smp.state.cfg.shard_optimizer_state:
                smp.save(
                    opt_state_dict,
                    os.path.join(output_dir, OPTIMIZER_NAME),
                    partial=True,
                    v3=smp.state.cfg.shard_optimizer_state,
                )
            if self.args.should_save:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
                if self.do_grad_scaling:
                    torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
        elif self.args.should_save and not self.deepspeed:
            # deepspeed.save_checkpoint above saves model/optim/sched
            if self.fsdp:
                torch.save(full_osd, os.path.join(output_dir, OPTIMIZER_NAME))
            else:
                torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))

            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)
            if self.do_grad_scaling:
                torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        if is_torch_tpu_available():
            rng_states["xla"] = xm.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)

        if self.args.world_size <= 1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)




