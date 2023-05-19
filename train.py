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
if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl
from packaging import version

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    #from smdistributed.modelparallel import __version__ as SMP_VERSION

    #IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

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
logger = logging.get_logger(__name__)



from model.EncoderDecoder import FlanT5encoder
class MyTrainer(Trainer):
    def __init__(self, args, device):

        self.extra_info = {'label2': [var.EVAL_LABEL, var.EST_SCORE]}
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

        training_dataset = pd.read_csv(args.train_path)
        training_dataset = prepare_dataset(training_dataset, args)
        if args.task != 'all':
            if args.task not in var.QUESTION_LIST:
                args.task = var.NAME_TO_QUESTION[args.task]
            training_dataset = training_dataset[training_dataset['qid'] == args.task]

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
        self.num_label = num_label


        #todo could apply other architecture: encoder_decoder, multi-classfication head
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
        if args.split:
            train, val, test = split_data_into_TrainValTest(training_dataset, args = args)
        elif args.eval_only:
            train = training_dataset
            val  = prepare_dataset(pd.read_csv(args.test_path), args)
            test = val
        else:
            raise 'not define how to split the data'
        # if args.task != 'all':
        #     if args.task not in var.QUESTION_LIST:
        #         args.task = var.NAME_TO_QUESTION[args.task]
        #     train = train[train['qid'] == args.task]
        #     val = val[val['qid'] == args.task]
        #     test = test[test['qid']==args.task]
        if args.debug:
            train, val, test = train[:3000], val[:1000], test[:1000]

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
        predictions = predicts.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        pred = np.argmax(predictions, axis=1)
        pred = list(map(lambda x: self.id2label[x], list(pred)))
        data_df['predict'] = pred
        data_df = data_df[['qid', 'text', 'predict', 'label_str']]
        data_df.to_csv(self.args.output_dir + 'test_predict.csv')
        return data_df

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        #modifed for label = 2
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
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
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




