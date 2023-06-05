import os
import torch
from torch import nn
from transformers import T5EncoderModel
from typing import Optional
from transformers.models.t5.modeling_t5 import T5PreTrainedModel, T5Config, \
    add_start_docstrings_to_model_forward, add_start_docstrings, \
    PARALLELIZE_DOCSTRING, copy, T5Stack, DEPARALLELIZE_DOCSTRING, T5_ENCODER_INPUTS_DOCSTRING, \
    BaseModelOutput, assert_device_map, get_device_map, _CONFIG_FOR_DOC, Union, replace_return_docstrings, Tuple
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, LeakyReLU
from model.modeling_bert import MeanBertPooler, TokenClassifierOutput2
from utils.utils import OLL2_loss


class FlanT5encoder(nn.Module):
    def __init__(self, lm, num_label):
        super().__init__()
        self.flan_t5_encoder = T5EncoderModel.from_pretrained(lm)
        self.linear = nn.Linear(1024, num_label)

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.FloatTensor] = None, ):
        hidden_states = self.flan_t5_encoder(input_ids, attention_mask)
        output = torch.div(
            torch.sum(torch.mul(hidden_states.last_hidden_state, attention_mask.unsqueeze(-1).float()), dim=1),
            torch.sum(attention_mask, dim=-1).unsqueeze(-1).float())
        logits = self.linear(output)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        return loss, logits

class T5EncoderModelClssification(T5EncoderModel):
    _keys_to_ignore_on_load_missing = [r"encoder.embed_tokens.weight"]

    def __init__(self, config: T5Config, **kwargs):
        super().__init__(config)
        #self.model = T5EncoderModel(config)
        self.num_questions = kwargs['args'].num_questions

        self.args = kwargs['args']
        self.num_labels = self.args.num_label
        classifier_dropout = 0.1
        self.config.classifier_dropout = classifier_dropout
        self.config.num_labels = self.num_labels
        self.dropout = nn.Dropout(classifier_dropout)

        #self.pooling = Pooling(config.hidden_size, pooling_mode=self.args.pooling) #['mean', 'max', 'cls']
        #if self.args.pooling == 'mean':
        self.pooling = MeanBertPooler(config)
        if self.args.multi_head:
            self.classifier = nn.ModuleList([nn.Linear(config.hidden_size, config.num_labels) for i in range(self.num_questions)])
            if self.args.non_linear_head:
                self.classifier = nn.ModuleList(
                    [nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), LeakyReLU(), nn.Linear(config.hidden_size, config.num_labels)) for i in range(self.num_questions)])
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            if self.args.non_linear_head:
                self.classifier = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), LeakyReLU(), nn.Linear(config.hidden_size, config.num_labels))
        # Initialize weights and apply final processing
        self.post_init()
        self.dist_matrix = [[0, 1, 2, 3, 4], [1, 0, 1, 2, 3], [2, 1, 0, 1, 2], [3, 2, 1, 0, 1], [4, 3, 2, 1, 0]]

    @add_start_docstrings_to_model_forward(T5_ENCODER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        question_ids=None,
        labels: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        own_attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, T5EncoderModel

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = T5EncoderModel.from_pretrained("t5-small")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_outputs = encoder_outputs[0]
        sequence_output = self.pooling(encoder_outputs, own_attention_mask)
        if self.args.multi_head:
            logits = torch.stack([self.classifier[qid](sequence_output[i]) for i, qid in enumerate(question_ids)])
        else:
            logits = self.classifier(sequence_output)


        loss = None
        if self.args.loss == 2: # 'regression':
            #max_scores = torch.argmax(scores, dim=1)
            m = nn.Softmax(dim=1)
            softmax_scores = m(logits)
            expectation_scores = torch.sum(softmax_scores * self.values,dim=1)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            if labels is not None:
                if self.args.loss is None:
                    if self.num_labels == 1:
                        self.args.loss = 2 # "regression"
                    elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.args.loss = 0 # "single_label_classification"
                    else:
                        self.args.loss = 3 # "multi_label_classification"

                if self.args.loss == 2:  #"regression":
                    loss_fct = MSELoss()
                    max_scores = expectation_scores.type(torch.cuda.FloatTensor)
                    labels = labels.type(torch.cuda.FloatTensor)
                    if self.num_labels == 1:
                        loss = loss_fct(max_scores.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(max_scores, labels)
                elif self.args.loss == 0: #"single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                elif self.args.loss == 3: #"multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)
                elif self.args.loss == 1: #"oll2":
                    loss_fct = OLL2_loss
                    loss = loss_fct(self.num_labels, self.dist_matrix, labels.view(-1),
                                    logits.view(-1, self.num_labels))
        if not return_dict:
            output = (logits,) + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput2(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs,
            attentions=attention_mask,
            pooler_output = sequence_output,
        )



