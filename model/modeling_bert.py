from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import ( nn,
                                                     add_start_docstrings_to_model_forward,
                                                     add_code_sample_docstrings, BERT_INPUTS_DOCSTRING,
                                                     _CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION, TokenClassifierOutput,
                                                     _CONFIG_FOR_DOC, _SEQ_CLASS_EXPECTED_LOSS, _SEQ_CLASS_EXPECTED_OUTPUT,
                                                     Optional, Union, Tuple, CrossEntropyLoss, SequenceClassifierOutput)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from utils.utils import OLL2_loss
import torch
class BertForTokenClassificationMultiHead(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        #if 'args' in kwargs:
        #    config.update(kwargs['args'])
        self.num_labels = config.num_labels
        self.num_questions = kwargs['args'].num_questions
        self.args = kwargs['args']

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        #self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        #self.item_classifier = torch.nn.Parameter(torch.randn(self.num_questions, config.hidden_size, config.num_labels))
        #self.item_bias = torch.nn.Parameter(torch.randn(self.num_questions, config.num_labels))
        #self.classifier = nn.ParameterList([nn.Linear(config.hidden_size, config.num_labels) for i in range(self.num_questions)])
        if self.args.multi_head:
            self.classifier = nn.ModuleList([nn.Linear(config.hidden_size, config.num_labels) for i in range(self.num_questions)])
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()
        self.dist_matrix = [[0, 1, 2, 3, 4], [1, 0, 1, 2, 3], [2, 1, 0, 1, 2], [3, 2, 1, 0, 1], [4, 3, 2, 1, 0]]

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        question_ids = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[1]

        sequence_output = self.dropout(sequence_output)
        #classifier = self.item_classifier[question_ids]
        #bias = self.item_bias[question_ids]
        #logits = torch.sum(classifier * sequence_output.unsqueeze(-1), dim=1) + bias
        #logits = self.classifier(sequence_output)
        #classifier = self.classifier[question_ids]
        #logits = classifier(sequence_output)

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
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
