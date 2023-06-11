from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import ( nn,
                                                     add_start_docstrings_to_model_forward,
                                                     add_code_sample_docstrings, BERT_INPUTS_DOCSTRING,
                                                     _CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION, TokenClassifierOutput,
                                                     _CONFIG_FOR_DOC, _SEQ_CLASS_EXPECTED_LOSS, _SEQ_CLASS_EXPECTED_OUTPUT,
                                                     Optional, Union, Tuple, CrossEntropyLoss, SequenceClassifierOutput)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, LeakyReLU
from utils.utils import OLL2_loss
import torch
from transformers.utils import ModelOutput
from model.pooling import Pooling, MeanBertPooler
class BertForTokenClassificationMultiHead(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler", r"pooling"]

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_questions = kwargs['args'].num_questions
        self.args = kwargs['args']
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.values = torch.tensor(list(range(1,self.num_labels + 1)))
        self.values = self.values.unsqueeze(1).T.to(self.device)
        self.dropout = nn.Dropout(classifier_dropout)
        #self.pooling = Pooling(config.hidden_size, pooling_mode=self.args.pooling) #['mean', 'max', 'cls']
        if self.args.pooling == 'mean':
            self.pooling = MeanBertPooler(config)
        if self.args.fair_train:
            self.feature_n = self.args.feature_n
            self.feature_embedding = nn.Embedding(self.feature_n, config.hidden_size)

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
        own_attention_mask: Optional[torch.Tensor]= None,
        feature_ids = None,
        **kwargs
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self.values = self.values.to(self.device)

        if self.args.freeze:
            with torch.no_grad():
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
        else:
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

        if self.args.pooling == 'mean':
            sequence_output = self.pooling(outputs[0], own_attention_mask)
        else:
            sequence_output = outputs[1]
        sequence_output = self.dropout(sequence_output)

        if self.args.fair_train and feature_ids is not None:
            feature_bias = self.feature_embedding(feature_ids)
            sequence_output = sequence_output + feature_bias



        if self.args.multi_head:
            logits = torch.stack([self.classifier[qid](sequence_output[i]) for i, qid in enumerate(question_ids)])
        else:
            logits = self.classifier(sequence_output)

        loss = None
        expectation_scores = None
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

        return TokenClassifierOutput2(
            loss=loss,
            logits=logits,
            hidden_states=sequence_output,
            attentions=attention_mask,
            pooler_output = sequence_output,
            expectation_scores= expectation_scores
        )
        #return outputs

class TokenClassifierOutput2(ModelOutput):
    """
    Base class for outputs of token classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    pooler_output = None,
    expectation_score = None
