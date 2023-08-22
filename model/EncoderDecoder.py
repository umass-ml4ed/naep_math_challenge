import os
import torch
from torch import nn
from transformers import T5EncoderModel, T5ForConditionalGeneration
from typing import Optional
from transformers.models.t5.modeling_t5 import T5PreTrainedModel, T5Config, Seq2SeqLMOutput,\
    add_start_docstrings_to_model_forward, add_start_docstrings, \
    PARALLELIZE_DOCSTRING, copy, T5Stack, DEPARALLELIZE_DOCSTRING, T5_ENCODER_INPUTS_DOCSTRING, \
    BaseModelOutput, assert_device_map, get_device_map, _CONFIG_FOR_DOC, Union, replace_return_docstrings, Tuple
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, LeakyReLU
from model.modeling_bert import MeanBertPooler, TokenClassifierOutput2
from model.pooling import ClsPooler
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
        self.values = torch.tensor(list(range(1,self.num_labels + 1)))
        self.values = self.values.unsqueeze(1).T.to(self.device)
        self.dropout = nn.Dropout(classifier_dropout)

        #self.pooling = Pooling(config.hidden_size, pooling_mode=self.args.pooling) #['mean', 'max', 'cls']
        if self.args.pooling == 'mean':
            self.pooling = MeanBertPooler(config)
        else:
            self.pooling = ClsPooler(config)
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
        feature_ids= None,
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
        self.values = self.values.to(self.device)

        encoder_outputs = encoder_outputs[0]
        sequence_output = self.pooling(encoder_outputs, own_attention_mask)
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
            expectation_scores = torch.sum(softmax_scores * self.values, dim=1)
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
            expectation_scores=expectation_scores,
        )


class T5EncoderDecoderModelClassification(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, **kwargs):
        super().__init__(config)
        self.num_questions = kwargs['args'].num_questions
        self.args = kwargs['args']
        self.num_labels = self.args.num_label
        config.num_labels = self.num_labels
        if self.args.multi_head:
            self.classifier = nn.ModuleList([nn.Linear(config.hidden_size, config.num_labels) for i in range(self.num_questions)])
            if self.args.non_linear_head:
                self.classifier = nn.ModuleList(
                    [nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), LeakyReLU(), nn.Linear(config.hidden_size, config.num_labels)) for i in range(self.num_questions)])
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            if self.args.non_linear_head:
                self.classifier = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), LeakyReLU(), nn.Linear(config.hidden_size, config.num_labels))



    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            question_ids = None,
            **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        if len(decoder_input_ids.shape) == 1:
            decoder_input_ids = decoder_input_ids.unsqueeze(1)


        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism


        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)


        if self.args.multi_head:
            lm_logits = torch.stack([self.classifier[qid](sequence_output[i]) for i, qid in enumerate(question_ids)])
        else:
            lm_logits = self.classifier(sequence_output)
        #lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output
        lm_logits = lm_logits.squeeze()
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )



