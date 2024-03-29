from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPTJForSequenceClassification, LlamaTokenizer, LlamaForSequenceClassification, AutoModelForCausalLM
from transformers import PreTrainedModel
from model.EncoderDecoder import FlanT5encoder, T5EncoderModelClssification, T5EncoderDecoderModelClassification
from model.modeling_bert import BertForTokenClassificationMultiHead
import os

LLAMA_LOCAL_FILEPATH = "/media/animal_farm/llama_hf"
ALPACA_LOCAL_FILEPATH = "/media/animal_farm/alpaca"

class ModelFactory():

    @classmethod
    def produce_model_and_tokenizer(cls, cfg, num_labels, id2label, label2id,**kwargs):
        print('=======Loading model, location is', cfg.lm, '==============================')

        if "gptj" in cfg.lm:
            if cfg.multi_head or cfg.loss==1:
                raise 'Not Implement'
            model = GPTJForSequenceClassification.from_pretrained(cfg.lm, num_labels=num_labels, id2label = id2label, label2id = label2id, ignore_mismatched_sizes=True)
            tokenizer = AutoTokenizer.from_pretrained(cfg.lm, model_max_length=model.config.max_position_embeddings)
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
        elif 't5' in cfg.lm:
            cfg.num_label = num_labels
            if cfg.e2e:
                model = T5EncoderDecoderModelClassification.from_pretrained(cfg.lm, args=cfg)
            else:
                model = T5EncoderModelClssification.from_pretrained(cfg.lm, args=cfg)
            tokenizer = AutoTokenizer.from_pretrained(cfg.lm)
        elif 'bert' in cfg.lm or 'BERT' in cfg.lm:
            if cfg.multi_head:
                model = BertForTokenClassificationMultiHead.from_pretrained(cfg.lm, num_labels=num_labels, args=cfg)
            else:
                model = BertForTokenClassificationMultiHead.from_pretrained(cfg.lm, num_labels=num_labels, args=cfg)
            tokenizer = AutoTokenizer.from_pretrained(cfg.lm, model_max_length=model.config.max_position_embeddings)
        elif 'mpnet' in cfg.lm:
            if cfg.multi_head or cfg.loss==1:
                raise 'Not Implement'
            tokenizer = AutoTokenizer.from_pretrained(cfg.lm)
            model = AutoModelForSequenceClassification.from_pretrained(cfg.lm, num_labels=num_labels, id2label = id2label, label2id = label2id)
        elif "llama" in cfg.lm:
            if cfg.multi_head or cfg.loss==1:
                raise 'Not Implement'
            if cfg.prompting:
                llama_model = AutoModelForCausalLM.from_pretrained(LLAMA_LOCAL_FILEPATH)
            else:
                llama_model = LlamaForSequenceClassification.from_pretrained(LLAMA_LOCAL_FILEPATH)
            llama_tokenizer = LlamaTokenizer.from_pretrained(LLAMA_LOCAL_FILEPATH)
            llama_tokenizer.pad_token = llama_tokenizer.eos_token
            llama_tokenizer.padding_side = "left"
            print(llama_model)
            for param in llama_model.model.parameters():
                param.requires_grad = False
            return (llama_model, llama_tokenizer)
        elif "alpaca" in cfg.lm:
            if cfg.prompting:
                alpaca_model = AutoModelForCausalLM.from_pretrained(ALPACA_LOCAL_FILEPATH)
            else:
                alpaca_model = LlamaForSequenceClassification.from_pretrained(ALPACA_LOCAL_FILEPATH)
            alpaca_tokenizer = LlamaTokenizer.from_pretrained(ALPACA_LOCAL_FILEPATH)
            alpaca_tokenizer.pad_token = alpaca_tokenizer.eos_token
            alpaca_tokenizer.padding_side = "left"
            return (alpaca_model, alpaca_tokenizer)
        elif "gpt2" in cfg.lm:
            model = AutoModelForCausalLM.from_pretrained(cfg.lm)
            tokenizer = AutoTokenizer.from_pretrained(cfg.lm)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

        else:
            raise 'invalid lm, check the setting please'
        return (model, tokenizer)

