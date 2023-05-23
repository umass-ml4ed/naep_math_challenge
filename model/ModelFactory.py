from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPTJForSequenceClassification
from transformers import PreTrainedModel
from model.EncoderDecoder import FlanT5encoder
from model.modeling_bert import BertForTokenClassificationMultiHead
class ModelFactory():

    @classmethod
    def produce_model_and_tokenizer(cls, cfg, num_labels, id2label, label2id,**kwargs):
        if "gptj" in cfg.lm:
            if cfg.multi_head or cfg.loss==1:
                raise 'Not Implement'
            model = GPTJForSequenceClassification.from_pretrained(cfg.lm, num_labels=num_labels, id2label = id2label, label2id = label2id, ignore_mismatched_sizes=True)
            tokenizer = AutoTokenizer.from_pretrained(cfg.lm, model_max_length=model.config.max_position_embeddings)
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
        elif 't5' in cfg.lm:
            if cfg.multi_head or cfg.loss==1:
                raise 'Not Implement'
            model = FlanT5encoder(cfg.lm, num_labels)
            tokenizer = AutoTokenizer.from_pretrained(cfg.lm)
        elif 'bert' in cfg.lm:
            tokenizer = AutoTokenizer.from_pretrained(cfg.lm)
            if cfg.multi_head:
                model = BertForTokenClassificationMultiHead.from_pretrained(cfg.lm, num_labels=num_labels, args=cfg)
            else:
                #model = AutoModelForSequenceClassification.from_pretrained(cfg.lm, num_labels=num_labels, id2label = id2label, label2id = label2id)
                model = BertForTokenClassificationMultiHead.from_pretrained(cfg.lm, num_labels=num_labels, args=cfg)
        elif 'mpnet' in cfg.lm:
            if cfg.multi_head or cfg.loss==1:
                raise 'Not Implement'
            tokenizer = AutoTokenizer.from_pretrained(cfg.lm)
            model = AutoModelForSequenceClassification.from_pretrained(cfg.lm, num_labels=num_labels, id2label = id2label, label2id = label2id)

        else:
            raise 'invalid lm, check the setting please'
        return (model, tokenizer)


class ModifiedModelThatCanTakeExtraInput():
    pass