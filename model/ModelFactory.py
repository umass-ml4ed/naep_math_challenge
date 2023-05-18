from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPTJForSequenceClassification
from GPTJWrapper import GPTJWrapper
from model.EncoderDecoder import FlanT5encoder
class ModelFactory():

    @classmethod
    def produce_model_and_tokenizer(cls, cfg, num_labels, id2label, label2id):
        if "gptj" in cfg.lm:
            model = GPTJForSequenceClassification.from_pretrained(cfg.lm, num_labels=num_labels, id2label = id2label, label2id = label2id, ignore_mismatched_sizes=True)
            model = GPTJWrapper(model)
            tokenizer = AutoTokenizer.from_pretrained(cfg.lm, model_max_length=model.config.max_position_embeddings)
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
        elif 't5' in cfg.lm:
            model = FlanT5encoder(cfg.lm, num_labels)
            tokenizer = AutoTokenizer.from_pretrained(cfg.lm)
        else:
            tokenizer = AutoTokenizer.from_pretrained(cfg.lm)
            model = AutoModelForSequenceClassification.from_pretrained(cfg.lm, num_labels=num_labels, id2label = id2label, label2id = label2id)
        return (model, tokenizer)