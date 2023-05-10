from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPTJForSequenceClassification
class ModelFactory():

    @classmethod
    def produce_model_and_tokenizer(cls, cfg, num_labels, id2label, label2id):
        if cfg.lm == "GPTJ":
            tokenizer = AutoTokenizer.from_pretrained("ydshieh/tiny-random-gptj-for-sequence-classification", num_labels=num_labels, id2label = id2label, label2id = label2id)
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
            model = GPTJForSequenceClassification.from_pretrained("ydshieh/tiny-random-gptj-for-sequence-classification", num_labels=num_labels, id2label = id2label, label2id = label2id, ignore_mismatched_sizes=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(cfg.lm)
            model = AutoModelForSequenceClassification.from_pretrained(cfg.lm, num_labels=num_labels, id2label = id2label, label2id = label2id)
        return (model, tokenizer)