import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
from utils.var import LLAMA_LOCAL_FILEPATH, ALPACA_LOCAL_FILEPATH


def generate_completion(prompt, model, tokenizer, device):
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    input_length = inputs.size()[1]
    output = model.generate(inputs, max_length=input_length + 50, num_return_sequences=1, do_sample=True)
    completion = tokenizer.decode(output[:, input_length:][0], skip_special_tokens=True)
    return completion

def get_model_tokenizer_from_str(model_name):
    print("Loading the model from the weights, this could take some time.")
    if model_name == "llama":
        llama_model = LlamaForCausalLM.from_pretrained(LLAMA_LOCAL_FILEPATH)
        llama_tokenizer = LlamaTokenizer.from_pretrained(LLAMA_LOCAL_FILEPATH)
        return (llama_model, llama_tokenizer)
    elif model_name == "alpaca":
        alpaca_model = AutoModelForCausalLM.from_pretrained(ALPACA_LOCAL_FILEPATH)
        alpaca_tokenizer = AutoTokenizer.from_pretrained(ALPACA_LOCAL_FILEPATH)
        return (alpaca_model, alpaca_tokenizer)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return (model, tokenizer)
        

def main():
    model_name = input("Enter the model name (llama or alpaca or gpt2): ")
    (model, tokenizer) = get_model_tokenizer_from_str(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Sending model to device this might take a while.")
    model.to(device)

    while True:
        prompt = input("Enter a prompt (or 'q' to quit): ")
        if prompt.lower() == 'q':
            break
        
        completion = generate_completion(prompt, model, tokenizer, device)
        print("Completion:", completion)
        print()

if __name__ == '__main__':
    main()