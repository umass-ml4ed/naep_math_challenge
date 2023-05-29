import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM

LLAMA_LOCAL_FILEPATH = "/media/animal_farm/llama_hf"
ALPACA_LOCAL_FILEPATH = "/media/animal_farm/alpaca"

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
    # model_name = input("Enter the model name (llama or alpaca or gpt2): ")
    model_name = "alpaca"
    (model, tokenizer) = get_model_tokenizer_from_str(model_name)

    gpu_num =2 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if gpu_num is not None:
        cuda_str = f'cuda:{gpu_num}'
        device = torch.device(cuda_str) if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)

    print("Sending model to device this might take a while.")
    model.to(device)

    prompt_file = "./prompts/input_template.txt"  # Path to the file containing the prompt

    should_reload = True

    while True:
        if should_reload:
            print(f"Reading prompt from file {prompt_file}.")
            with open(prompt_file, "r") as file:
                prompt = file.read()

            completion = generate_completion(prompt, model, tokenizer, device)
            print("Completion:", completion)
            print()
            should_reload = False

        user_input = input("Enter 'r' to reload prompt from file or 'q' to quit: ")
        if user_input.lower() == 'q':
            break
        elif user_input.lower() == 'r':
            should_reload = True
        else:
            print("Invalid input. Please try again.")



if __name__ == '__main__':
    main()