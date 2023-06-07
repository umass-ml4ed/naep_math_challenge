import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
from transformers import AutoConfig
from train import MyTrainer
import pandas as pd
import re
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time


LLAMA_LOCAL_FILEPATH = "/media/animal_farm/llama_hf"
ALPACA_LOCAL_FILEPATH = "/media/animal_farm/alpaca"

def generate_completion(prompt, model, tokenizer, device):
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    input_length = inputs.size()[1]
    output = model.generate(inputs, max_length=input_length + 50, num_return_sequences=1, do_sample=True)
    completion = tokenizer.decode(output[:, input_length:][0], skip_special_tokens=True)
    return completion

def batch_generate_completions(prompts, model, tokenizer, device):
    # There's a silly bug here somewhere
    inputs = tokenizer(prompts, is_split_into_words=False, return_tensors='pt', padding=True, truncation=False, return_token_type_ids=False).to(device)
    input_lengths = inputs['input_ids'].shape[1]
    outputs = model.generate(**inputs, max_length=input_lengths + 50, num_return_sequences=1, do_sample=True)
    outputs = outputs[:, input_lengths:]
    completions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return completions

def get_model_tokenizer_from_str(model_name):
    print("Loading the model from the weights, this could take some time.")
    if model_name == "llama":
        llama_model = LlamaForCausalLM.from_pretrained(LLAMA_LOCAL_FILEPATH)
        llama_tokenizer = LlamaTokenizer.from_pretrained(LLAMA_LOCAL_FILEPATH)
        llama_tokenizer.padding_side = "left"
        return (llama_model, llama_tokenizer)
    elif model_name == "alpaca":
        config = AutoConfig.from_pretrained(ALPACA_LOCAL_FILEPATH)
        # print(config.max_position_embeddings)
        alpaca_model = AutoModelForCausalLM.from_pretrained(ALPACA_LOCAL_FILEPATH)
        alpaca_tokenizer = AutoTokenizer.from_pretrained(ALPACA_LOCAL_FILEPATH)
        alpaca_tokenizer.padding_side = "left" # allows for batch
        alpaca_tokenizer.pad_token = alpaca_tokenizer.eos_token
        alpaca_tokenizer.model_max_length = 2048
        return (alpaca_model, alpaca_tokenizer)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return (model, tokenizer)

def prompt_tuning_loop():
     # model_name = input("Enter the model name (llama or alpaca or gpt2): ")
    model_name = "alpaca"
    (model, tokenizer) = get_model_tokenizer_from_str(model_name)

    gpu_num = 2 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if gpu_num is not None:
        cuda_str = f'cuda:{gpu_num}'
        device = torch.device(cuda_str) if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)

    print("Sending model to device this might take a while.")
    model.to(device)

    prompt_file = "./conf/prompting_4.txt"  # Path to the file containing the prompt

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

# TODO handle 2017 cases (needed when using other folds)
selection_mapping = {"TRUE FALSE": "Selection A", "FALSE TRUE": "Selection B", "FALSE FALSE": "Selection None"}
score_converstion = {"5":"3", "4":"2A", "3":"2B", "2":"1A", "1":"1B"}

def main():
    # Create the prompts and extract labels from dataset
    total_tic = time.time()
    with open("./conf/prompting_4.txt", 'r') as file:
        template = file.read()
    df = pd.read_csv("data/train_VH271613.csv")
    test_fold_df = df.loc[df['fold'] == 9] # Only test fold
    starting_sample = test_fold_df # .iloc[:20]
    prompts = []
    selections = []
    explanations = []
    for index, row in starting_sample.iterrows():
        prompt = "" + template
        selection = row["partB_response_val"]
        select_string = selection_mapping[selection]
        explanation = row["predict_from"]
        selections.append(select_string)
        explanations.append(explanation)
        prompt = prompt.replace("{{student_selection}}", select_string)
        prompt = prompt.replace("{{answer_explanation}}", explanation)
        prompts.append(prompt)


    # Query the model for responses
    model_name = "alpaca"
    (model, tokenizer) = get_model_tokenizer_from_str(model_name)

    gpu_num = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if gpu_num is not None:
        cuda_str = f'cuda:{gpu_num}'
        device = torch.device(cuda_str) if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)

    print("Sending model to device this might take a while.")
    model.to(device)

    
    preds = []
    score_pattern = r"Score \s*([A-Za-z0-9]+):"
    print("Generating completions")

    # TODO link to a config
    batch_size = 10  # Set your desired batch size


    completions = []

    batched_prompts = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]
    for batch in tqdm(batched_prompts):
        batch_completions = batch_generate_completions(batch, model, tokenizer, device)
        completions.extend(batch_completions)

    for completion in completions:
        match = re.search(score_pattern, completion)
        score_value = "-1"
        if match:
            score_value = match.group(1)
            if score_value in score_converstion:
                score_value = score_converstion[score_value]
        preds.append(score_value)
    relaxed_preds = [item[0] if item != '-1' else '-1' for item in preds]
    # print(preds)

    hard_labels = starting_sample["label"]
    soft_labels = [str(s) for s in starting_sample["score_to_predict"]]
    # print(hard_labels)
    new_df = pd.DataFrame({
        'Part B Selection': selections,
        'Part B Explanation': explanations,
        'Completion': completions,
        'Predicted Score': preds,
        'Actual Score': hard_labels,  # Include an existing column from 'df' if needed
        'Predicted Relaxed Score': relaxed_preds,
        'Relaxed Score': soft_labels,
        # 'Full Prompt': prompts
    })

    # Write the new DataFrame to a CSV file
    new_df.to_csv('batch_prompt_4.csv', index=False)

    # calcuate and report accuracy/kappa for relaxed and unrelaxed
    print("###Metrics for Strict###")
    print(f"Accuracy: {accuracy_score(hard_labels, preds)}")
    print(f"Kappa: {cohen_kappa_score(hard_labels, preds)}")

    print("###Metrics for Relax###")
    print(f"Accuracy: {accuracy_score(soft_labels, relaxed_preds)}")
    print(f"Kappa: {cohen_kappa_score(soft_labels, relaxed_preds)}")
    total_toc = time.time()
    print("Inference on ", len(completions), " examples, total time:", total_toc - total_tic, "seconds.")

if __name__ == '__main__':
    main()