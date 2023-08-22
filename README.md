# naep_math_chanllenge

## How to run

### 0. install all requirement 
### 1. Save dataset in **/data**  folder
### 2. run **preprocessing.py**
> python preprocessing.py 
### 3. train the model with specific setting 
> #one example here: 
> 
> #run training with gpt2, in context learning setting 
> with 2 examples and closed form student solutions
> 
> python main.py lm=gpt2 task=all in_context=True closed_form=True examples=True  n_examples=2

