import os
import utils.var as var
def safe_makedirs(path_):
    if not os.path.exists(path_):
        try:
            os.makedirs(path_)
        except FileExistsError:
            pass

def prepare_dataset(data, args):
    data['label'] = data['label'].astype(str)
    data['label'] = data['label'].replace(
        {'1.0': '1', '2.0': '2', '3.0': '3', 1.0: '1', 2.0: '2', 3.0: '3', 1: '1', 2: '2', 3: '3'})
    # unify labels' names
    training_dataset = data.rename(columns=var.COLS_RENAME)
    if args.label == 0:
        """
        For label = 0, we use "score_to_predict" column as labels
        """
        training_dataset['label'] = training_dataset[var.LABEL0]
    elif args.label == 1:
        pass
    else:
        raise 'no definition'

    if args.base:  # the basic classification problem
        """
        The base case: 
        input: sutdent response 
        output: label 

        BASE_COLS = ['qid', 'label','text']
        Get corresponding information and rename the column 
        """
        data = data[var.BASE_COLS]
    elif args.in_context:
        useful_cols = var.BASE_COLS
        if args.closed_form:
            useful_cols += [var.CONTEXT_ALL]
        data = data[useful_cols]

    else:
        raise 'No task information defined'
    return data
