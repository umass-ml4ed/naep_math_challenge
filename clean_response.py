import pandas as pd
import numpy as np
import torch

import neuspell
from neuspell import BertChecker

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checker = BertChecker(device=device)
checker.from_pretrained()

train_df = pd.read_csv("./data/train.csv")

train_df["predict_from"] = train_df["predict_from"].astype(str).replace('"', "")
train_df["predict_from"] = train_df["predict_from"].replace()
train_df["predict_from"] = train_df["predict_from"].replace(r'^\s*$', np.nan, regex=True)

empty_pred_idx_list = train_df.index[train_df["predict_from"] == np.nan].tolist()

train_text = train_df["predict_from"].astype(str).values.tolist()

train_text_spell_checked = checker.correct_strings(train_text)

train_df["predict_from"] = pd.Series(train_text_spell_checked)

train_df.loc[empty_pred_idx_list, "predict_from"] = np.nan

train_df.to_csv("./data/train_spell_checked.csv", index=False)