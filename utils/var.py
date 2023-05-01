#question id to name

QUESTION_LIST = [ "VH134067", 'VH266015', "VH302907","VH507804", "VH139380","VH304954",
                  "VH525628","VH266510_2017", "VH266510_2019", "VH269384","VH271613"]
QUESTION_NAME = ['rule','div','geo','4card','pattern','sub','least','slope_2017','slope_2019','8card','age']

QUESTION_TO_NAME = {key: value for key, value in zip(QUESTION_LIST,QUESTION_NAME)}

COLS_RENAME = {'accession':'qid','predict_from':'text'}
LABEL0 = 'score_to_predict'
BASE_COLS = ['qid', 'label', 'text']

CONTEXT_ALL = 'context_all'