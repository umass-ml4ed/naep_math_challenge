#question id to name

QUESTION_LIST = [ "VH134067", 'VH266015', "VH302907","VH507804", "VH139380","VH304954",
                  "VH525628","VH266510_2017", "VH266510_2019", "VH269384","VH271613"]
QUESTION_NAME = ['rule','div','geo','4card','pattern','sub','least','slope_2017','slope_2019','8card','age']

QUESTION_TO_NAME = {key: value for key, value in zip(QUESTION_LIST,QUESTION_NAME)}
NAME_TO_QUESTION = {key: value for value, key in zip(QUESTION_LIST,QUESTION_NAME)}


COLS_RENAME = {'accession':'qid','predict_from':'text'}
LABEL0 = 'score_to_predict'
LABEL2 = 'r_label'
EST_SCORE = 'est_score'
EVAL_LABEL = 'eval_label'
BASE_COLS = ['qid', 'label', 'text', 'fold']
CONTEXT_ALL = 'context_all'

PRE_QID = 'Question id: '
PRE_CLOSED = 'Closed form response: '
PRE_EXAMPLE = 'Examples: '
PRE_OVERALL_EXAMPLE = ''
PRE_QUERY_GRADE =  'Score this response: '
PRE_SCORE = 'Score: '
SEP = "[SEP]"
