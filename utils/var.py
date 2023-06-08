#question id to name

QUESTION_LIST = [ "VH134067", 'VH266015', "VH302907","VH507804", "VH139380","VH304954",
                  "VH525628","VH266510_2017", "VH266510_2019", "VH269384","VH271613"]
QUESTION_NAME = ['rule','div','geo','4card','pattern','sub','least','slope_2017','slope_2019','8card','age']

Imbalance = ["VH271613", "VH266510_2017", "VH266510_2019", "VH269384", "VH507804"]
SampleList = [["2","2A",'2B',"3"], ["2","2A",'2B'],  ["2","2A",'2B'],  ["2","2A",'2B'],  ["2","2A",'2B']]
SampleDict = {key: value for key, value in zip(Imbalance, SampleList)}

QUESTION_TO_NAME = {key: value for key, value in zip(QUESTION_LIST,QUESTION_NAME)}
NAME_TO_QUESTION = {key: value for value, key in zip(QUESTION_LIST,QUESTION_NAME)}


COLS_RENAME = {'accession':'qid','predict_from':'text'}
LABEL0 = 'score_to_predict'
LABEL2 = 'r_label'
EST_SCORE = 'est_score'
EVAL_LABEL = 'eval_label'
BASE_COLS = ['id', 'qid', 'label', 'label1', 'text', 'fold', 'text1', 'text2']
CONTEXT_ALL = 'context_all'

PRE_QID = 'Question id: '
#PRE_CLOSED = 'Closed form response: '
PRE_CLOSED = ''
PRE_EXAMPLE = 'Examples: '
PRE_OVERALL_EXAMPLE = ''
PRE_QUERY_GRADE =  'Score this response: '
PRE_SCORE = 'Score: '
SEP = "\n"

LLAMA_LOCAL_FILEPATH = "/media/animal_farm/llama_hf"
ALPACA_LOCAL_FILEPATH = "/media/animal_farm/alpaca"

#group info
group_info={'srace10': [1,2,3,4,5,6,7], 'accom2':[1,2], 'iep':[1,2], 'lep':[1,2]}
