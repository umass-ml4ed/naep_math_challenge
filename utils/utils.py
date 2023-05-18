import os
#import utils.var as var
def safe_makedirs(path_):
    if not os.path.exists(path_):
        try:
            os.makedirs(path_)
        except FileExistsError:
            pass


