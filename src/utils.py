#common function for entire project
import os
import sys
sys.path.append('c:/Users/ariha/Desktop/Krish_Naik/mlproject')
import pandas as pd
import numpy as np
import dill
from src.exception import CustomException

def save_obj(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)