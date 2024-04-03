import pandas as pd
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier

import pickle

class Inference:
    def __init__(self, data):
        self.data = data
        self.data.columns = self.data.columns.str.lower()

    def preprocessing(self):
        with open('./final_files/final_features.pickle', 'rb') as f:
            self.final_features = pickle.load(f)
        with open('./final_files/model.pickle', 'rb') as f:
            self.model = pickle.load(f)

        self.data[['price', 'parentcategoryid', 'categoryid', 'subcategoryid', 'level']] = self.data[
            ['price', 'parentcategoryid', 'categoryid', 'subcategoryid', 'level']].fillna(0)

    def prediction(self):
        self.data['prediction'] = self.model.predict_proba(self.data[self.final_features])[:, 1]

# Тестирование

test_data = pd.read_parquet("./data/merged_data_test.parquet")
test_data_obj = Inference(test_data)
test_data_obj.preprocessing()
test_data_obj.prediction()