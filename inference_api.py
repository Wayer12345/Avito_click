from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
import pandas as pd

app = FastAPI()


class model_input(BaseModel):

    histctr : float
    subcategoryid : int
    price: float
    position: int
    categoryid: int
    parentcategoryid: int


with open('./final_files/model.pickle', 'rb') as f:
    model = pickle.load(f)

@app.get('/')
def home():
    return {"health_check": "OK"}

@app.post('/avito_click_prediction')
def model_pred(input_parameters : model_input):

    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    histctr = input_dictionary['histctr']
    subcategoryid = input_dictionary['subcategoryid']
    price = input_dictionary['price']
    position = input_dictionary['position']
    categoryid = input_dictionary['categoryid']
    parentcategoryid = input_dictionary['parentcategoryid']

    input_list = [histctr, subcategoryid, price, position, categoryid, parentcategoryid]

    final_input = pd.DataFrame([input_list], columns = ['histctr', 'subcategoryid', 'price', 'position', 'categoryid', 'parentcategoryid'])
    final_input = final_input.fillna(0)

    prediction = model.predict_proba(final_input)[:, 1]

    if prediction < 0.5:
        return "No Click"
    else:
        return "Click"

