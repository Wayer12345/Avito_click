import json
import requests
import pandas as pd

url = 'http://localhost:80/avito_click_prediction'

data = pd.read_csv('./data/testSearchStream.tsv', sep = '\t')

input_data_for_model = {
    'histctr' : float(data.iloc[500].tolist()[0]),
    'subcategoryid' : int(data.iloc[500].tolist()[1]),
    'price': float(data.iloc[500].tolist()[2]),
    'position': int(data.iloc[500].tolist()[3]),
    'categoryid': int(data.iloc[500].tolist()[4]),
    'parentcategoryid': int(data.iloc[500].tolist()[5])
}

input_json = json.dumps(input_data_for_model)

response = requests.post(url, data=input_json)

print(response.text)
