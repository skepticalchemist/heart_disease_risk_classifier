import json
import requests


#
# This code allows to make predictions using the model deployed on Heroku
#

# features to test
features={
            "bmi": 23.63,
            "smoking": 1,
            "alcohol_drinking": 0,
            "stroke": 1,
            "physical_health": 0,
            "mental_health": 15,
            "diff_walking": 0,
            "sex": "female",
            "age_category": 11,
            "race": "other",
            "diabetic": "no",
            "physical_activity": 1,
            "gen_health": 3,
            "sleep_time": 8,
            "asthma": 0,
            "kidney_disease": 1,
            "skin_cancer": 1
            }

prediction = requests.post(
    "https://heart-disease-xgboost.herokuapp.com/classify",
    headers={"content-type": "application/json"},
    data=json.dumps(features)
).text

print(20*'-.')
print(f'Heart Disease >>> {prediction}')
print(20*'-.')