import pickle
import sys

#
# This code allows to make predictions using the local model
#
with open('../model/model_xgb.bin', 'rb') as fh:
    dv, model_xgb = pickle.load(fh)

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

X_test_dv = dv.transform(features)

y_pred = model_xgb.predict_proba(X_test_dv)[0, 1]
print(y_pred)
