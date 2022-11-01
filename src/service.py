import bentoml
import numpy as np
from bentoml.io import JSON

model_ref = bentoml.xgboost.get("heart_disease_model:cwhjxjsys2beufhw")
dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()

svc = bentoml.Service("heart_disease_classifier", runners=[model_runner])


@svc.api(input=JSON(), output=JSON())
async def classify(application_data):
    vector = dv.transform(application_data)
    prediction = await model_runner.predict.async_run(vector)
    print(prediction)
    result = prediction[0]

    if result > 0.7:
        return {
                "status": "High risk"
        }
    elif result > 0.5:
        return {
            "status": "Risk"
        }
    else:
        return {
            "status": "Healthy"
        }
