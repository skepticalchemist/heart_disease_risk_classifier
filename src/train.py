import bentoml
import pandas as pd
import pickle
import sys

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier

# importing user defined funtions
sys.path.append("../src")
from modeling import get_metrics


# data loading
df = pd.read_csv('../data/heart_2020_cleaned_preproc_ordinal.csv')
print(df.shape)

# separating features and target
X = df.drop('heart_disease', axis=1)
y = df['heart_disease']

# data splitting: 60:20:20
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
    )
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, 
    y_train_full, 
    test_size=0.25, 
    random_state=42, 
    stratify=y_train_full
    )

# create a data dictionary of features
features = [
    'bmi',
    'smoking',
    'alcohol_drinking',
    'stroke',
    'physical_health',
    'mental_health',
    'diff_walking',
    'sex',
    'age_category',
    'race',
    'diabetic',
    'physical_activity',
    'gen_health',
    'sleep_time',
    'asthma',
    'kidney_disease',
    'skin_cancer'
]

train_dict = X_train[features].to_dict(orient='records')
val_dict = X_val[features].to_dict(orient='records')

# encode dictionary of features
dv = DictVectorizer(sparse=False)
X_train_dv = dv.fit_transform(train_dict)
X_val_dv = dv.transform(val_dict)

# model training
model_xgb = XGBClassifier(
    scale_pos_weight=10.07, 
    learning_rate=0.1, 
    max_depth=4, 
    min_child_weight=6, 
    objective='binary:logistic', 
    verbosity=1, 
    random_state=42, 
    n_jobs=-1
    )

model_xgb.fit(X_train_dv, y_train)
y_pred = model_xgb.predict(X_val_dv)

get_metrics(y_pred, y_val)

# save the model using pickle
with open("../model/model_xgb.bin", 'wb') as fh:
    pickle.dump((dv, model_xgb), fh)

# save the model using bentoml
# the deployed model was saved using BentoML on the notebook 03_Modeling:
#
# bentoml.xgboost.save_model(
#    'heart_disease_model',
#    model_xgb,
#    custom_objects={
#        'dictVectorizer': dv
#    })