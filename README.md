# Heart Disease Risk Classifier

Here you'll find a solution to selecting and training a classification model to predict the risk for heart disease, putting it to a web service using [BentoML](https://www.bentoml.com/) and docker and deploying it on a cloud platform as [Heroku](https://www.heroku.com/).

This repository is part of my MLzoomcamp Midterm Project. 

## The problem

The [risk for heart disease](https://www.cdc.gov/heartdisease/risk_factors.htm) can be associated with health conditions and factors such as lifestyle, age, and family history. A binary classification model based on these variables collected from patients can help in healthcare diagnosis.

## The data

A [reduced version](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease) of the [CDC](https://www.cdc.gov/) dataset (February 2022 update) containing several features that can be associated with the risk of heart disease is used to build the classification model. Reduced here means that the 300 variables from the original dataset were reduced to 20. 

The imbalanced data consists of 17 attributes and one target variable `heart_disease`.

## The model

The best model was selected from:
* K-Nearest Neighbors
* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost
* Gaussian NB
* Complement NB

First baseline models were obtained without handling the class imbalance, and then news models were build handling class imbalance with class weights. Techiniques like oversampling and/or undersampling were not applied here.

## The metrics

Several metrics were calculated but the most useful for this problem were F1, recall and MCC (Matthews correlation coefficient).

## Structure of the repository

The repository has the following file structure:
```bash
.
├── LICENSE
├── README.md
├── requirements.txt
├── model
│   └── model_xgb.bin
├── notebooks
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   └── 03_Modeling.ipynb
└── src
    ├── bentofile.yaml
    ├── Dockerfile
    ├── modeling.py
    ├── predict_on_heroku.py
    ├── predict.py
    ├── service.py
    ├── train.py
    ├── utility.py
```

### Directory contents:
* _/model_:
  * contains only the local model, identical to the deployed one
* _/src_:
  * contains user defined functions (utility.py), config files and some python scripts
* _/notebooks_
  * _01_EDA.ipynb_
    * Performs an exploratory data analysis (EDA) using the library [Sweetviz](https://github.com/fbdesignpro/sweetviz)
  * _02_Preprocessing.ipynb_
    * Performs some data preprocessing steps, and
    * Does feature importance analysis
  * _03_Modeling.ipynb_
    * Encodes categorical features
    * Allows to build, select and evaluate models
    * Performs model hyperparameter tuning
    * Saves the selected model using BentoML


> Note: It's recommended to open the notebook _01_EDA_ with https://nbviewer.org/ otherwise you won't be able to visualize the EDA. 

## How to run this project

* Create a virtual environment of your choice and activate it
* Clone this repository
* Install project dependencies: 
```bash
 pip install -r requirements.txt 
```

* Inside the repository create a new directory called `data` and manually download this [Kaggle](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease) dataset to it.
* Run and inspect all notebooks from 01 to 03
* The best model is saved on notebook 03 using BentoML. From here you will need the files `service.py` and `bentofile.yaml` located in the `/src`, from where you should run:
```bash
$ bento build
$ bento containerize `tag`
```
You'll get the `tag` after running `bento build` command.
If you want to run the container, run:
```bash
$ docker run -it --rm -p 3000:3000 `tag`
```
After having containerinzing the best model using BentoML, if you want to deploy it on the cloud platform Heroku you can find friendly instructions in this [video](https://youtu.be/quBNcEzDhyA).
* 

## How to access the model deployed on the cloud

You can use the web app to make predictions in two ways:
1. You can make a request using the python code `/src/predict_on_heroku.py`. You'll find a data example inside the file which you can use to test the app.
2. Access the link https://heart-disease-xgboost.herokuapp.com/
   * On **Service APIs** menu click expand the `POST` menu
   * Click on `Try it out`
   * Paste one of the examples bellow (json/dictionary) into the field **Request body**. Pay attention to not use doble curly brackets
   * Click on `Execute`
   * Scrool down and look for the prediction on the field **Server response**

```json
example_1 = {
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

example_2 = {
  "bmi": 25.86,
  "smoking": 1,
  "alcohol_drinking": 0,
  "stroke": 0,
  "physical_health": 0,
  "mental_health": 0,
  "diff_walking": 0,
  "sex": "female",
  "age_category": 11,
  "race": "white",
  "diabetic": "no",
  "physical_activity": 1,
  "gen_health": 3,
  "sleep_time": 7,
  "asthma": 1,
  "kidney_disease": 0,
  "skin_cancer": 0
}
```

