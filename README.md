# Heart Disease Risk Classifier

Here you'll find how to selecting and training a classification model to predict the risk for heart disease, putting it to a web service, and deploying it using BentoML and docker on Heroku.

This repository is part of my MLzoomcamp Midterm Project. 

## The problem

The [risk for heart disease](https://www.cdc.gov/heartdisease/risk_factors.htm) can be associated with health conditions and factors such as lifestyle, age, and family history. A binary classification model based on these variables collected from patients can help in healthcare diagnosis.

## The data

A [reduced version](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease) of the [CDC](https://www.cdc.gov/) dataset (February 2022 update) containing several features that can be associated with the risk of heart disease is used to build the classification model. Reduced here means that the 300 variables from the original dataset were reduced to 20.  


Notebooks contents:
* _01_EDA.ipynb_
  *Performs an exploratory data analysis (EDA) using the library [Sweetviz](https://github.com/fbdesignpro/sweetviz)
* _02_Preprocessing.ipynb_
  * Performs data preprocessing steps, and
  * Feature importance analysis
* _03_Modeling.ipynb_
  * Enconding categorical features
  * Model selection 
    * Build, select and evaluate models:
      * K-Nearest Neighbors
      * Logistic Regression
      * Decision Tree
      * Random Forest
      * XGBoost
      * Gaussian NB
      * Complement NB
    * Without handling class imbalance
    * Handling class imbalance with class weight
  * Hyperparameter tuning
  * BentoML: saving the selected model

## How to run this project

* Create a virtual environment of your choice
* Clone this repository
* Install project dependencies: 
```bash
 pip install -r requeriments.txt 
```

* Inside the repository create a new directory called `data` and download this [Kaggle](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease) dataset to it.


# Directory structure of the repository

```bash

```

# How to access the model deployed on Heroku

https://heart-disease-xgboost.herokuapp.com/