import pandas as pd
import numpy as np
from math import ceil


# User defined functions

def dataset_dimensions(df):
    """
    Show dimensions of the dataframe.
    """
    print("Dimensions of the dataset:")
    print(f" Number of rows: {df.shape[0]: >9}")
    print(f" Number of columns: {df.shape[1]}\n")

    return


def camel_to_snake_case(str):
    """
    Convert a name from CamelCase to snake_case.
    """
    return ''.join(['_'+i.lower() if i.isupper()
               else i for i in str]).lstrip('_')


def rename_columns(df):
    """
    Rename dataframe columns from CamelCase to snake_case
    """
    column_names = list(df.columns)

    new_column_names = []

    for name in column_names:
        new_column_names.append(camel_to_snake_case(name))

    for column in range(len(new_column_names)):
        if  new_column_names[column]== 'b_m_i':
            new_column_names[column] = 'bmi'

    df.columns = new_column_names 


def column_unique_values(df):
    """
    Show unique values for each column in the dataframe.
    """
    for col in df.columns:
        print(f"{col: >24}: {df[col].nunique()}")

    return


def column_missing_values(df):
    """
    Show the total number of missing values per columns, if they are present.
    """
    if df.isnull().sum()[df.isnull().sum() > 0].any():
        print("Columns with missing values:\n")
        return df.isnull().sum()[df.isnull().sum() > 0]
    else:
        print('No missing values found!')

