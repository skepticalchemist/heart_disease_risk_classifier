from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import preprocessing

from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    ConfusionMatrixDisplay,
    confusion_matrix
)
from sklearn.model_selection import learning_curve, train_test_split, RepeatedStratifiedKFold, cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, plot_importance

from collections import Counter

pd.set_option('display.expand_frame_repr', False)


def get_metrics(y_pred, y_test, return_metrics=False):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    matthews_cc = matthews_corrcoef(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    specificity = recall_score(y_test, y_pred, pos_label=0)

    if return_metrics:
        return accuracy, f1, matthews_cc, precision, recall, roc_auc, specificity
    else:
        print(25 * '-')
        print(8 * ' ', 'METRICS', 8 * ' ')
        print(25 * '-')
        print(f"{'Accuracy': <14}{' =   ': >2}{round(accuracy, 4): <6}")
        print(f"{'F1 Score': <14}{' =   ': >2}{round(f1, 4): <6}")
        print(f"{'MCC': <14}{' =   ': >2}{round(matthews_cc, 4): <6}")
        print(f"{'Precision': <14}{' =   ': >2}{round(precision, 4): <6}")
        print(f"{'Recall': <14}{' =   ': >2}{round(recall, 4): <6}")
        print(f"{'ROC/AUC': <14}{' =   ': >2}{round(roc_auc, 4): <6}")
        print(f"{'Specificity': <14}{' =   ': >2}{round(specificity, 4): <6}")
        print(12 * '-.')


def plot_cf_matrix(y_pred, y_test, title):

    confusion = confusion_matrix(y_test, y_pred)

    class_names = ["no_heart_disease", "heart_disease"]
    disp = ConfusionMatrixDisplay(confusion, display_labels=class_names).plot()
    disp.ax_.set(
        title=title,
        xlabel='Predicted Cases',
        ylabel='Actual Cases'
    )
    plt.show()
    plt.close()


def train_and_evaluate_models(X_train, y_train, X_test, y_test, models, model_names, plot_title):

    accuracy = []
    f1 = []
    precision = []
    recall = []
    specificity = []
    matthews_cc = []
    roc_auc = []

    for model in range(len(models)):
        regression_model = models[model]
        regression_model.fit(X_train, y_train)
        y_pred = regression_model.predict(X_test)

        accuracy.append(np.round(accuracy_score(y_test, y_pred), 2))
        f1.append(np.round(f1_score(y_test, y_pred), 2))
        precision.append(np.round(precision_score(y_test, y_pred), 2))
        recall.append(np.round(recall_score(y_test, y_pred), 2))
        matthews_cc.append(np.round(matthews_corrcoef(y_test, y_pred), 2))
        roc_auc.append(np.round(roc_auc_score(y_test, y_pred), 2))
        specificity.append(np.round(recall_score(y_test, y_pred, pos_label=0), 2))

    eval_accuracy = {'Modeling Algorithm': model_names, 'Accuracy': accuracy}
    eval_f1 = {'Modeling Algorithm': model_names, 'F1 Score': f1}
    eval_precision = {'Modeling Algorithm': model_names, 'Precision': precision}
    eval_recall = {'Modeling Algorithm': model_names, 'Recall': recall}
    eval_specificity = {'Modeling Algorithm': model_names, 'Specificity': specificity}
    eval_matthews_cc = {'Modeling Algorithm': model_names, 'MCC': matthews_cc}
    eval_roc_auc = {'Modeling Algorithm': model_names, 'ROC/AUC': roc_auc}

    # create a table containing the performance of each model
    df_accuracy = pd.DataFrame(eval_accuracy)
    df_f1 = pd.DataFrame(eval_f1)
    df_precision = pd.DataFrame(eval_precision)
    df_recall = pd.DataFrame(eval_recall)
    df_specificity = pd.DataFrame(eval_specificity)
    df_matthews_cc = pd.DataFrame(eval_matthews_cc)
    df_roc_auc = pd.DataFrame(eval_roc_auc)

    score_table = pd.concat(
        [
            df_accuracy,
            df_f1["F1 Score"],
            df_precision["Precision"],
            df_recall["Recall"],
            df_specificity['Specificity'],
            df_matthews_cc['MCC'],
            df_roc_auc['ROC/AUC']
        ], axis=1
    )

    print(90*'-')
    print(34*' ', 'Score Table', 34*' ')
    print(90 * '-')
    print(score_table.sort_values(by="F1 Score", ascending=[True]).to_string(index=False))
    print(45 * '-.')
    print()

    # plot model performances
    plt.style.use("fivethirtyeight")
    ax = score_table.sort_values(by=["MCC"], ascending=True).plot(
        kind="barh",
        x="Modeling Algorithm",
        figsize=(10, 12),
        stacked=False,
        color=["#4682B4", "#7846B4", "#B47846", "#82B446", '#B4464B', '#00468B', '#008B8B']
    )
    ax.legend(bbox_to_anchor=(1, 0.7))
    plt.grid(which="minor", axis="x", color="k")
    plt.title(plot_title)
