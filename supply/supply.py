import pandas as pd
import os
from dfply import *
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from scipy.special import expit
import sklearn.metrics as metrics
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Création des dictionnaires vides
supply_data = {}
NW, LNW, FSW1, FSW2, NWB = {},{},{},{}, {}
d = {}

#This function sets the working directory
def set_wd(wd="/home/thomas/Documents/IN104/Projet_IN104/IN104-RICHOU_Romaric-Thomas_ZEREZ-TV/supply/"):
    os.chdir(wd)

# Importation des informations de storage
def import_excel(f_name = "storage_data.xlsx"):
    f = pd.read_excel(f_name, sheet_name=None)
    # print(f)
    for i in f:
        f[i] = f[i] >> mutate(Date = pd.to_datetime(f[i]['gasDayStartedOn']))
    return f

# Importation des informations de prix
def import_csv(f_name = "price_data.csv"):
    f = pd.read_csv(f_name, ';')
    return f >> mutate(Date = pd.to_datetime(f['Date']))

# classification
def reglineaire(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    lr = LogisticRegression()
    lr.fit(x,y)
    y_pred = lr.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    return {'recall': metrics.recall_score(y_test, y_pred), 'neg_recall': cm[1,1]/(cm[0,1] + cm[1,1]), 'confusion': cm, 'precision': metrics.precision_score(y_test, y_pred), 'neg_precision':cm[1,1]/cm.sum(axis=1)[1], 'roc': metrics.roc_auc_score(y_test, probs), 'class_mod': "the logistic regression"}

if __name__ == '__main__':

    #set working directory
    set_wd()

    #import storage data
    storage_data = import_excel()

    #import price data
    price_data = import_csv()

    for i in storage_data:
        supply_data[i] = pd.merge(storage_data[i], price_data, how = 'inner', on= 'Date')
        NW[i] = supply_data[i].withdrawal - supply_data[i].injection
        NWB[i], LNW[i], FSW1[i], FSW2[i] = [], [], [], []
        for j in range(len(NW[i])):
            if j >= 1:
                LNW[i].append(NW[i][j-1])
            if NW[i][j] >= 0:
                NWB[i].append(1)
            else:
                NWB[i].append(0)

            FSW1[i].append(max(supply_data[i].full[j] - 45, 0))
            FSW2[i].append(max(45 - supply_data[i].full[j], 0))
        x = [LNW[i], FSW1[i], FSW2[i], price_data.SAS_GPL, price_data.SAS_NBP, price_data.SAS_NCG, price_data.SAS_TTF]
        y = NWB[i]
        d[i] = reglineaire(x, y)









