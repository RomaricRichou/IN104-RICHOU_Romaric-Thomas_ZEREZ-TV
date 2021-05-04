import pandas as pd
import os
from dfply import *
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Création des deux dictionnaires vides
supply_1 = {}
supply_2 = {}

#This function sets the working directory
def set_wd(wd="/home/thomas/Documents/IN104/Projet_IN104/IN104-RICHOU_Romaric-Thomas_ZEREZ-TV/supply/"):
    os.chdir(wd)

# Importation des informations de storage
def import_excel(f_name = "storage_data.xlsx"):
    f = pd.read_excel(f_name)
    return f >> mutate(Date = pd.to_datetime(f['gasDayStartedOn']))

# Importation des informations de prix
def import_csv(f_name = "price_data.csv"):
    f = pd.read_csv(f_name, ';')
    return f >> mutate(Date = pd.to_datetime(f['Date']))

# classification
def classification(NWB, LNW):
    x = NWB
    y = LNW
    lr = LogisticRegression()
    lr.fit(x,y)

if __name__ == '__main__':

    #set working directory
    set_wd()

    #import storage data
    storage_data = import_excel()
    # print(supply_data)
    #import price data
    price_data = import_csv()
    # print(price_data)

    supply_data = pd.merge(storage_data, price_data, how = 'inner', on= 'Date')
    NW = supply_data.withdrawal - supply_data.injection
    LNW,NWB,FSW1,FSW2 = [],[],[],[]
    for i in range(len(NW)):
        if i >= 1:
            LNW.append(NW[i-1])
        if NW[i] >= 0:
            NWB.append(1)
        else:
            NWB.append(0)
        FSW1.append(max(supply_data.full[i] - 45, 0))
        FSW2.append(max(45 - supply_data.full[i], 0))

    classification(NW, LNW)
