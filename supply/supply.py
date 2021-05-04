import pandas as pd
import os
from dfply import *
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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
    print(supply_data)