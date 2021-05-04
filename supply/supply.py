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



if __name__ == '__main__':

    #set working directory
    set_wd()

    #import supply data
    supply_data = import_excel()