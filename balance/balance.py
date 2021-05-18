import pickle

## Importing supply data
filename = '/home/thomas/Documents/IN104/Projet_IN104/IN104-RICHOU_Romaric-Thomas_ZEREZ-TV/supply/finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))


## Importing consumption data
filename = '/home/thomas/Documents/IN104/Projet_IN104/IN104-RICHOU_Romaric-Thomas_ZEREZ-TV/demand/consumption_model.sav'
loaded_consumption = pickle.load(open(filename, 'rb'))