import pickle
import os

## Importing supply data
filename = '/home/thomas/Documents/IN104/Projet_IN104/IN104-RICHOU_Romaric-Thomas_ZEREZ-TV/supply/finalized_model.sav'
filename_2 = '/home/thomas/Documents/IN104/Projet_IN104/IN104-RICHOU_Romaric-Thomas_ZEREZ-TV/supply/X.sav'
filename_3 = '/home/thomas/Documents/IN104/Projet_IN104/IN104-RICHOU_Romaric-Thomas_ZEREZ-TV/supply/stockage.sav'
loaded_model = pickle.load(open(filename, 'rb'))
X = pickle.load(open(filename_2, 'rb'))
stockage = pickle.load(open(filename_3, 'rb'))

## Importing consumption data
filename = '/home/thomas/Documents/IN104/Projet_IN104/IN104-RICHOU_Romaric-Thomas_ZEREZ-TV/demand/consumption_model.sav'
loaded_consumption = pickle.load(open(filename, 'rb'))

y_pred_binary = {}
y_pred_num = {}

if __name__ == '__main__':
    for i in loaded_model['classification']:
        y_pred_binary[i] = pd.DataFrame({})
        y_pred_binary[i]['Valueb'] = loaded_model['classification'][i]['class_mod'].predict(X[0][i][0])
        y_pred_binary[i]['Date'] = X[0][i][1]
        stockage[i] = pd.merge(stockage[i], y_pred_binary[i], how = 'inner', on = 'Date')
        temp = stockage[i] >> mask(stockage[i]['Valueb'] > 0)

        y_pred_num[i] = pd.DataFrame({})
        y_pred_num[i]['Value'] = loaded_model['regression'][i]['l_reg'].predict(X[1][i][0])
        y_pred_num[i]['Date'] = X[1][i][1]
        temp = pd.merge(temp, y_pred_num[i], how = 'inner', on = 'Date')

        stockage[i] = pd.merge(temp, stockage[i], on = 'Date', how = 'outer')
        stockage[i] = stockage[i].fillna(0)

    stockage_f = pd.DataFrame({})
    stockage_f['Date'] = []
    for i in stockage:
        stockage_i=pd.DataFrame({})
        stockage_i['Date'] = stockage[i]['Date']
        stockage_i[i] = stockage[i]['Value']
        stockage_f = pd.merge(stockage_f, stockage_i, on = 'Date', how = 'outer')
    supply = stockage_f.sum(axis = 1, numeric_only = True)
    stockage_final = pd.DataFrame({})
    stockage_final['Date'] = stockage_f['Date']
    stockage_final['Supply'] = supply

    supply_demand = pd.merge(stockage_final, loaded_consumption, on = 'Date', how = 'inner')