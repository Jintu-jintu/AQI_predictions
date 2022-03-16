import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pickle
from sklearn.model_selection import train_test_split
data=pd.read_csv('pollution_us_2000_2016.csv')
data['Date Local'] = pd.to_datetime(data['Date Local'], format='%Y-%m')

#preprocessing
data1 = data.drop(['Unnamed: 0', 'Address', 'State', 'County', 'County Code', 'NO2 Units'
                  ,'SO2 Units', 'CO Units', 'O3 Units', 'Site Num'], axis = 1)
data1.drop(data1[data1['City'] == 'Not in a city'].index, inplace = True)
data1 = data1.reset_index().drop('index', axis = 1)
data1.insert(0, 'key', range(0, data1.shape[0]))

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data1['City'] = encoder.fit_transform(data1['City'])

from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor()
train_co = data1.drop('Date Local', axis = 1)[data1['CO AQI'].isna() == False]
test_co = data1.drop('Date Local', axis = 1)[data1['CO AQI'].isna() == True]
y_co = train_co['CO AQI']
train_co.drop(['CO AQI', 'SO2 AQI'], axis = 1, inplace = True)
dt_model.fit(train_co, y_co)
test_co.drop(['CO AQI', 'SO2 AQI'], axis = 1, inplace = True)
co_pred = dt_model.predict(test_co)
train_co['CO AQI'] = y_co
test_co['CO AQI'] = co_pred

train_so2 = data1.drop('Date Local', axis = 1)[data1['SO2 AQI'].isna() == False]
test_so2 = data1.drop('Date Local', axis = 1)[data1['SO2 AQI'].isna() == True]
y_so2 = train_so2['SO2 AQI']
train_so2.drop(['CO AQI', 'SO2 AQI'], axis = 1, inplace = True)
dt_model.fit(train_so2, y_so2)
test_so2.drop(['CO AQI', 'SO2 AQI'], axis = 1, inplace = True)
so2_pred = dt_model.predict(test_so2)
train_so2['SO2 AQI'] = y_so2
test_so2['SO2 AQI'] = so2_pred


merged1 = pd.concat([train_co, test_co], ignore_index = True)
merged2 = pd.concat([train_so2, test_so2], ignore_index = True)

df_1  = data1.drop(['CO AQI','SO2 AQI'],axis=1)
co_df = merged1[['CO AQI','key']]
so2_df = merged2[['SO2 AQI','key']]

imputed_df = pd.merge(co_df, so2_df, on='key').sort_values(by = 'key').reset_index().drop('index',axis = 1)
data_imputed = pd.merge(df_1, imputed_df, on= 'key').sort_values(by = 'key')

data_truncated = data_imputed.groupby([ 'Date Local','State Code', 
                                       'City']).mean().reset_index().drop('key', axis = 1)

#feature engineering
def aqiclass(x):
    if 0<= x <= 50:
        return 'good'
    elif 50< x <= 100:
        return 'moderate'
    elif 100< x <= 150:
        return 'unhealthy for sensitive groups'
    elif 150< x <= 200:
        return 'unhealthy'
    elif 200< x <= 300:
        return 'very unhealthy'
    else:
        return 'hazardous'

data_truncated['AQI_class'] = data_truncated[['NO2 AQI','CO AQI','SO2 AQI',
                                              'O3 AQI']].max(axis=1).apply(aqiclass)

data_truncated['AQI_class'] = encoder.fit_transform(data_truncated['AQI_class'])

#taking data for modeling
x = data_truncated[['NO2 AQI', 'SO2 AQI', 'CO AQI', 'O3 AQI']] 
y = data_truncated['AQI_class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
import xgboost
xgb_model = xgboost.XGBClassifier()
#Fitting the model
xgb_model.fit(x_train, y_train)
#Saving the model to disk
pickle.dump(xgb_model, open('model1.pkl','wb') )