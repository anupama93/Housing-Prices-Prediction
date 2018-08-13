import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.grid_search import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# creating dataframe from csv file
df = pd.read_csv('data.csv')
del df["Id"]


# filled meadian value in misising values
numerical_data = df.select_dtypes(exclude = ['O']).fillna(df.median())
categorical_data = df.select_dtypes(include = ['O']).fillna('NA')


# done one hot encoding for 
data = numerical_data.join(categorical_data)
feature_list = list(data.columns)
df_one_hot = pd.get_dummies(data)
df_one_hot.head()


for i in list(df_one_hot.columns):
    mean = df_one_hot[i].mean()
    sd = df_one_hot[i].std()
    df_one_hot[i] = (df_one_hot[i]-mean)/sd
    
X = df_one_hot.drop('SalePrice', axis=1)
Y = df_one_hot["SalePrice"]



def final_abs_error_xg_boost(X, Y, n_splits = 5):
    kf = KFold(n_splits, random_state= 50, shuffle=True)
    error = 0
    
    param_grid = {
              'max_depth': [4, 6 , 8, 10],
              'learning_rate': [0.01,0.1,0.2,0.3]
              
              }
    for train_index, test_index in kf.split(X):
    
            X_train, X_test = X.as_matrix()[train_index], X.as_matrix()[test_index]
            Y_train, Y_test = Y.as_matrix()[train_index], Y.as_matrix()[test_index]
            
            xb = XGBRegressor(n_estimators = 2000)
            gs_cv = GridSearchCV(xb, param_grid, n_jobs=8).fit(X_train,Y_train)
            
            # xb = XGBRegressor(max_depth = gs_cv.best_params_['max_depth'],min_samples_leaf = gs_cv.best_params_['min_samples_leaf'],n_estimators = gs_cv.best_params_['n_estimators'])
            Y_pred = gs_cv.predict(X_test)
            error = error + sum(np.absolute(Y_pred-Y_test))/(len(Y_test))
                                                         
    return error/n_splits

print(final_abs_error_xg_boost(X, Y, 5))