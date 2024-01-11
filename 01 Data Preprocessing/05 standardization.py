import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split



data = [['France', 1, 2, 'No'], ['Thailand', 2, 4, 'Yes'], ['Spain', 100, 10, 'Yes']]
dataset = pd.DataFrame(data, columns=['Country', 'Feature1', 'Feature2', 'Purchased'])

# Create matrix 
X = dataset.iloc[:, :-1].values # independent variables
Y = dataset.iloc[:, -1].values # dependent variable 

# Encoding
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X)) # convert sparse matrix to numpy array
print(f'X after encoding: \n{X}\n')

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0 )

# Standardization
sc = StandardScaler()
X_train[:, 2:] = sc.fit_transform(X_train[:, 2:]) # standardize only numerical data, not dummy variables
print(f'X_train after standardization: \n{X_train}\n')

X_test[:, 2:] = sc.transform(X_test[:, 2:]) # standardize test set using the scaler that fits from train set
print(f'X_test after standardization: \n{X_test}\n')