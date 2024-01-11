from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd 

X = [['France', 1, 2], ['Thailand', 2, 4]]
X_df = pd.DataFrame(X, columns=['Country', 'Feature1', 'Feature2'])
X = X_df.values # convert DataFrame back to numpy array

# Encode categorical data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X)) # convert sparse matrix to numpy array
print(X)