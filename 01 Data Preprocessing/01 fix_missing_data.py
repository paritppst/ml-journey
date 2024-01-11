from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd


''' 
    A generule of thumb is that, 
    you should select all numerical values 
    to replace the missing data 
    (So there will be no errors) 
'''

# Convert X into a DataFrame
X = [['France', 1, 2], ['Thailand', '', 4]]
X_df = pd.DataFrame(X, columns=['Country', 'Feature1', 'Feature2'])

# Replace empty strings with NaN
X_df.replace('', np.nan, inplace=True)

X = X_df.values # convert DataFrame to numpy array
print(f'Initial np array:\n {X}\n')

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3]) # fit imputer object to matrix X
X[:, 1:3] = imputer.transform(X[:, 1:3]) # replace missing data with mean of column
print(f'After feature scaling np array:\n {X}')