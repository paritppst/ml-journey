import pandas as pd
from sklearn.model_selection import train_test_split

data = [['France', 1, 2, 'No'], ['Thailand', 2, 4, 'Yes'], ['Spain', 100, 10, 'Yes']]
dataset = pd.DataFrame(data, columns=['Country', 'Feature1', 'Feature2', 'Purchased'])

X = dataset.iloc[:, :-1].values # independent variables
Y = dataset.iloc[:, -1].values # dependent variable 

# Splitting the dataset into the Training set and Test set
# test_size=0.2 means 20% of the dataset will be used for testing
# random_state=0 means the data will be split in the same way every time
# (so we can compare the results of different models)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0 )

print(f'X_train:\n{X_train}\n')
print(f'X_test:\n{X_test}\n') 

print(f'Y_train:\n{Y_train}\n')
print(f'Y_test:\n{Y_test}\n')