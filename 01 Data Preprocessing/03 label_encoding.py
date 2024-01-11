from sklearn.preprocessing import LabelEncoder
import pandas as pd 

Y = [['No'], ['Yes']]
Y_df = pd.DataFrame(Y, columns=['Purchased'])
Y = Y_df.values.ravel() # Flatten Y into 1D array

# Encode categorical data
le = LabelEncoder()
Y = le.fit_transform(Y)
print(Y)