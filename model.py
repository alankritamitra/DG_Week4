# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Importing the dataset
df = pd.read_csv('iris.csv')

# mapping
variety_mappings = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

# Encoding the target variables
df = df.replace(['Setosa', 'Versicolor' , 'Virginica'],[0, 1, 2])

X = df.iloc[:, 0:-1] 
y = df.iloc[:, -1] 
# Initializing the Logistic Regression model
logreg = LogisticRegression() 
# Fitting the model
logreg.fit(X, y) 


def classify(a, b, c, d):
    arr = np.array([a, b, c, d]) 
    arr = arr.astype(np.float64) 
    query = arr.reshape(1, -1) 
    prediction = variety_mappings[logreg.predict(query)[0]] 
    return prediction # Return the prediction
