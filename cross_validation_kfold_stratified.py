# 22 April 2025
# Cross Validation K-fold, stratified

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# Churn_Modelling.csv is used
dataset = pd.read_csv(r"C:\Users\GauravKunal\Desktop\DS\Machine Learning\#2 Classification\#8 Cross Validation\Churn_Modelling.csv")
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values


# Encoding Categorical data
# Label Encoding the 'Gender' Column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
# it converts categorical value to numerical
X[:,2] = le.fit_transform(X[:,2]) 


# One hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Training XGboost on the Training set
# ctrl+i not work on this library for seeing parameters just run the code to see parameters
from xgboost import XGBClassifier 
# classifier = XGBClassifier() # ac-0.8545
# classifier = XGBClassifier(n_estimator = 1000, max_depth =4,learning_rate=0.001,random_state=0) # ac-0.7975
classifier = XGBClassifier(random_state=0) # ac- 0.8545
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)
y_pred

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

bias = classifier.score(X_train, y_train)
bias

variance = classifier.score(X_test, y_test)
variance



# Only when model overfits then use k-fold

# Now at this point suppose your model is overfits let's apply k-fold
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X=X_train, y =y_train,cv =5 )
print("Accuracy:{:.2f} %".format(accuracies.mean()*100))

# by this way we reduce the overfitting 
# For understandig the concept of K-fold meet classmate pg no.155









# stratified cross validation

stratified cv code

ChatGPT said:
Here's a simple and clear example of Stratified K-Fold Cross-Validation
using Scikit-learn, commonly used for classification tasks 
to ensure each fold has the same class proportion as the full dataset.

✅ Python Code: Stratified K-Fold CV Example

from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load dataset
X, y = load_iris(return_X_y=True)

# Initialize StratifiedKFold
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) #--> give by gpt
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0) #--> give by kodi


# Model
model = RandomForestClassifier(random_state=42)

# To store accuracy for each fold
accuracies = []

# Stratified Cross-Validation Loop
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    
    print(f"Fold {fold} Accuracy: {acc:.4f}")

print(f"\nAverage Accuracy: {np.mean(accuracies):.4f}")
📌 Notes:
StratifiedKFold ensures class distribution is maintained in each fold — very useful for imbalanced datasets.

You can replace the model and dataset with your own.
















