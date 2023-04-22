"""
4.Iteration
What is the best performing classifier you can build? 
Select one of your classifiers and go through at least three iterations of 
making a modification to it to see if you can improve its performance on a 
statistic of your choice

Model taken for analysis is : Random Forest(hyper parameter tunning)

First Iteration : increased number of trees
Second Iteration : increase the depth
third Iteration : modified the minimum number of samples required to split an internal nodeclear

submitted by : Gmon Kuzhiyanikkal
Date: 22 Feb 2022
"""



from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
heart_df = pd.read_csv("heart.csv")


# Convert categorical variables to dummy variables and drop original columns
# Convert categorical variables to dummy variables
cat_cols = ["Sex", "ChestPainType", "FastingBS", "RestingECG", "ST_Slope", "ExerciseAngina"]

# one-hot encode categorical columns
for col in cat_cols:
    dummies = pd.get_dummies(heart_df[col], prefix=col, drop_first=True)
    heart_df = pd.concat([heart_df, dummies], axis=1)
    heart_df.drop(columns=col, inplace=True)

# Split the data into features and target variable
X = heart_df.drop(["HeartDisease"], axis=1)
y = heart_df["HeartDisease"]

# Handle missing values
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Perform PCA
pca = PCA()
pca.fit(X)
explained_var = pca.explained_variance_ratio_
n_components = pca.n_components_
cumulative_var = explained_var.cumsum()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Add polynomial features of degree 2
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


#first iteration 
# Random Forest Classifier with 200 trees
rfc = RandomForestClassifier(n_estimators=200, random_state=42)
rfc.fit(X_train, y_train)

# Predict test data
y_pred = rfc.predict(X_test)

print("First Iteration: Random Forest Classifier with 200 trees")
print("----------------------------")
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', cm)


#Second iteration:
#I will modify the maximum depth of each tree in the Random Forest model from the default value of None to 4.


# Random Forest Classifier with 200 trees and max depth of 7
rfc = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42)
rfc.fit(X_train, y_train)

# Predict test data
y_pred = rfc.predict(X_test)
print('\n')

print("second Iteration: modify the maximum depth of each tree in the Random Forest model")
print("----------------------------")
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', cm)
print('\n')

#Third iteration:
#I will modify the minimum number of samples required to split an internal node in the Random Forest model from the default value of 2 to 5.

# Random Forest Classifier with 200 trees, max depth of 7, and min samples split of 7
rfc = RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_split=7, random_state=42)
rfc.fit(X_train, y_train)

# Predict test data
y_pred = rfc.predict(X_test)

print("Third Iteration: modify the minimum number of samples required to split an internal nodeclear")
print("----------------------------")

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', cm)



