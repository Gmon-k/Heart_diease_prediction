"""
2.Classification
Use at least three different ML methods to solve this task.
Model implmented below are
1)Logistic Regression
2)Decision Tree
3)Random Forest

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

# Fit logistic regression model
lr = LogisticRegression(max_iter=10000)
lr.fit(X_train_poly, y_train)

# Evaluate model on testing data
y_pred = lr.predict(X_test_poly)
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Compute bias and variance at default operating point
n = len(y_test)
p_pred = lr.predict_proba(X_test_poly)[:, 1]
bias = np.mean((p_pred - y_test) ** 2)
variance = np.mean(p_pred * (1 - p_pred))

print("Logistic Regression:")
print("----------------------------")
print("Accuracy:", acc)
print('\n')




# Fit decision tree model
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train_poly, y_train)

# Evaluate model on testing data
y_pred = dt.predict(X_test_poly)
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Compute bias and variance at default operating point
n = len(y_test)
p_pred = np.zeros(n)
for i in range(n):
    p_pred[i] = np.mean(dt.predict(X_test_poly[i].reshape(1, -1)) == y_test.iloc[i])
bias = np.mean((p_pred - y_test) ** 2)
variance = np.mean(p_pred * (1 - p_pred))

print("Decision Tree:")
print("----------------------------")
print("Accuracy:", acc)
print('\n')



# Initialize the Random Forest model with default hyperparameters
rf_model = RandomForestClassifier(random_state=42)

# Fit the model on the training data
rf_model.fit(X_train_poly, y_train)

# Make predictions on the test data
y_pred = rf_model.predict(X_test_poly)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Compute F1 score
f1 = f1_score(y_test, y_pred)

# Compute bias and variance at default operating point
y_pred_train = rf_model.predict(X_train_poly)
bias = np.mean((y_pred_train - y_train) ** 2)
y_pred_all = rf_model.predict(X_test_poly)
variance = np.mean((y_pred_all - np.mean(y_pred_all)) ** 2) - bias


print("Random Forest:")
print("----------------------------")
print("Accuracy:", acc)
print('\n')





