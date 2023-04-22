
""""
Extensions 2

Use more ML methods.

I have implemented support vector machine. Compared its result with other models. 

submitted by : Gmon Kuzhiyanikkal
Date: 22 Feb 2022
"""



from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA



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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the SVM classifier and fit it to the training data
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(X_train, y_train)

# Use the trained model to make predictions on the testing data
y_pred = svm.predict(X_test)

# Compute the evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the evaluation metrics
print("\n")
print("Support vector Machine")
print("-------------------------------------")
print("Accuracy: {}".format(accuracy))
print("Precision: {}".format(precision))
print("Recall: {}".format(recall))
print("F1 Score: {}".format(f1))
print("Confusion Matrix: \n{}".format(conf_matrix))
