"""
3.Evaluation
For at least one of your methods, generate a Receiver Operator Curve [ROC] 
by varying the operating point of the classifier.

Model taken for analysis is : Logistic Regression

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

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Train the Logistic Regression model on the training data
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_poly, y_train)

# Make predictions on the test data
y_pred_proba = lr_model.predict_proba(X_test_poly)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
