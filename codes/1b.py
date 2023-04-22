"""
1.Pre-processing, Data Mining, and Visualization
b)What pre-processing (if any) did you execute on the variables?
c)Which independent variables are strongly correlated (positively or negatively)?
d)How many significant signals exist in the independent variables?
e)What derived or alternative features might be useful for analysis 
  (e.g. polynomial features)?

submitted by : Gmon Kuzhiyanikkal
Date: 22 Feb 2022
"""



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
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

# Plot number of components and explained variance
fig, ax = plt.subplots(2, 1, figsize=(10, 8))
ax[0].plot(range(1, n_components+1), cumulative_var, marker='o')
ax[0].set_xlabel('Number of components')
ax[0].set_ylabel('Explained variance ratio (cumulative)')
ax[1].bar(range(1, n_components+1), explained_var)
ax[1].set_xlabel('Number of components')
ax[1].set_ylabel('Explained variance ratio')
plt.tight_layout()
plt.show()


# Perform PCA
pca = PCA()
pca.fit(X)
explained_var = pca.explained_variance_ratio_

# Find number of components that explain at least 90% of the variance
cumulative_var = explained_var.cumsum()
n_components = len(cumulative_var[cumulative_var <= 0.9])
print("\n")
print("-------------------------")
print(f"Number of components with at least 90% variance: {n_components}")
print("\n")





# Use linear regression to identify features that are strongly correlated with heart disease
print("\n")
print("---------------------")
print("Correlation with the data")
print("-------------------------")
lr = LinearRegression()
lr.fit(X, y)
coefficients = pd.Series(lr.coef_, index=heart_df.drop("HeartDisease", axis=1).columns)
coefficients.sort_values(ascending=False, inplace=True)
print(coefficients)


print("\n")
print("---------------------")
print("Print the shape of the new feature matrix before adding the polynomial features")
print("-------------------------")
print(X.shape)

# Add polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Print the shape of the new feature matrix
print("\n")
print("---------------------")
print("Print the shape of the new feature matrix after adding the polynomial features")
print("-------------------------")
print(X_poly.shape)


