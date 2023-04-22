"""
1.Pre-processing, Data Mining, and Visualization
a)What variables do you plan to use as the input features?

submitted by : Gmon Kuzhiyanikkal
Date: 22 Feb 2022

"""


import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('heart.csv')

# Define list of column names to plot
columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']

# Define number of rows and columns for subplots
n_rows = 3
n_cols = 4

# Define figure size and spacing
fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, 15))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

# Loop through columns and create double histogram
for i, col in enumerate(columns):
    row_idx = i // n_cols
    col_idx = i % n_cols
    ax[row_idx, col_idx].hist(data[data['HeartDisease'] == 0][col], bins=30, alpha=0.5, label='No Heart Disease', color='blue')
    ax[row_idx, col_idx].hist(data[data['HeartDisease'] == 1][col], bins=30, alpha=0.5, label='Heart Disease', color='red')
    ax[row_idx, col_idx].set_title(col)
    ax[row_idx, col_idx].legend(loc='upper right')

# Show plot
plt.show()
