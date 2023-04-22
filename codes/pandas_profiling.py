"""
1.Pre-processing, Data Mining, and Visualization
a)What variables do you plan to use as the input features?
b)What pre-processing (if any) did you execute on the variables?
c)Which independent variables are strongly correlated (positively or negatively)?
d)How many significant signals exist in the independent variables?

submitted by : Gmon Kuzhiyanikkal
Date: 22 Feb 2022

PS: it has used pandas profiling to get a idea of the data.
"""

import pandas as pd
from pandas_profiling import ProfileReport

data = pd.read_csv("heart.csv")
report = ProfileReport(data)
report.to_file("your_report.csv")


