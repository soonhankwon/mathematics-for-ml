import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
from matplotlib import pyplot as plt

data = pd.read_csv("./Advertising.csv")
print(data.shape)
"""
(200, 5)
"""

plt.scatter(data['TV'], data['Sales'])
plt.title('TV vs Sales')
plt.xlabel('TV')
plt.ylabel('Sales')

plt.show()
