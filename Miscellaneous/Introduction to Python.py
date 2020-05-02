"""
Created on Sun Jun 30 22:53:48 2019

@author: MatthiasQ
"""

import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
    #Allows you to code graphics in 'ggplot' manners


np.random.seed(1234) #Just like R's rng
data = np.round(np.random.normal(loc = 5, scale = 2, size = 100))
plt.hist(x = data, bins = 10, range = (-3, 10), edgecolor = 'black',
         title = 'Sample Histogram')
#Notice how the code above has a similar layout to the ggplot code


"Importing a Dataset"

#You can use either Numpy or Pandas

import numpy as np
import pandas as pd

"To import a .csv file"
#Use the command pd.read_csv to import .csv files

wines = pd.read_csv(filepath_or_buffer = "C:/Users/MatthiasQ.MATTQ/Downloads/wine-reviews/winemag-data-130k-v2.csv",
                    nrows = 130000,
                    sep = ',')

#To determine the type of variable in use
print(wines.dtypes)

