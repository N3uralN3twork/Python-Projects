#Set the working directory
import os
abspath = os.path.abspath('C:/Users/MatthiasQ.MATTQ/Desktop/Python Projects/Sci-kit Learn')
os.chdir(abspath)
#Import the necessary libraries
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


"Grabbing sample data sets"
df = sns.load_dataset('iris')
ts = pd.read_csv("Datasets/Monthly Sunspots.csv", sep = ",")




plt.style.use("ggplot")

"Histogram"
sns.distplot(df["petal_width"])
plt.show()

"Bar Chart"
sns.countplot(x = df["species"])
plt.show()

"Box plot"
sns.boxplot(x = df["species"], y = df["sepal_length"]).set_title("Boxplot of Sepal Length by Species")
plt.show()

"Scatterplot"
sns.scatterplot(x = df["petal_length"], y = df["petal_width"], hue = df["species"])
plt.show()

"Relationship plot"
sns.relplot(x = "petal_length", y = "petal_width",
            data = df, hue = "species", size = "sepal_length")
plt.show()

"Time-Series plot"
ts.dtypes
#Convert Month object to datetime
ts["Month"] = pd.to_datetime(ts["Month"], format = "%Y-%m")

#Create the time-series plot
sns.lineplot(x = "Month", y = "Sunspots", data = ts)
plt.show()