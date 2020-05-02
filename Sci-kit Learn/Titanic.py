import os
abspath = os.path.abspath('C:/Users/MatthiasQ.MATTQ/Desktop/Python Projects/Sci-kit Learn')
os.chdir(abspath)
import pandas as pd
import re

df = pd.read_csv("TitanicTrain.csv")

#To extract the title from each name:
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

df["Title"] = df["Name"].apply(get_title)

df["Title"].value_counts()
#Drop rows with "Major, Mlle, Col, Lady, Countess, Capt, Ms, Sir, Mme, Don, Jonkheer"

to_drop = ["Major", "Mlle", "Col", "Lady", "Countess", "Capt", "Ms", "Sir", "Mme", "Don", "Jonkheer"]
df = df[~df["Title"].isin(to_drop)]

