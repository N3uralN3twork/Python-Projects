"Merging and Joining Dataframes with Pandas"
#Source: https://www.shanelynn.ie/merge-join-dataframes-python-pandas-index-1/

#I am learning how to do the various joins after realizing I didn't know what they
#were during my phone interview yesterday.

#We have 3 datasets:
#user_usage.csv = Contains users monthly mobile usage statistics
#user_device.csv = Contains details on individual usage of the app
#android_devices.csv = Contains device and manufacturer data


########################################################
###             Load in the 3 datasets               ###
########################################################
import pandas as pd
import numpy as np

user_usage = pd.read_csv("UserUsage.csv")
user_device = pd.read_csv("UserDevice.csv")
android_devices = pd.read_csv("AndroidDevices.csv")


#Shared Attributes:
"There are linking attributes between the datasets:" \
"use-id is shared between user_usage and usage_device"
"device is share between android_devices and user_device


#Goal:
"""Figure out usage statistics between different devices"""

########################################################
###             Merging DataFrames                   ###
########################################################
"""1. For each row in the user_usage dataset:
    Make a column that contains the device code and platform
    Take device columns and find retail branding and model in the devices dataset
    Look at different statistics by the device manufacturer"""

user_usage.head(5)
user_device.head(5)

result = pd.merge(user_usage, #Main dataset you want to add to
                  user_device[["use_id", "platform", "device"]], #Extra columns you want to add
                  on = "use_id") #What they both have in common
result.head(4)

"The Merge Command:"
#It takes a left dataframe(user_usage), a right dataframe(user_device),
#and then a column that they both have in common

"Inner, Left, and Right merge types:"
"""Inner Merge: keeps only common values from both datasets
        Only use_id that are common between two datasets are kept
   Left Merge/Join: Keep every row in the first dataset, 
                    but if missing values are present in the second dataset, add NaN values as a result
   Right Merge/Join: Keep every row in the second dataset,
                    but if missing values are present in the first dataset, add NaN values as a result
   Outer Merge/Join: Returns all the rows from both datasets using matches where possible, everything else is NaN
   """

"Left Join Example:"

result2 = pd.merge(user_usage,
                   user_device[["use_id", "platform", "device"]],
                   on = "use_id",
                   how = "left") #Where you specify the type of join to use
#It should keep every row from user_usage
len(user_usage)
len(result2)


"Rigt Join Example:"
result3 = pd.merge(user_usage,
                   user_device[["use_id", "platform", "device"]],
                   on = "use_id",
                   how = "right")
#It should keep every row from user_device
len(user_device)
len(result3)

"Full Outer Join Example:"
#Every row is retained from both datasets, every missing row is filled with NaN
result3 = pd.merge(user_usage,
                   user_device[["use_id", "platform", "device"]],
                   on = "use_id",
                   how = "outer")
len(result3)




result = pd.merge(user_usage,
                  user_device[["use_id", "platform", "device"]],
                  on = "use_id",
                  how = "left")

android_devices.rename(columns={"Retail Branding": "manufacturer"}, inplace=True)

result4 = pd.merge(result,
                   android_devices[["manufacturer", "Model"]],
                   left_on = "device",
                   right_on = "Model",
                   how = "left")

len(result4)

#I think that's enough for right now.







