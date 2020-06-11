import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

test = df


Nominal = list(df.select_dtypes(include=['object', 'category']).columns)
# Remove the ordinal and response variables, one by one
Nominal.remove("Income")
Nominal.remove("Seniority")
# One-hot encode the nominal variables
test = pd.get_dummies(test, drop_first = True, columns = Nominal)


def OrdinalEncode(data, OrdinalColumns: list):
    #You have to make a list of your ordinal variables first
    OE = OrdinalEncoder()
    for feature in OrdinalColumns:
        try:
            data[feature] = OE.fit_transform(data[feature])
        except:
            print('Error encoding ' + feature)
    return data

Ordinal = ["Seniority"] #This is where you make a list of your ordinal features.
test = OrdinalEncode(data=test, OrdinalColumns=Ordinal)

OE = OrdinalEncoder()
OE.fit_transform(df[["Seniority"]])