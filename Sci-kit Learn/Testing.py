pd.set_option('display.max_columns', 10)

from sklearn.preprocessing import OrdinalEncoder

X = df["Seniority"]
Seniority = df["Seniority"]

def OrdinalEncode(data, OrdinalColumns: list):
    #You have to make a list of your ordinal variables first
    OE = OrdinalEncoder()
    for feature in OrdinalColumns:
        try:
            data[feature] = OE.fit_transform(data[feature])
        except:
            print('Error encoding ' + feature)
    return data
OrdinalEncode(df, Categorical)

df[Categorical] = OrdinalEncoder().fit_transform(df[Categorical])





