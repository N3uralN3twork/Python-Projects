#Added 27 January 2020
import os
abspath = os.path.abspath('C:/Users/MatthiasQ.MATTQ/Desktop/Python Projects/Sci-kit Learn')
os.chdir(abspath)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
df = pd.read_csv("Datasets/mushrooms.csv")


def dummyEncode(data):
    columnsToEncode = list(df.select_dtypes(include=['category', 'object']))
    LE = LabelEncoder()
    for feature in columnsToEncode:
        try:
            df[feature] = LE.fit_transform(df[feature])
        except:
            print('Error encoding ' + feature)
    return df


data = dummyEncode(data = df)

y = df['class']
x = df.drop(["class"], axis = 1)

scaler = StandardScaler()
x = scaler.fit_transform(x)

"Principal Component Analysis"
pca = PCA(n_components = 17)
pca.fit_transform(x)

explained_variance = pca.explained_variance_
print(explained_variance)

with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(17), explained_variance, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=4)

from sklearn.linear_model import LogisticRegression

model_LR= LogisticRegression()

model_LR.fit(X_train, y_train)

y_prob = model_LR.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities
y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.
model_LR.score(X_test, y_pred)

from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')