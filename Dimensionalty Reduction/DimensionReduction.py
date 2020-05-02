###############################################################################
###                     1.  Define Working Directory                        ###
###############################################################################
#Source: https://www.kaggle.com/arthurtok/interactive-intro-to-dimensionality-reduction

import os
abspath = os.path.abspath("C:/Users/MatthiasQ.MATTQ/Desktop/Python Projects/Dimensionalty Reduction")
os.chdir(abspath)

###############################################################################
###                     2. Import Libraries                                 ###
###############################################################################
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# Import the 3 dimensionality reduction methods
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


"Curse of Dimensionality"
#Too many variables.
#We project our high-dimensional data to a lower dimension while keeping relevant
#information.

"Get the dataset"
train = pd.read_csv("train.csv")

#Save the labels
target = train["label"]
train = train.drop(["label"], axis = 1)

#########################################################
#             A. Principal Components Analysis (PCA)    #
#########################################################
"It projects the original features of the data onto a smaller set of features (subspace)"
"It tries to find the directions that contain the largest spread of information relative" \
"to all of the other data."

"""""Calculating the Eigenvectors"
X = train.values
X_std = StandardScaler().fit_transform(X)
#Calculating Eigenvectors and Eigenvaluess of Cov Matrix
mean_vec = np.mean(X_std, axis = 0)
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
# Create a list of (eigenvalue, eigenvector) tuples
eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

#Sort eigenvalue, eigenvector from high to low
eig_pairs.sort(key = lambda x:x[0], reverse = True)

#Calculate the Explained Variance from the Eigenvalues
tot = sum(eig_vals)
var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance
cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance

print(var_exp)"""

"Don't forget to scale data first!"

X = train.values
X_std = StandardScaler().fit_transform(X)

#Call PCA with 5 Components
pca = PCA(n_components = 300)
pca.fit(X_std)
sum(pca.explained_variance_ratio_) #I think 94% is good enough, we reduced the size by at least 400 variables
X_PCA_300 = pca.fit_transform(X_std)

#TO find the cumulative explained variance:
sum(pca.explained_variance_ratio_)



#LDA Implementation
lda = LDA(n_components = 300)
lda.fit(X_std, y = target)
sum(lda.explained_variance_ratio_)
X_LDA_300 = lda.fit_transform(X_std, target.values)



"Store Results to a DataFrame:"
result = []
for i in range(1, 301):
    result.append("PC" + str(i))

Scree = pd.DataFrame({"PrinComp": result})
Scree["VarExplained"] = pca.explained_variance_ratio_


"Scree Plot"
ax = sns.barplot(x = "PrinComp", y = "VarExplained",
            data = Scree, color = 'c')
ax.set(xlabel = "Principal Component",
       ylabel = "Variance Explained")


