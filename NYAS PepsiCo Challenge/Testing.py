from sklearn import datasets

breast_cancer = datasets.load_breast_cancer()
X_cancer = breast_cancer.data
y_cancer = breast_cancer.target

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

models1 = {
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier()
}

params1 = {
    'ExtraTreesClassifier': { 'n_estimators': [16, 32] },
    'RandomForestClassifier': [
        { 'n_estimators': [16, 32] },
        {'criterion': ['gini', 'entropy'], 'n_estimators': [8, 16]}],
    'AdaBoostClassifier':  { 'n_estimators': [16, 32] },
    'GradientBoostingClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] }
}

helper1 = EstimatorSelectionHelper(models1, params1)
helper1.fit(X_cancer, y_cancer, scoring="f1", n_jobs=2)
results = helper1.score_summary()






from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=2)
imputer.fit_transform(clean_df)












