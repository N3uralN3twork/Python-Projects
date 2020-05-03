









from pycm import ConfusionMatrix
cm = ConfusionMatrix(actual_vector = y_test,
                     predict_vector = nb_preds)
print(cm)





