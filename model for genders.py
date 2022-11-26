from tkinter import ttk
from tkinter import messagebox
from tkinter import *
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# reading data
penguins_df = pd.read_csv("penguins.csv")
penguins_df=penguins_df.dropna()
# print(penguins_df.count())

# penguins_df=penguins_df[penguins_df.]
# # changing data types from object to category
penguins_df["gender"] = penguins_df["gender"].astype('category')
penguins_df["species"] = penguins_df["species"].astype('category')
#
# encoding categorical data
enc1 = OrdinalEncoder()
enc2 = OrdinalEncoder()
penguins_df["gender_cat"] = enc1.fit_transform(penguins_df[["gender"]])
penguins_df["y"] = enc2.fit_transform(penguins_df[["species"]])



# # dropping unused columns
# # penguins_df.drop("gender", inplace=True, axis=1)
Y = penguins_df["gender_cat"]
X=penguins_df.drop("gender", axis=1)
X=X.drop("gender_cat", axis=1)
X=X.drop("species", axis=1)
print(X.count())
X_train, X_Test, Y_train, Y_Test = train_test_split(X,Y,test_size=0.2)
# #
# #
# training the model
C=0.05
svc = svm.SVC(kernel='linear', C=C).fit(X_train, Y_train)
svm.SVC.predict()

accuracy = svc.score(X_Test, Y_Test)
print(svc.predict(svm, X_train[0]))
print(' SVC accuracy: ' + str(accuracy))
