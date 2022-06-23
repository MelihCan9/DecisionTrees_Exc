import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('loan_data.csv')

from sklearn.model_selection import train_test_split

X = df.drop('not.fully.paid', axis=1)
y = df['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(X_train, y_train)

predict = dtc.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test,predictions))
print("***************************************")
print(classification_report(y_test,predictions))


from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)

print(confusion_matrix(y_test, rfc_pred))
print("***************************************")
print(classification_report(y_test,rfc_pred))