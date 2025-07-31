import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("telecom_customer_churn.csv")

df

df.info()

df.describe()

df.isnull().sum()

df.duplicated().sum()

df.columns

X = df.drop(['Customer ID','Churn Category'],axis=1)

y = df['Churn Category']

y.value_counts()

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()

X_cleaned = X.copy()
y_cleaned = y.dropna()

X_cleaned = X_cleaned.loc[y_cleaned.index]


X_resampled, y_resampled = ros.fit_resample(X_cleaned, y_cleaned)



y.value_counts()

from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder()

X = oe.fit_transform(X)

from sklearn.model_selection import train_test_split

X_resampled_encoded = oe.fit_transform(X_resampled)

X_train, X_test, y_train, y_test = train_test_split(X_resampled_encoded, y_resampled, random_state=2529)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))

