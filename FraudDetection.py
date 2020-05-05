import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('creditcard.csv')
fraud=df.loc[df['Class']==1]
normal=df.loc[df['Class']==0]
x=df.iloc[:,:-1]
y=df['Class']
sns.relplot(x='Amount',y='Time',hue='Class',data=df)
plt.show()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=20)
rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)

from sklearn.metrics import accuracy_score,classification_report
print(accuracy_score(y_test,y_pred)*100)
print(classification_report(y_test,y_pred))
