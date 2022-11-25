# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values from dataframe and apply label encoder.
3. Apply decision tree classifier on the dataframe.
4. obtain the value of accuracy and data prediction. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: M.Suwetha
RegisterNumber:  212221230112
*/
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
###dataframe
![d1](https://user-images.githubusercontent.com/94165336/203972586-29bd2dd4-452e-45b6-a115-e0f735bf529f.png)
###null values
![d2](https://user-images.githubusercontent.com/94165336/203972604-050196c5-632a-42d5-92b5-6c209736b282.png)
![d3](https://user-images.githubusercontent.com/94165336/203972622-44037f55-d34e-4b6a-894b-910fb13734a7.png)
###Label encoder
![d4](https://user-images.githubusercontent.com/94165336/203972642-e0470b31-5d4e-4f8d-bc9f-a27ff427af3a.png)
![d5](https://user-images.githubusercontent.com/94165336/203972661-a0949793-925b-4513-af1c-908a7fda1a56.png)
###Accuracy
![d6](https://user-images.githubusercontent.com/94165336/203972683-97847a37-111d-485f-a65f-ee87c6bbd163.png)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
