# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset Salary.csv using pandas and view the first few rows.

2.Check dataset information and identify any missing values.

3.Encode the categorical column "Position" into numerical values using LabelEncoder.

4.Define feature variables x as "Position" and "Level", and target variable y as "Salary".

5.Split the dataset into training and testing sets using an 80-20 split.

6.Create a DecisionTreeRegressor model instance.

7.Train the model using the training data.

8.Predict the salary values using the test data.

9.Evaluate the model using Mean Squared Error (MSE) and R² Score.

10.Use the trained model to predict salary for a new input [5, 6].

## Program:
```


import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

```

## Output:
### Reading of dataset
![image](https://github.com/user-attachments/assets/2c684145-2e4a-4df0-8caa-530fdefffc25)

### Value of df.head()
![Screenshot 2025-04-27 123309](https://github.com/user-attachments/assets/93155c59-f9bd-409a-be09-17164fe3dd00)

###  Value of df.isnull().sum()
![Screenshot 2025-04-27 123253](https://github.com/user-attachments/assets/c045fc3b-d400-4d52-b692-dad1b9fc72c6)
### df.info()
![Screenshot 2025-04-27 123258](https://github.com/user-attachments/assets/838e022f-a851-4365-9517-0f661420a18c)
### Data after encoding calculating Mean Squared Error
![Screenshot 2025-04-27 123319](https://github.com/user-attachments/assets/03efd50f-7da7-4be8-b8bd-d129199f29de)
### R2 value
![Screenshot 2025-04-27 123324](https://github.com/user-attachments/assets/591470ae-bdb6-4fe9-a79c-1a1a4e69f8c7)
### Model prediction with [5,6] as input
![image](https://github.com/user-attachments/assets/d54f4baa-3e1f-4bea-8700-96be391a80a2)





## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
