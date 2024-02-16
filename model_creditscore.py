import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
import joblib


df= pd.read_csv("credit.csv")

data=df.copy()

data["Age"] = data["Age"].astype(int)
data["Num_of_Loan"] = data["Num_of_Loan"].astype(int)
data["Num_Bank_Accounts"] =data["Num_Bank_Accounts"].astype(int)
data["Credit_History_Age"] = data["Credit_History_Age"].astype(int)
data["SSN"] = data["SSN"].astype(int)
data["Num_Credit_Inquiries"] = data["Num_Credit_Inquiries"].astype(int)
data["Num_Credit_Card"] = data["Num_Credit_Card"].astype(int)
data["Delay_from_due_date"] = data["Delay_from_due_date"].astype(int)
data["Num_of_Delayed_Payment"] = data["Num_of_Delayed_Payment"].astype(int)
data["Changed_Credit_Limit"] = data["Changed_Credit_Limit"].astype(int)


le= LabelEncoder()
train=data.drop(["ID","Customer_ID","Month","Name","Type_of_Loan","Occupation",
                 "Amount_invested_monthly","Credit_Utilization_Ratio"],axis=1)



train["Credit_Score"] = le.fit_transform(train["Credit_Score"])
train["Payment_Behaviour"] = le.fit_transform(train["Payment_Behaviour"])
train["Payment_of_Min_Amount"] = le.fit_transform(train["Payment_of_Min_Amount"])
train["Credit_Mix"] = le.fit_transform(train["Credit_Mix"])

X = train.drop(["Credit_Score","Monthly_Inhand_Salary"],axis=1)
Y = pd.DataFrame(train["Credit_Score"])



smote = SMOTE()
X,Y = smote.fit_resample(X,Y)



scaler_x = MinMaxScaler()
X_scale= scaler_x.fit_transform(X)






Y= np.squeeze(Y)

x_train,x_test,y_train,y_test = train_test_split(X_scale,Y, test_size = 0.2, random_state=42)



rf_cls =RandomForestClassifier()

model_rf = rf_cls.fit(x_train,y_train)



#y_pred_rf = model_rf.predict(x_test)

#accuracy_rf = accuracy_score(y_test,y_pred_rf)

#accuracy_rf



#filename="model.pickle"

#with open(filename,"wb") as file:
  #pickle.dump(model_rf,file)

pipeline = make_pipeline(MinMaxScaler(),RandomForestClassifier())

model = pipeline.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test,y_pred)

joblib.dump(model,'filename.joblib')

#model = joblib.load('filename.joblib')