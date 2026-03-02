import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib   

#Load the data
df=pd.read_csv('employee_data.csv')

#Convert words to numbers
df['OverTime']=df['OverTime'].map({'Yes':1,'No':0})

#Set features x and y
X=df[['Age','YearsAtCompany','MonthlyIncome','OverTime']]
y=df['Attrition']

#Split data into test and train sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#build and train model
model=RandomForestClassifier(n_estimators=200,random_state=42)
model.fit(X_train,y_train)

#evaluate
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Model Accuracy:{accuracy*100:.2f}%")

#Save the model to a file
joblib.dump(model, 'attrition_model.pkl')
print("✅ Model saved as attrition_model.pkl")