
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv('/content/StudentPerformanceFactors.csv')
data.head()

data.shape

sns.boxplot(data)

data.dtypes

data.isna().sum()

data['Teacher_Quality'] = data['Teacher_Quality'].fillna(data['Teacher_Quality'].mode()[0])

data['Parental_Education_Level'] = data['Parental_Education_Level'].fillna(data['Parental_Education_Level'].mode()[0])

data['Distance_from_Home'] = data['Distance_from_Home'].fillna(data['Distance_from_Home'].mode()[0])

data.hist()

data.isna().sum()

data.hist(figsize=(15, 10))

data.dtypes

# Encoding number_of_projects using Label Encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data['Parental_Involvement'].nunique()

data['Parental_Involvement']=le.fit_transform(data['Parental_Involvement'])
data.head()

data['Access_to_Resources'].nunique()

data['Access_to_Resources']=le.fit_transform(data['Access_to_Resources'])
data.head()

data['Extracurricular_Activities'].nunique()

data['Motivation_Level'].nunique()

data['Motivation_Level']=le.fit_transform(data['Motivation_Level'])
data.head()

data['Internet_Access'].nunique()

data['Internet_Access']=le.fit_transform(data['Internet_Access'])
data.head()

data['Family_Income'].nunique()

data['Family_Income']=le.fit_transform(data['Family_Income'])
data.head()

data['Teacher_Quality'].nunique()

data['Teacher_Quality']=le.fit_transform(data['Teacher_Quality'])
data.head()

data['School_Type'].nunique()

data['School_Type']=le.fit_transform(data['School_Type'])
data.head()

data['Peer_Influence'].nunique()

data['Peer_Influence']=le.fit_transform(data['Peer_Influence'])
data.head()

features=[
    'Hours_Studied',
    'Attendance',
    'Previous_Scores',
    'Sleep_Hours'
]
target='Exam_Score'

X=data[features]
y=data[target]

sns.histplot(data['Exam_Score'])
plt.show()

corr=data[
    [
        'Hours_Studied',
        'Attendance',
        'Previous_Scores',
        'Sleep_Hours',
        'Exam_Score'
    ]
].corr()
corr

sns.heatmap( corr,annot=True ,cmap='Blues' )

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(random_state=42)
rf.fit(X_train,y_train)

pred=rf.predict(X_test)
from sklearn.metrics import r2_score
r2_score(y_test,pred)

"""Hyper parameter processing"""

from sklearn.model_selection import cross_val_score
scores=cross_val_score(rf,X,y,cv=5,scoring='r2')

scores

from sklearn.model_selection import GridSearchCV
param_grid = { 'n_estimators': [100,200,300],
              'max_depth': [3,5,10,None],
              'min_samples_split': [2,5,10],
              }

grid = GridSearchCV(RandomForestRegressor(random_state=42),param_grid,cv=5,scoring='r2')

grid.fit(X_train,y_train)

grid.best_params_

best_model=grid.best_estimator_

pred=best_model.predict(X_test)

r2_score(y_test,pred)

"""Pickle"""

import joblib
joblib.dump(best_model,"student.pkl")

hours=5
attendence=75
previous=65
sleep=6
sample=pd.DataFrame(
    {
        'Hours_Studied':[hours],
        'Attendance':[attendence],
        'Previous_Scores':[previous],
        'Sleep_Hours':[sleep]
    }
)

predicted_score=best_model.predict(sample)[0]
predicted_score

from google.colab import userdata
key=userdata.get('GoogleAPI2')

import google.generativeai as genai
genai.configure(api_key=key)

model=genai.GenerativeModel( 'gemini-2.5-flash')

!pip install -q -U google-generativeai

prompt=f"""
You are an export academic mentor.
Student Details :
Hours Studied : {hours}
Attendence: {attendence}
Previous Score {previous}
Sleep Hours : {sleep}
Predicted Exam Score : {predicted_score:.2f}
Provide:
1.Study Improvement Plan
2.Daily Study Schedule
3.Revision Strategy
4.Exam Preparation Advice
Use Bullet Points.
"""
response=model.generate_content(prompt)
print(response.text)

!pip install streamlit

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# import streamlit as st
# import pandas as pd
# import joblib
# import google.generativeai as genai
# model=joblib.load("student.pkl")
# import os
# # genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
# gemini=genai.GenerativeModel(
#     'gemini-2.5-flash'
# )
# st.title("Student Performance Predictor")
# hours=st.slider(
#     "Hours Studied",
#     0,
#     12,
#     5)
# attendence=st.slider(
#     "Attendence%",
#     0,
#     100,
#     75)
# previous=st.slider(
#     "Previous Score",
#     0,
#     100,
#     65)
# sleep=st.slider(
#     "Sleep Hours",
#     3,
#     10,
#     7)
# if st.button("Predict Exam Score"):
#     sample=pd.DataFrame(
#         {
#             'Hours_Studied':[hours],
#             'Attendance':[attendence],
#             'Previous_Scores':[previous],
#             'Sleep_Hours':[sleep]
#         }
#     )
#     score=model.predict(sample)[0]
#     st.success(f"Predicted Exam Score:{score:.2f}")
#     prompt=f"""
#     You are an expert academic mentor
#     Student Details:
#     Hours Studied:{hours}
#     Attendence:{attendence}
#     previous_score:{previous}
#     Sleep Hours:{sleep}
#     Predicted Exam Score:{score:.2f}
#     Provide:
#     1. Study Improvement Plan
#     2. Daily Study Schedule
#     3. Revision Strategy
#     4. Time Management Tips
#     5. Exam Preparation
#     Use bullet points
#     """
#     response=gemini.generate_content(prompt)
#     st.subheader(
#         "AI study coach"
#     )
#     st.write(response.text)

!pip install -q pyngrok
from pyngrok import ngrok
token=userdata.get('ngrok')
ngrok.set_auth_token(token)
public_url=ngrok.connect(8501)
print(public_url)

!streamlit run app.py &>/content/logs.txt &

!nps localtunnel --port 8501
