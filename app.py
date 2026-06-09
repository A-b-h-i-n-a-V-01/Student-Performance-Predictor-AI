import streamlit as st
import pandas as pd
import joblib
import google.generativeai as genai
import os

model=joblib.load("student.pkl")

# Securely load Gemini API Key
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

if api_key:
    genai.configure(api_key=api_key)
    gemini = genai.GenerativeModel('gemini-2.5-flash')
else:
    st.sidebar.warning("⚠️ GEMINI_API_KEY not found in environment variables or Streamlit secrets.")
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
    if api_key:
        genai.configure(api_key=api_key)
        gemini = genai.GenerativeModel('gemini-2.5-flash')
    else:
        st.info("Please enter your Gemini API key in the sidebar to enable the AI Study Coach feature.")
        gemini = None
st.title("Student Performance Predictor")
hours=st.slider(
    "Hours Studied",
    0,
    12,
    5)
attendence=st.slider(
    "Attendence%",
    0,
    100,
    75)
previous=st.slider(
    "Previous Score",
    0,
    100,
    65)
sleep=st.slider(
    "Sleep Hours",
    3,
    10,
    7)
if st.button("Predict Exam Score"):
    sample=pd.DataFrame(
        {
            'Hours_Studied':[hours],
            'Attendance':[attendence],
            'Previous_Scores':[previous],
            'Sleep_Hours':[sleep]
        }
    )
    score=model.predict(sample)[0]
    st.success(f"Predicted Exam Score:{score:.2f}")
    prompt=f"""
    You are an expert academic mentor
    Student Details:
    Hours Studied:{hours}
    Attendence:{attendence}
    previous_score:{previous}
    Sleep Hours:{sleep}
    Predicted Exam Score:{score:.2f}
    Provide:
    1. Study Improvement Plan
    2. Daily Study Schedule
    3. Revision Strategy
    4. Time Management Tips
    5. Exam Preparation
    Use bullet points
    """
    if gemini:
        response=gemini.generate_content(prompt)
        st.subheader(
            "AI study coach"
        )
        st.write(response.text)
    else:
        st.info("⚠️ Enter a Gemini API Key in the sidebar to generate custom study plans and advice.")
