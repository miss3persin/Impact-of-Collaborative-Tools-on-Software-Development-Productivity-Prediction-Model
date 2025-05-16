import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved Linear Regression model
model = joblib.load("linear_regression_productivity_model.pkl")

st.title("Predictive Model for Software Development Productivity")
st.write("Enter the collaborative tool usage metrics to predict productivity score.")

# User inputs
num_commits = st.slider("Number of commits", min_value=0, max_value=100, value=20)
issues_resolved = st.slider("Number of issues resolved", min_value=0, max_value=50, value=10)
slack_msgs = st.slider("Number of Slack messages", min_value=0, max_value=200, value=50)
jira_tickets = st.slider("Number of Jira tickets handled", min_value=0, max_value=50, value=8)
team_velocity = st.slider("Team velocity (story points)", min_value=0, max_value=100, value=30)

# Prepare input data as a DataFrame
input_data = pd.DataFrame({
    "num_commits": [num_commits],
    "issues_resolved": [issues_resolved],
    "slack_msgs": [slack_msgs],
    "jira_tickets": [jira_tickets],
    "team_velocity": [team_velocity]
})

# Predict productivity
predicted_score = model.predict(input_data)[0]

st.subheader("Predicted Productivity Score")
st.write(f"**{predicted_score:.2f}** (out of 100)")

# Show input values bar chart
st.subheader("Input Values Overview")
st.bar_chart(input_data.T)

