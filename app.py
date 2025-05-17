import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load("linear_regression_productivity_model.pkl")

st.title("Predictive Model for Software Development Productivity")
st.write("Enter the collaborative tool usage metrics to predict productivity score.")

# User inputs
num_commits = st.slider("Number of commits", 0, 100, 20)
issues_resolved = st.slider("Number of issues resolved", 0, 50, 10)
slack_msgs = st.slider("Number of Slack messages", 0, 200, 50)
jira_tickets = st.slider("Number of Jira tickets handled", 0, 50, 8)
team_velocity = st.slider("Team velocity (story points)", 0, 100, 30)

# Prepare input dataframe
input_data = pd.DataFrame({
    "num_commits": [num_commits],
    "issues_resolved": [issues_resolved],
    "slack_msgs": [slack_msgs],
    "jira_tickets": [jira_tickets],
    "team_velocity": [team_velocity]
})

# Prediction
predicted_score = model.predict(input_data)[0]

st.subheader("Predicted Productivity Score")
st.write(f"### {predicted_score:.2f} (out of 100)")

st.subheader("Input Values Overview")
st.bar_chart(input_data.T)

# COCOMO Effort Estimation
st.subheader("COCOMO Effort Estimation (Organic Mode)")

# User input for estimated project size (KLOC)
kloc = st.number_input("Estimated project size (in KLOC)", min_value=1.0, max_value=1000.0, value=50.0, step=1.0)

# COCOMO basic estimation formula
def cocomo_estimate(KLOC):
    a, b = 2.4, 1.05  # Organic mode constants
    effort = a * (KLOC ** b)  # Person-months
    return effort

# Get COCOMO effort
estimated_effort = cocomo_estimate(kloc)
st.write(f"📊 **COCOMO Estimated Effort:** {estimated_effort:.2f} person-months")

# --- ML to Effort Conversion ---
st.subheader("Effort Estimate Based on Predicted Productivity")

# Assume some constant output
estimated_output = 1000  # e.g., LOC, story points

# Prevent divide-by-zero issues
safe_productivity = max(predicted_score, 1)
ml_effort_estimate = estimated_output / safe_productivity

st.write(f"🧠 **Estimated Effort from ML Prediction:** {ml_effort_estimate:.2f} person-months (assuming output of 1000 units)")

# Comparison
st.subheader("Comparison: ML Productivity vs COCOMO Effort")
if ml_effort_estimate > estimated_effort:
    st.warning("🔸 ML-based estimate suggests *more effort* may be required than COCOMO predicts.")
elif ml_effort_estimate < estimated_effort:
    st.success("✅ ML-based estimate suggests *higher productivity* than COCOMO predicts.")
else:
    st.info("⚖️ ML-based and COCOMO estimates are about the same.")

# Optional info box
with st.expander("ℹ️ What is COCOMO and why compare it?"):
    st.markdown("""
    The **COCOMO (Constructive Cost Model)** is a traditional method to estimate software effort based on project size (in KLOC).
    
    On the other hand, our **ML model** uses real-time collaboration metrics to predict productivity.  
    By assuming a fixed output (like 1000 LOC), we estimate how much effort that productivity level might require — and compare it to what COCOMO says.

    This lets us bridge **traditional estimation** and **modern team behavior-based prediction**.
    """)


# Visualization of the comparison
import plotly.graph_objects as go

st.subheader("📉 Visual Comparison of Effort Estimates")

effort_data = {
    "COCOMO Estimate": estimated_effort,
    "ML-Based Estimate": ml_effort_estimate
}

fig = go.Figure(data=[
    go.Bar(
        x=list(effort_data.keys()),
        y=list(effort_data.values()),
        text=[f"{v:.2f}" for v in effort_data.values()],
        textposition='auto',
        marker_color=['#636EFA', '#EF553B']
    )
])

fig.update_layout(
    yaxis_title="Effort (Person-Months)",
    xaxis_title="Estimation Method",
    title="Effort Estimates: COCOMO vs ML Productivity",
    template="plotly_dark",
    height=400
)

st.plotly_chart(fig, use_container_width=True)


# --- Additional Visualizations ---

# Load the full dataset (you need to upload or include it in the repo)
@st.cache_data
def load_dataset():
    return pd.read_csv("simulated_productivity_dataset.csv")

df = load_dataset()

st.subheader("Feature Correlation Heatmap")
corr = df.corr()

fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Scatter plots of each feature vs productivity
st.subheader("Feature vs Productivity Scatter Plots")
for feature in input_data.columns:
    fig, ax = plt.subplots()
    ax.scatter(df[feature], df["productivity_score"], alpha=0.3)
    ax.axvline(x=input_data[feature][0], color='red', linestyle='--', label='Your input')
    ax.set_xlabel(feature)
    ax.set_ylabel("Productivity Score")
    ax.legend()
    st.pyplot(fig)

# Simulate past predictions history
st.subheader("Simulated Productivity Predictions History")

history_df = df.sample(30).copy()
history_df["predicted_score"] = model.predict(history_df[input_data.columns])

fig, ax = plt.subplots()
ax.plot(history_df.index, history_df["productivity_score"], label="Actual Productivity")
ax.plot(history_df.index, history_df["predicted_score"], label="Predicted Productivity", linestyle="--")
ax.set_xlabel("Sample Index")
ax.set_ylabel("Productivity Score")
ax.legend()
st.pyplot(fig)

# Feature importance from Linear Regression coefficients
st.subheader("Feature Importance (Linear Regression Coefficients)")

coefs = model.coef_
features = input_data.columns

coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": coefs
}).sort_values(by="Coefficient", key=abs, ascending=False)

fig, ax = plt.subplots()
ax.barh(coef_df["Feature"], coef_df["Coefficient"], color="skyblue")
ax.set_xlabel("Coefficient Value")
ax.set_title("Linear Regression Feature Importance")
st.pyplot(fig)
