
# 📈 Predictive Model for Evaluating the Impact of Collaborative Tools on Software Development Productivity

## 🧠 Project Overview

This project is a machine learning-based system designed to **analyze and predict how collaborative tools** (such as GitHub, Jira, and Slack) **influence software development productivity**. It also includes a comparison with the traditional **COCOMO model** to benchmark effort estimation.

The goal is to provide data-driven insights for decision-makers in software teams, helping them understand how their communication and collaboration habits affect productivity.

---

## 🚀 Features

* ✅ **Data Simulation**: Realistic, large-scale synthetic dataset mimicking real-world team behavior (10,000+ samples).
* ✅ **Preprocessing & Feature Engineering**: Transformation of raw collaboration metrics into model-ready features.
* ✅ **Machine Learning Models**: Trained and compared Linear Regression, Random Forest, and Gradient Boosting models (with tuned and default versions).
* ✅ **Hyperparameter Tuning**: Optimized Random Forest and Gradient Boosting with `GridSearchCV`.
* ✅ **Effort Estimation**: Used the COCOMO model (organic mode) to estimate project effort based on KLOC (thousands of lines of code).
* ✅ **Effort Comparison**: Converted ML-predicted productivity into estimated effort and compared it with COCOMO's estimate.
* ✅ **Interactive Streamlit Dashboard**: Easy-to-use dashboard to:

  * Upload or input data
  * View model predictions
  * Enter KLOC and view COCOMO estimates
  * Compare COCOMO vs ML-based effort
  * Visualize results with interactive charts

---

## ⚙️ How It Works

### 🔢 Inputs

* Collaboration tool usage data (simulated):

  * GitHub: commits, PRs, code reviews
  * Jira: issues resolved, time to resolve
  * Slack: messages sent, active days
* Productivity score (target): scaled between 0–100

### 🧮 Processing Steps

1. **Data Cleaning & Scaling** (with `StandardScaler`)

2. **Model Training**:

   * Linear Regression (baseline)
   * Random Forest (default and tuned)
   * Gradient Boosting (default and tuned)

3. **Model Evaluation**:

   * Metrics: Mean Squared Error (MSE) and R² Score
   * Best Result: Linear Regression with R² ≈ 0.15

4. **Effort Estimation Using COCOMO**:

   * Formula: `Effort = a * (KLOC)^b`
   * Organic Mode constants: `a = 2.4`, `b = 1.05`

5. **Effort from ML**:

   * Estimate based on: `Effort = Output / Productivity`
   * Output assumed: 1000 LOC or story points

6. **Comparison**:

   * Bar chart visualization (COCOMO vs ML-estimated effort)
   * Interpretation messages shown based on comparison

---

## 💻 Getting Started

### 🔧 Requirements

To run locally:

* Python 3.8+
* Required packages:

  ```bash
  pip install streamlit pandas scikit-learn plotly joblib
  ```

### 📦 Running the App

```bash
streamlit run app.py
```

Or deploy free via **[Streamlit Cloud](https://streamlit.io/cloud)**.

---

## 📊 Sample Results

**Best Model: Linear Regression**

* MSE: 25.26
* R²: 0.15

**Interpretation**: The ML model captures general trends but is not perfect (low R²). However, it still offers practical value for productivity comparison and insight.

---

## 🧠 Why COCOMO?

COCOMO (Constructive Cost Model) is a traditional effort estimation model based on code size (KLOC). We included it to:

* Provide a **benchmark** against our modern ML-based productivity predictions
* Help quantify ML results in terms of **person-months**
* Visualize the **gap between theoretical vs observed team behavior**

---

## 📂 Project Structure

```
📁 project/
│
├── app.py                  # Streamlit dashboard app
├── dataset.csv             # Simulated dataset
├── best_model.pkl          # Best model (Linear Regression)
├── predictive_model.ipynb  # Code file for training model
├── requirements.txt        # Python dependencies
└── README.md               # Project description
```

---

## 📌 Future Improvements

* Integrate live APIs from GitHub, Jira, Slack
* Improve model accuracy with time-series or team-specific features
* Add authentication and team-by-team breakdown in dashboard
* Consider neural networks for more advanced predictions
