It seems like I can’t do more advanced data analysis right now. Please try again later.

However, I can provide you with the complete `README.md` content here so you can copy and paste it directly into your project:

---

# 📈 Predictive Model for Evaluating the Impact of Collaborative Tools on Software Development Productivity

## 🧠 Project Overview

This project is a machine learning-based system designed to **analyze and predict how collaborative tools** (such as GitHub, Jira, and Slack) **influence software development productivity**. It also includes a comparison with the traditional **COCOMO model** to benchmark effort estimation.

The goal is to provide data-driven insights for decision-makers in software teams, helping them understand how their communication and collaboration habits affect productivity.

---

## 🚀 Features

* ✅ **Data Simulation**: Realistic, large-scale synthetic dataset mimicking real-world team behavior (10,000+ samples).
* ✅ **Preprocessing & Feature Engineering**: Transformation of raw collaboration metrics into model-ready features.
* ✅ **Machine Learning Models**: Trained and compared Linear Regression and Random Forest models to predict productivity scores.
* ✅ **Hyperparameter Tuning**: Optimized Random Forest with `GridSearchCV`.
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

3. **Model Evaluation**:

   * Metrics: Mean Squared Error (MSE) and R² Score
   * Best Result: Tuned Random Forest with R² ≈ 0.15

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

  ```
  streamlit
  pandas
  scikit-learn
  plotly
  joblib
  ```

Install with:

```bash
pip install -r requirements.txt
```

### 📦 Running the App

```bash
streamlit run app.py
```

Or deploy free via **[Streamlit Cloud](https://streamlit.io/cloud)**.

---

## 📊 Sample Results

**Best Model: Random Forest (Tuned)**

* MSE: 26.26
* R²: 0.11
* Best Params: `{'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 100}`

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
├── model.joblib            # Trained Random Forest model
├── data.csv                # Simulated dataset
├── requirements.txt        # Python dependencies
└── README.md               # Project description
```

---

## 📌 Future Improvements

* Integrate live APIs from GitHub, Jira, Slack
* Improve model accuracy with time-series or team-specific features
* Add authentication and team-by-team breakdown in dashboard
* Consider neural networks for more advanced predictions

---

## 👨‍💻 Author

Built by \[Your Name] as a final-year university project.
Guided by the objective to blend **machine learning** with **software engineering estimation models**.

---

## 📃 License

MIT License (or your university's license policy)

---

Let me know if you’d like the `requirements.txt` file generated too!
