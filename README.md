# Machine Learning & Deep Learning Portfolio — Tejamanikanta Gudla

End-to-end machine learning projects spanning time series forecasting, classification, anomaly detection, and predictive modeling — built with Python, Scikit-learn, TensorFlow, and MLflow across real-world public datasets.

**Author:** Tejamanikanta Gudla | Dallas, TX
**LinkedIn:** [linkedin.com/in/tejamanikantagudla](https://www.linkedin.com/in/tejamanikantagudla/)

---

## Projects

### 1. Electric Vehicle Adoption Trends & Forecasting

An end-to-end ML pipeline analyzing and forecasting electric vehicle adoption patterns across U.S. states — combining time series forecasting, classification modeling, and feature importance analysis to identify the key drivers of EV market growth.

#### Problem Statement
EV adoption rates vary dramatically across U.S. states. Understanding which economic, policy, and infrastructure factors drive adoption — and forecasting where growth will accelerate — enables better infrastructure investment and policy planning decisions.

#### Dataset
- 120,000+ records across 16 U.S. states
- Sources: U.S. Department of Energy, state DMV records, NREL charging infrastructure data
- Features: registration counts, charging station density, incentive programs, electricity prices, median income, urbanization rate

#### ML Pipeline

```
Raw Data (APIs + CSV)
    │
    ▼
Data Ingestion & Validation
  ├── Python (Pandas, Requests)
  ├── Schema validation
  └── Null rate monitoring
    │
    ▼
Feature Engineering
  ├── Lag variables (1, 3, 6, 12 month)
  ├── Rolling averages
  ├── Policy dummy encoding
  └── Infrastructure density ratios
    │
    ▼
Models
  ├── ARIMA / SARIMA         (time series baseline)
  ├── Prophet                (trend + seasonality decomposition)
  ├── Random Forest          (ensemble forecasting)
  └── Gradient Boosting      (XGBoost — best performer)
    │
    ▼
Evaluation
  ├── Cross-validation (5-fold)
  ├── RMSE, MAE, MAPE
  └── Feature importance ranking
    │
    ▼
Visualization
  └── Tableau / Power BI dashboards
      (adoption trajectories, regional heatmaps, forecast confidence bands)
```

#### Key Results

| Model | MAPE | Notes |
|---|---|---|
| ARIMA baseline | 18.4% | No external features |
| Prophet | 14.2% | Captures seasonality well |
| Random Forest | 11.8% | Feature engineering critical |
| XGBoost (best) | **10.6%** | 15% improvement over baseline |

**Top predictive features:** charging station density (38%), state incentive value (22%), electricity price (15%), urbanization rate (12%)

**Key finding:** States with >10 public chargers per 100K residents showed 3.2x higher adoption rates independent of income level — infrastructure availability outweighs economic incentives as a growth driver.

---

### 2. NYPD Incident Analysis using ML Models

Predictive modeling and pattern analysis on NYPD crime incident data — classifying incident severity, identifying temporal and geographic crime clusters, and building an anomaly detection layer to flag unusual incident patterns.

#### Problem Statement
NYPD records millions of incident reports annually. This project applies ML classification and clustering techniques to identify actionable patterns — which precincts show rising severity trends, which incident types cluster geographically, and where anomalous activity spikes occur.

#### Dataset
- NYPD publicly available incident records
- Features: incident type, borough, precinct, date/time, suspect demographics, offense classification, arrest flag

#### Models Applied

**Classification — Incident Severity Prediction**
Random Forest and Gradient Boosting classifiers predicting incident severity level based on type, location, and time features.
- Accuracy: 84% | AUC: 0.88
- Key features: offense type (45%), hour of day (18%), precinct (14%)

**Clustering — Geographic Crime Patterns**
K-Means and DBSCAN clustering on incident coordinates to identify spatial crime hotspots and cluster boundaries. Silhouette score used for optimal K selection.

**Anomaly Detection — Spike Identification**
Isolation Forest detecting precincts with statistically unusual incident volume spikes — flagging 4 precincts showing abnormal Q3 2023 patterns vs. historical baseline.

**Time Series Analysis**
ARIMA-based forecasting of weekly incident volume by borough, with seasonal decomposition identifying day-of-week and month-of-year patterns.

#### Key Results
- Identified 3 geographic clusters with >40% higher severity incident density than borough average
- Anomaly detection flagged 2 precincts with statistically significant summer 2023 volume increases
- Classification model achieved 84% accuracy on held-out test set with 5-fold cross-validation

#### Stack
Python · Pandas · Scikit-learn · XGBoost · Statsmodels · Matplotlib · Seaborn · Folium (geospatial) · MLflow

---

## Skills Demonstrated

| Category | Techniques |
|---|---|
| Supervised Learning | Random Forest, Gradient Boosting (XGBoost), Logistic Regression |
| Time Series | ARIMA, SARIMA, Prophet, seasonal decomposition |
| Unsupervised Learning | K-Means, DBSCAN clustering, Isolation Forest |
| Feature Engineering | Lag variables, rolling windows, encoding, interaction terms |
| Evaluation | Cross-validation, AUC, RMSE, MAE, MAPE, silhouette score |
| Experiment Tracking | MLflow (parameters, metrics, model versioning) |
| Visualization | Matplotlib, Seaborn, Plotly, Folium, Tableau |

---

## About

MS in Data Science · University of North Texas (GPA 3.91)
3+ years building ML pipelines in enterprise environments.
Experienced with Scikit-learn, TensorFlow, XGBoost, MLflow, PySpark, Azure Databricks.
Open to Data Scientist, ML Engineer, and Analytics Engineer roles.
