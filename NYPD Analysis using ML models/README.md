# NYPD Incident Analysis using Machine Learning Models

Predictive modeling and pattern analysis on NYPD public crime incident data — applying classification, clustering, anomaly detection, and time series analysis to surface actionable patterns in incident severity, geographic concentration, and temporal trends.

**Tools:** Python (Pandas, Scikit-learn, XGBoost, Folium, Statsmodels) · Machine Learning · CSV

---

## Overview

The NYPD publishes detailed incident-level crime data covering millions of records annually. This project applies a multi-model ML approach to answer four analytical questions: Can we predict incident severity from contextual features? Where do incidents spatially cluster? Are there precincts showing anomalous activity? What do time series trends reveal about borough-level crime dynamics?

## Dataset Features
NYPD incident records including: offense description, offense classification (felony/misdemeanor/violation), borough, precinct, latitude/longitude, date/time, victim demographics, suspect demographics, arrest flag, jurisdiction code.

## ML Pipeline

Data cleaning and feature engineering — standardize offense classification, parse datetime into hour/day/month/season features, encode categorical variables, derive repeat_location_flag and nighttime_flag, handle class imbalance with SMOTE.

Four models applied in parallel:
1. Severity Classification (Random Forest + XGBoost)
2. Geographic Clustering (K-Means + DBSCAN)
3. Anomaly Detection (Isolation Forest)
4. Time Series Forecasting (ARIMA per borough)

## Model Results

**Model 1 — Incident Severity Classification**

| Model | Accuracy | AUC |
|---|---|---|
| Logistic Regression (baseline) | 71% | 0.79 |
| Random Forest | 82% | 0.88 |
| XGBoost (best) | 84% | 0.91 |

Top features: offense_type (45%), hour_of_day (18%), precinct (14%), borough (11%)

**Model 2 — Geographic Crime Clustering**
K-Means (optimal K=7 via silhouette score) identified 7 distinct incident clusters — 3 high-density urban core clusters and 4 lower-density peripheral clusters. DBSCAN identified 12 micro-hotspot zones with 5x the surrounding area incident density.

**Model 3 — Anomaly Detection**
Isolation Forest (contamination=0.05) flagged 4 precincts in Q3 2023 with volume spikes above 2 standard deviations. 2 precincts showed anomalous severity index shifts (more felonies, same total volume).

**Model 4 — Time Series Forecasting**
ARIMA models per borough on weekly incident counts revealed strong seasonality (summer peaks, February troughs) and multi-year downward trends in 3 of 5 boroughs.
- Brooklyn: highest variance, hardest to forecast (MAPE: 14.2%)
- Staten Island: most predictable, lowest volume (MAPE: 8.1%)

## Key Findings
- Offense type is the strongest severity predictor — contextual features add 20% incremental accuracy
- 3 geographic clusters account for 58% of all felony incidents despite covering 12% of geographic area
- Nighttime incidents (10pm-4am) are 2.4x more likely to be felonies than daytime incidents
- Weekly incident volume shows a consistent Friday spike across all boroughs (22% above weekly mean)
- Summer months show 31% higher incident volume than winter months

## Geospatial Visualization
Interactive Folium maps showing incident density heatmap by borough, K-Means cluster boundaries, anomaly-flagged precinct highlights, and time-animated incident density by month.

## Skills Demonstrated
Multi-model ML pipeline · XGBoost classification · K-Means and DBSCAN clustering · Isolation Forest anomaly detection · ARIMA time series · SMOTE class balancing · Folium geospatial mapping · Feature engineering · Crime analytics
