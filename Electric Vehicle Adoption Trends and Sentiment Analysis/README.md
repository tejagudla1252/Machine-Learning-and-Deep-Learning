# Electric Vehicle Adoption Trends and Sentiment Analysis

End-to-end ML research project combining time series forecasting and NLP sentiment analysis to understand and predict U.S. electric vehicle adoption — with state-level registration data, zip code demographics, and a video demonstration.

**Tools:** Python · Scikit-learn · TensorFlow · Statsmodels · NLTK · CSV
**Project Type:** Group Research Project (Group 24) · INFO 5082 · University of North Texas

---

## Overview

This project takes a dual-track approach: a quantitative track using ML models to forecast regional EV adoption from registration and demographic data, and a qualitative track using NLP sentiment analysis to understand public perception of EVs from text data. Together they provide both a predictive and attitudinal picture of EV market dynamics.

## Files

| File | Description |
|---|---|
| `Project_Group_24/` | Full project codebase — notebooks, scripts, models |
| `Project_Group_24.zip` | Zipped project archive |
| `Yearly_sales_by_state.csv` | Annual EV sales aggregated by U.S. state |
| `registration_counts.csv` | Raw EV registration count data |
| `all_us_zipcodes.csv` | U.S. zip code reference data for geographic joins |
| `zip_code_data_subset.csv` | Demographic data subset joined to zip codes |
| `EV Report.pdf` | Full research report with methodology and findings |
| `Final Research Project_Group24_5082.pptx` | Presentation deck |
| `Documentation and Installation Steps Group 24.docx` | Setup and installation guide |
| `video2150655397.mp4` | Video demonstration of the project and key findings |

## Architecture

Track 1 — Forecasting Pipeline: Data ingestion from registration + zip code demographics, feature engineering (lag variables, rolling averages, demographic ratios), model training (ARIMA, Random Forest, XGBoost, LSTM), evaluation with MAPE.

Track 2 — Sentiment Pipeline: Text preprocessing (NLTK tokenization, stopword removal), TF-IDF vectorization, sentiment classification (Logistic Regression baseline, LSTM classifier), sentiment-adoption correlation analysis.

## Model Results

**Forecasting (Track 1)**

| Model | MAPE | Notes |
|---|---|---|
| ARIMA baseline | 18.4% | No external features |
| Random Forest | 12.1% | Feature engineering key |
| XGBoost (best) | 10.6% | Best performer |
| LSTM | 11.8% | Captures non-linear trends |

**Sentiment Analysis (Track 2)**
- Logistic Regression baseline: 81% accuracy
- LSTM text classifier: 87% accuracy on held-out test set

## Key Findings

**Adoption Forecasting**
- Charging station density is the single strongest predictor of adoption growth (feature importance: 38%)
- State incentive value adds 22% of predictive power — but only above a $3,500 threshold
- Zip codes with median income above $75K show 4.1x higher adoption rates than lower-income areas
- XGBoost improved forecast accuracy by 42% over the ARIMA baseline with engineered features

**Sentiment Analysis**
- Public sentiment toward EVs has trended positive since 2020 (62% to 74% positive)
- Range anxiety is the number one negative sentiment driver (appears in 41% of negative mentions)
- States with higher positive EV sentiment show 18% faster adoption growth 6 months later
- Tesla mentions dominate (67% of all EV brand mentions) and skew more positive than other brands

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook EV_Forecasting.ipynb
jupyter notebook EV_Sentiment_Analysis.ipynb
```

## Skills Demonstrated
Time series forecasting · ARIMA · XGBoost · LSTM · NLP sentiment analysis · TF-IDF · Feature engineering · Geographic data joins · Multi-track research design · Academic research reporting
