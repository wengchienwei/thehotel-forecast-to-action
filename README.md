# Hotel Demand Forecasting and AI Planning

End-to-end hotel revenue forecasting: Ridge regression demand model, GenAI copilot design for executive briefings, and agentic planning workflow from forecast alert to controlled action.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![Ridge Regression](https://img.shields.io/badge/model-Ridge%20Regression-2C5F8A.svg)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)

---

## Business Value

THEHOTEL Group operates 11 hotels across 13 weeks of Q1. The Revenue Director asks three questions this analysis is designed to answer:

1. Which hotels are at risk, and which are our growth drivers?
2. Can we forecast weekly revenue for Q1 2021 to size staffing and procurement?
3. How should THEHOTEL use AI end-to-end, from insight to action?

**Key findings:**
- Group Q1 2021 forecast: $126.8M, up 1.2% vs 2019. Hotel 29 is the revenue heavyweight at $22.5M Q1 total
- Five hotels at high confidence (MAPE below 6%): Ridge forecasts used directly for planning. Two hotels (29, 30) better served by the historical mean baseline
- Hotels 23, 24, 27 show structural decline (negative pre-COVID CAGR); Hotels 22, 25, 26, 28, 29 are growth drivers
- Week 13 is the group-level peak for 6 of 11 hotels — procurement and staffing plans must follow hotel-specific seasonality, not a group average

**Pipeline highlights:**
- Demand model: Ridge regression with 5-fold CV on 9 engineered features; two-gate model selection (does Ridge beat baseline? then tier by MAPE)
- GenAI copilot: two-mode design — pre-quarter planning brief (Dagster-scheduled) and weekly deviation alert (agent-triggered)
- Agentic workflow: Forecast Monitoring Agent detects deviations, invokes the copilot, routes below-threshold actions to an Action Execution Agent and above-threshold decisions to the Revenue Director for sign-off

---

## Project Structure

```
thehotel-forecast-to-action/
├── notebooks/
│   └── hotel_demand_forecasting.ipynb   # EDA, feature engineering, Ridge model, evaluation, forecast
├── figures/
│   ├── fig_eda_scale.png                # mean weekly revenue by hotel
│   ├── fig_eda_segments.png             # CAGR vs volatility bubble chart
│   ├── fig_eda_covid.png                # pre-COVID vs 2020 YoY growth
│   ├── fig_eda_seasonality.png          # weekly seasonality index
│   ├── fig_eval_mape.png                # per-hotel MAPE: baseline vs Ridge
│   ├── fig_coef.png                     # Ridge regression coefficients
│   └── fig_forecast.png                 # 2019 hold-out fit and FY2021 Q1 forecast
├── hotel_demand_forecasting_slides.pdf
├── requirements.txt
└── README.md
```

**Note:** `hotels_sales.csv` is not included (proprietary to course).

---

## Quick Start

<details>
<summary><b>Setup and Execution — Click to expand</b></summary>

### Prerequisites
- Python 3.10+
- VS Code with a dedicated virtual environment (no GPU required)

### Setup

1. **Clone repository**
```bash
git clone https://github.com/wengchienwei/thehotel-forecast-to-action.git
cd thehotel-forecast-to-action
```

2. **Create environment and install dependencies**
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name=hotel_forecast --display-name="hotel_forecast"
```

3. **Run notebook**

Place `hotels_sales.csv` in the project root, then open `hotel_demand_forecasting.ipynb` and select the `hotel_forecast` kernel.

```
Note: Dataset not included (proprietary to course).
Expected path: hotels_sales.csv in the project root.
```

</details>

---

## Methodology

**Pipeline:**
1. **EDA** — revenue scale analysis, hotel segmentation by pre-COVID CAGR and volatility, COVID impact by hotel, week-level seasonality and peak identification
2. **Feature engineering** — 9 features computed from prior years only (no leakage): lag, rolling mean, historical average, YoY momentum, week seasonality index, hotel scale, segment, and volatility
3. **Forecasting** — baseline (hotel × week historical mean) vs Ridge regression with 5-fold CV alpha selection; 2020 excluded as a shock year; validation on 2019 hold-out
4. **Model selection** — two-gate logic: Ridge must improve on baseline by more than 0.5 pp to be selected; among passing hotels, confidence tier reflects validation MAPE
5. **GenAI copilot design** — Mode 1: pre-quarter planning brief (Dagster-scheduled, Q1 horizon); Mode 2: weekly deviation alert (agent-triggered, 10% threshold)
6. **Agentic workflow design** — Forecast Monitoring Agent with defined triggers, 6 execution steps, human approval thresholds, audit trail, and failure mode safeguards

**Modeling decision:** Ridge over Random Forest or LightGBM. With 143 training rows, regularisation prevents overfitting and handles correlated features (lag, rolling mean, historical average) without dropping any. 5-fold CV is chosen over LOOCV to avoid high variance from within-hotel autocorrelation.

**2020 treatment:** excluded from training as a shock year; 2019 used as the forecast anchor. Limitation: Hotels 21 and 24 grew in 2020 — their 2021 forecasts may be conservative if demand exceeds 2019 levels.

---

## Tech Stack

**Core:** Python 3.10, pandas, numpy, matplotlib

**Modelling:** scikit-learn (Ridge, RidgeCV, StandardScaler, mean_absolute_error)

---

## License

Code: MIT License

Dataset: Not included (proprietary to course)

---

## Author

Chien-Wei Weng
MSc Data Sciences and Business Analytics
CentraleSupélec × ESSEC Business School

Developed with AI assistance from Claude (Anthropic). See Appendix A7 of the presentation for the full AI usage declaration.

---

**Academic Project | AI in Business and Data Monetization (Spring 2026)**
