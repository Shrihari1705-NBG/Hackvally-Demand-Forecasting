# Demand Forecasting Report

**SYNTHETIC DATA – EXPERIMENTAL FORECASTING RESULTS**

- Model: `HistGradientBoostingRegressor(loss='poisson')`
- Training rows: `273,164`
- Validation rows: `58,468`
- Test rows: `58,656`
- Rows removed for insufficient history: `15,792`

## Test metrics

- ML MAE: `0.4527`
- Seasonal-naive MAE: `0.4778`
- ML WAPE: `1.386692995071987`
- Seasonal-naive WAPE: `1.463651556298308`

The model comparison is an experiment on simulated demand and does not establish performance on actual Belagavi rides.
No random split, lag leakage, API, or frontend integration was used.
