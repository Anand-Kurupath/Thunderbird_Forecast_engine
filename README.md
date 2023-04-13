# Author: 
Anand K
Senior Data Analyst, QB - CCN
McKinsey & Company
E-Mail: anand_k@mckinsey.com
Phone: +918075969244

# External drivers:
- agriculture stress index, vegitation health index, normalized difference vegitation index -> https://www.fao.org/giews/earthobservation/country/index.jsp?code=MEX


# Thunderbird Forecast Engine
This is a python library for solving forecasting problems. For faster execution, Models are fit globally to capture global parameters and execute parallelly.


# Compatibility
- Python version = 3.9.13
- pandas==1.5.3
- numpy==1.23.5
- pyculiarity==0.0.7
- xgboost==1.7.5
- plotly==5.14.0
- pyyaml==6.0
- openpyxl==3.1.2
- seaborn==0.12.2
- mlforecast==0.6.0
- optuna==3.1.0
- statsforecast==1.5.0
- nbformat==5.8.0
- shap==0.41.0

## Algorithms :

### Modelling techniques covered:

1. Linear Regression - OLS, Ridge, Huber, RANSAC, TheilSen
2. Ensemble - XGBoost, Histogram Gradient Boosing, Random Forest
3. Artificial Neural Network - Multilayer Perceptron
4. Lazy learning algorithm - KNN
5. Univariate - Arima, ETS, Theta, AR, MSTL, ARCH
6. Naive models - Historic Average, Naive, Random walk drift, Seasonal Naive, Window Average, Simple exponential smoothing, Seasonal exponential smoothing, Holtwinters, ADIDA, Croston, Croston SBA, IMPA, TSB


### Optimization technique used:


- Optuna


## Installation
From the python console/terminal, run

```bash
 pip install -r requirements.txt
```

## Usage Steps:


1. Save the raw data in the folder Data --> input_data --> actual_sales/sales_order



2. In the config --> catalogs folder, Open conf_master.yaml file to enter run_type : 'sales_order' for using sales order as inputs, 
   run_type : 'actual_sales' for using actual sales as inputs
  

      - Benchmark machine compute summary : 10 cores, 64 GB RAM 
      - Benchmark run-time for run type sales_order/Mexico with hyperparameter tuning : ~15 mins (300+ SKUs, From 2019-01-01 data, with 10 external drivers)
      - Benchmark run-time for run type sales_order/Mexico without hyperparameter tuning : ~8 mins (300+ SKUs, From 2019-01-01 data, with 10 external drivers)
      
      - Benchmark run-time for run type actual sales/Entire LATAM with hyperparameter tuning : ~30 mins (4000+ SKUs, From 2019-01-01, without external drivers)
      - Benchmark run-time for run type actual sales/Entire LATAM without hyperparameter tuning : ~15 mins (4000+ SKUs, From 2019-01-01, without external drivers)



3. In the config --> catalogs folder, Open forecast_engine_global.yaml file to enter model configuarations
      - Configure forecast_engine_global.yaml : train_end_date & forecast_engine_global.yaml : forecast_start_date 
      - To choose hyperparameter tuning, set 1 in forecast_engine_global.yaml : hyperparameter_tuning
      - Configure forecast_engine_global.yaml : forecast_horizon
      - Configure forecast_engine_global.yaml : min_month_thresh, series with less than the value will be dropped
      - Configure forecast_engine_global.yaml : intermittence_thresh, series with intermittence > than the value will forecast through sparse model pipeline
      - Configure forecast_engine_global.yaml : use_external_features, set 0 to exclude external features
      - Configure forecast_engine_global.yaml : external_feature_names, mention exclude external feature column names






4. From the python console/terminal, run


```bash
      python3 thunderbird_global_model_pipeline.py
```


5. Results & outputs will be stored in the respective folders 