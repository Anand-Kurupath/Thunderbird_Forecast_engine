def ml_forecast(train, regular_series, multivariate_models, horizon, processed_data_path, cores_to_use = -1, lags=[1,2,3,4,6,9,12],static_features = None, dynamic_features = None,oos=None):

    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression,HuberRegressor,RANSACRegressor,TheilSenRegressor, Ridge
    from sklearn.svm import LinearSVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from mlforecast import MLForecast
    from window_ops.rolling import rolling_mean, rolling_max, rolling_min, rolling_std, seasonal_rolling_max, seasonal_rolling_mean, seasonal_rolling_min, seasonal_rolling_std
    from window_ops.ewm import ewm_mean
    import pandas as pd
    import os
    import numpy as np
    from numba import njit

    @njit
    def diff(x, lag):
        x2 = np.full_like(x, np.nan)
        for i in range(lag, len(x)):
            x2[i] = x[i] - x[i-lag]
        return x2

    train = train[train['unique_id'].isin(regular_series)]
        
    models_mv = [
                RandomForestRegressor(random_state=0, n_estimators=30),
                XGBRegressor(random_state=0, n_estimators=30),
                HistGradientBoostingRegressor(scoring="neg_mean_absolute_error",random_state=0),
                LinearRegression(),
                HuberRegressor(),
                RANSACRegressor(max_trials = 400),
                TheilSenRegressor(),
                Ridge(alpha=.01),
                LinearSVR(fit_intercept=True, epsilon = .01, C = .01,max_iter =  200),
                KNeighborsRegressor(),
                MLPRegressor()
            ]
    
    model = MLForecast(models=models_mv, freq='MS', lags=lags, 
                      lag_transforms={ 3: [(rolling_mean, 4), (rolling_min, 4), (rolling_max, 4), (rolling_std, 4), (ewm_mean, 0.3),
                                  (diff, 1), (diff, 3), (diff, 4),
                                  (seasonal_rolling_mean, 4, 3),(seasonal_rolling_min, 4, 3),(seasonal_rolling_max, 4, 3), (seasonal_rolling_std, 4, 3),
                                  (rolling_mean, 6), (rolling_min, 6), (rolling_max, 6), (rolling_std, 6),
                                  (seasonal_rolling_mean, 4, 6),(seasonal_rolling_min, 4, 6), (seasonal_rolling_max, 4, 6),  (seasonal_rolling_std, 4, 6),
                                  (rolling_mean, 9), (rolling_min, 9), (rolling_max, 9), (rolling_std, 9),
                                  ],
                                  6: [(rolling_mean, 4), (rolling_min, 4), (rolling_max, 4), (rolling_std, 4), (ewm_mean, 0.3),
                                  (diff, 1), (diff, 3), (diff, 4),
                                  (seasonal_rolling_mean, 4, 3),(seasonal_rolling_min, 4, 3),(seasonal_rolling_max, 4, 3), (seasonal_rolling_std, 4, 3),
                                  (rolling_mean, 6), (rolling_min, 6), (rolling_max, 6), (rolling_std, 6),
                                  (seasonal_rolling_mean, 4, 6),(seasonal_rolling_min, 4, 6), (seasonal_rolling_max, 4, 6),  (seasonal_rolling_std, 4, 6),
                                  (rolling_mean, 9), (rolling_min, 9), (rolling_max, 9), (rolling_std, 9),
                                  ],
                                  9: [(rolling_mean, 4), (rolling_min, 4), (rolling_max, 4), (rolling_std, 4), (ewm_mean, 0.3),
                                  (diff, 1), (diff, 3), (diff, 4),
                                  (seasonal_rolling_mean, 4, 3),(seasonal_rolling_min, 4, 3),(seasonal_rolling_max, 4, 3), (seasonal_rolling_std, 4, 3),
                                  (rolling_mean, 6), (rolling_min, 6), (rolling_max, 6), (rolling_std, 6),
                                  (seasonal_rolling_mean, 4, 6),(seasonal_rolling_min, 4, 6), (seasonal_rolling_max, 4, 6),  (seasonal_rolling_std, 4, 6),
                                  (rolling_mean, 9), (rolling_min, 9), (rolling_max, 9), (rolling_std, 9),
                                  ],
                                  12: [(rolling_mean, 4), (rolling_min, 4), (rolling_max, 4), (rolling_std, 4), (ewm_mean, 0.3),
                                  (diff, 1), (diff, 3), (diff, 4),
                                  (seasonal_rolling_mean, 4, 3),(seasonal_rolling_min, 4, 3),(seasonal_rolling_max, 4, 3), (seasonal_rolling_std, 4, 3),
                                  (rolling_mean, 6), (rolling_min, 6), (rolling_max, 6), (rolling_std, 6),
                                  (seasonal_rolling_mean, 4, 6),(seasonal_rolling_min, 4, 6), (seasonal_rolling_max, 4, 6),  (seasonal_rolling_std, 4, 6),
                                  (rolling_mean, 9), (rolling_min, 9), (rolling_max, 9), (rolling_std, 9),
                                     ]},
                        date_features=['month','quarter'], 
                      num_threads=cores_to_use,
                      )
    

    model.models = dict((k, model.models[k]) for k in list(model.models.keys()) if k in multivariate_models)


    if static_features: model.fit(train, id_col='unique_id', time_col='ds', target_col='y', static_features=static_features)
    else: model.fit(train, id_col='unique_id', time_col='ds', target_col='y', static_features=[])
    if dynamic_features: ml_ = model.predict(horizon=horizon, dynamic_dfs=[oos[['unique_id','ds']+dynamic_features]])
    else: ml_ = model.predict(horizon=horizon)
    
    try:
        ax = pd.Series(model.models_['XGBRegressor'].feature_importances_, index=model.ts.features_order_).sort_values(ascending=False)[0:20].plot.bar(
                figsize=(30,50), title='XGBRegressor Feature Importance', xlabel='Features', ylabel='Importance')
        ax.figure.savefig(os.path.join(processed_data_path,'xgb_feature_importance.pdf'))
    except:
        pass

    try:
        ar = pd.Series(model.models_['RandomForestRegressor'].feature_importances_, index=model.ts.features_order_).sort_values(ascending=False)[0:20].plot.bar(
                        figsize=(30,50), title='RandomForestRegressor Feature Importance', xlabel='Features', ylabel='Importance')
        ar.figure.savefig(os.path.join(processed_data_path,'rf_feature_importance.pdf'))
    except:
        pass

    return ml_,model
    