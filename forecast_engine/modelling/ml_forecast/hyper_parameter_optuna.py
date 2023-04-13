
import forecast_engine.modelling.ml_forecast.smape as smape
def hyper_paramerter_tune(train, valid, regular_series, h, cores_to_use = -1, static_features=[]):

        import optuna
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from xgboost import XGBRegressor
        from sklearn.experimental import enable_hist_gradient_boosting
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import LinearRegression,HuberRegressor,RANSACRegressor,TheilSenRegressor,Ridge
        from sklearn.svm import LinearSVR
        from mlforecast import MLForecast
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.neural_network import MLPRegressor
        from window_ops.rolling import rolling_mean, rolling_max, rolling_min, rolling_std, seasonal_rolling_max, seasonal_rolling_mean, seasonal_rolling_min, seasonal_rolling_std
        import numpy as np
        from window_ops.ewm import ewm_mean

        train = train[train['unique_id'].isin(regular_series)]
        valid = valid[valid['unique_id'].isin(regular_series)]
        
        def objective(trial):
                
                # Define our candidate hyperparameters

                
                params_xgb = {
                        "n_estimators" : trial.suggest_int('n_estimators', 5, 100),
                        'max_depth':trial.suggest_int('max_depth', 1, 5),
                        'reg_alpha':trial.suggest_uniform('reg_alpha',0,6),
                        'reg_lambda':trial.suggest_uniform('reg_lambda',0,2),
                        'min_child_weight':trial.suggest_int('min_child_weight',0,5),
                        'gamma':trial.suggest_uniform('gamma', 0, 4),
                        'learning_rate':trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
                        'colsample_bytree':trial.suggest_uniform('colsample_bytree',0.4,0.9),
                        'subsample':trial.suggest_uniform('subsample',0.4,0.9),
                        }  
                
                params_rf = {
                        'n_estimators': trial.suggest_int('n_estimators', 5, 100),
                        'max_depth': trial.suggest_int('max_depth', 3, 7),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
                        } 

                params_lsvr = {
                        'epsilon': trial.suggest_uniform('epsilon',.01,10),
                        'C' :  trial.suggest_uniform('C',.01,10), 
                        'max_iter': trial.suggest_int('max_iter',50, 500),
                        }

                params_histgb = {
                        'max_depth' :  trial.suggest_int('max_depth',1,9),
                        'learning_rate' : trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
                        'max_bins' : trial.suggest_int('max_bins',10, 200),
                        'max_iter': trial.suggest_int('max_iter',200, 500),
                        }

                params_ridge = {
                        'alpha' : trial.suggest_uniform('alpha',.001,10),
                        }
                
                params_knn = {
                        'n_neighbors' : trial.suggest_int('n_neighbors',2,10),
                        'weights': trial.suggest_categorical('weights', ['uniform','distance']),
                        }
                

                params_mlp = {
                        'learning_rate_init': trial.suggest_float('learning_rate_init', 0.0001, 0.1, step=0.005),
                        'first_layer_neurons': trial.suggest_int('first_layer_neurons', 10, 100, step=10),
                        'second_layer_neurons': trial.suggest_int('second_layer_neurons', 10, 100, step=10),
                        'activation': trial.suggest_categorical('activation', ['identity', 'tanh', 'relu']),
                }





                models = [
                        XGBRegressor(random_state=0, **params_xgb),
                        RandomForestRegressor(random_state=0, **params_rf),
                        HistGradientBoostingRegressor(random_state=0,scoring="neg_mean_absolute_error",**params_histgb),
                        LinearSVR(random_state=0,fit_intercept=True, **params_lsvr),
                        Ridge(random_state=0,**params_ridge),
                        KNeighborsRegressor(**params_knn),
                        MLPRegressor(hidden_layer_sizes=(params_mlp['first_layer_neurons'], params_mlp['second_layer_neurons']),
                                learning_rate_init=params_mlp['learning_rate_init'],
                                activation=params_mlp['activation'],
                                random_state=0,
                                max_iter=100),
                        ]

                model = MLForecast(models=models,
                                freq='MS',
                                lags=[1,2,3,4,6,9,12],
                                lag_transforms={ 3: [(rolling_mean, 4), (rolling_min, 4), (rolling_max, 4), (rolling_std, 4),  (ewm_mean, 0.3),
                                     (seasonal_rolling_mean, 4, 3),(seasonal_rolling_min, 4, 3),(seasonal_rolling_max, 4, 3), (seasonal_rolling_std, 4, 3),
                                     (rolling_mean, 6), (rolling_min, 6), (rolling_max, 6), (rolling_std, 6),
                                     (seasonal_rolling_mean, 4, 6),(seasonal_rolling_min, 4, 6), (seasonal_rolling_max, 4, 6),  (seasonal_rolling_std, 4, 6),
                                     (rolling_mean, 9), (rolling_min, 9), (rolling_max, 9), (rolling_std, 9),
                                     ],
                                     6: [(rolling_mean, 4), (rolling_min, 4), (rolling_max, 4), (rolling_std, 4),  (ewm_mean, 0.3),
                                     (seasonal_rolling_mean, 4, 3),(seasonal_rolling_min, 4, 3),(seasonal_rolling_max, 4, 3), (seasonal_rolling_std, 4, 3),
                                     (rolling_mean, 6), (rolling_min, 6), (rolling_max, 6), (rolling_std, 6),
                                     (seasonal_rolling_mean, 4, 6),(seasonal_rolling_min, 4, 6), (seasonal_rolling_max, 4, 6),  (seasonal_rolling_std, 4, 6),
                                     (rolling_mean, 9), (rolling_min, 9), (rolling_max, 9), (rolling_std, 9),
                                     ],
                                     9: [(rolling_mean, 4), (rolling_min, 4), (rolling_max, 4), (rolling_std, 4), (ewm_mean, 0.3),
                                     (seasonal_rolling_mean, 4, 3),(seasonal_rolling_min, 4, 3),(seasonal_rolling_max, 4, 3), (seasonal_rolling_std, 4, 3),
                                     (rolling_mean, 6), (rolling_min, 6), (rolling_max, 6), (rolling_std, 6),
                                     (seasonal_rolling_mean, 4, 6),(seasonal_rolling_min, 4, 6), (seasonal_rolling_max, 4, 6),  (seasonal_rolling_std, 4, 6),
                                     (rolling_mean, 9), (rolling_min, 9), (rolling_max, 9), (rolling_std, 9),
                                     ],
                                     12: [(rolling_mean, 4), (rolling_min, 4), (rolling_max, 4), (rolling_std, 4), (ewm_mean, 0.3),
                                     (seasonal_rolling_mean, 4, 3),(seasonal_rolling_min, 4, 3),(seasonal_rolling_max, 4, 3), (seasonal_rolling_std, 4, 3),
                                     (rolling_mean, 6), (rolling_min, 6), (rolling_max, 6), (rolling_std, 6),
                                     (seasonal_rolling_mean, 4, 6),(seasonal_rolling_min, 4, 6), (seasonal_rolling_max, 4, 6),  (seasonal_rolling_std, 4, 6),
                                     (rolling_mean, 9), (rolling_min, 9), (rolling_max, 9), (rolling_std, 9),
                                     ]}, 
                                date_features=['month','quarter'],
                                num_threads=cores_to_use)

                model.fit(train, id_col='unique_id', time_col='ds', target_col='y', static_features=static_features)
                p = model.predict(horizon=h)
                p = p.merge(valid[['unique_id', 'ds', 'y']], on=['unique_id', 'ds'], how='left')

                error_xgb = smape.smape(p['y'], p['XGBRegressor'])
                error_rf = smape.smape(p['y'], p['RandomForestRegressor'])
                error_histgb = smape.smape(p['y'], p['HistGradientBoostingRegressor'])
                error_lsvr = smape.smape(p['y'], p['LinearSVR'])
                error_ridge = smape.smape(p['y'], p['Ridge'])
                error_knn = smape.smape(p['y'], p['KNeighborsRegressor'])
                error_mlp = smape.smape(p['y'], p['MLPRegressor'])
                

                accuracy = 100 - np.mean([error_xgb,error_rf,error_histgb,error_lsvr,error_ridge,error_knn,error_mlp])

                return accuracy


        study = optuna.create_study(study_name="mv_hyperparameters",directions=["maximize"])
        study.optimize(objective, n_trials=30, timeout= 500)

        return study.best_params


