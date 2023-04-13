import warnings
warnings.filterwarnings('ignore')
import forecast_engine.preprocess.nodes as preprocess
def sf_forecast(data,regular_series,univariate_models,horizon,season_length,lags=[1,3,4],window_size=3):

    train_regular = data[data['unique_id'].isin(regular_series)]



    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta ,AutoRegressive, MSTL, GARCH, ARCH, HistoricAverage, Naive, RandomWalkWithDrift, SeasonalNaive, WindowAverage, SimpleExponentialSmoothingOptimized, SeasonalExponentialSmoothingOptimized, HoltWinters, ADIDA, CrostonOptimized, CrostonSBA, IMAPA, TSB
    models_uni = [
                AutoARIMA(season_length = season_length),
                AutoETS(model='ZZZ',  season_length = season_length),
                AutoTheta(season_length = season_length),
                AutoRegressive(lags = lags),
                MSTL(season_length = season_length),
                ARCH(),
                HistoricAverage(),
                Naive(),
                RandomWalkWithDrift(),
                SeasonalNaive(season_length=season_length),
                WindowAverage(window_size=window_size),
                SimpleExponentialSmoothingOptimized(),
                SeasonalExponentialSmoothingOptimized(season_length = season_length),
                HoltWinters(season_length = season_length),
                ]
    




    sf = StatsForecast(models=models_uni, freq = 'MS', n_jobs=-1, fallback_model = HistoricAverage())
    sf.models = preprocess.mode_uni(univariate_models,sf,0)
    
    
    constant = .001
    train_regular['y'] += constant
    
    sf.fit(train_regular)
    sf_= sf.predict(h=horizon)
    sf_.iloc[:,1:] -= constant



    
   




    return sf_