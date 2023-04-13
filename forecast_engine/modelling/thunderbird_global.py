import forecast_engine.modelling.ml_forecast.ml_forecast as ml_forecast
import forecast_engine.modelling.ml_forecast.stats_forecast as sf_forecast
import forecast_engine.modelling.ml_forecast.smape as smape
import forecast_engine.modelling.ml_forecast.skus_to_exclude as skus_exclude
import forecast_engine.modelling.ml_forecast.output_prep as output_prep
import forecast_engine.modelling.ml_forecast.mape_prep as mape_prep
import forecast_engine.modelling.utils.config as config
import forecast_engine.preprocess.nodes as preprocess
import forecast_engine.modelling.ml_forecast.ml_forecast_tuned as ml_forecast_tuned
import forecast_engine.modelling.ml_forecast.hyper_parameter_optuna as tune_models
import forecast_engine.modelling.local_pipeline.forecast_engine.forecast_engine as forecast_engine
import forecast_engine.modelling.local_pipeline.models.ml_models.naive_models as naive_models
import forecast_engine.modelling.local_pipeline.models.ml_models.holt_winters as holt_winters
import forecast_engine.modelling.local_pipeline.models.ml_models.croston as croston




import datetime
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Load config and create output paths
config_catalog = config.load_catalog_params(os.getcwd(), 'conf/catalogs')
config_parameters = config.load_catalog_params(os.getcwd(), 'conf/parameters')
now = str(datetime.datetime.now().replace(second=0, microsecond=0))
version = str(config_catalog['data_preprocessing']['version'])
run_level = config_catalog['data_preprocessing']['run_level']
model_output_path = "/".join([os.getcwd(),config_catalog['output_file_path']['model_output_path'],str(config_catalog['data_preprocessing']['version'])])
newpath = model_output_path 
processed_data_path =  "/".join([newpath,'processed_data'])
final_results_path =  "/".join([newpath,'final_results'])
mape_summary_path =  "/".join([newpath,'mape_summary'])
# Output Paths
if not os.path.exists(newpath):
    os.makedirs(newpath)
    os.makedirs(processed_data_path)
    os.makedirs(final_results_path)
    os.makedirs(mape_summary_path)





import forecast_engine.modelling.utils.logger as logger
class thunderbird_global():
    
    all = {}
    

    # Inputs
    country = config_catalog['data_preprocessing']['country']
    actual_sales =  "/".join([os.getcwd(),config_catalog['data_preprocessing']['actual_sales_file_path']])
    weather =  "/".join([os.getcwd(),config_catalog['data_preprocessing']['drivers']['weather']])
    macro =  "/".join([os.getcwd(),config_catalog['data_preprocessing']['drivers']['macro']])
    asi =  "/".join([os.getcwd(),config_catalog['data_preprocessing']['drivers']['asi']])
    ndvi =  "/".join([os.getcwd(),config_catalog['data_preprocessing']['drivers']['ndvi']])
    vhi =  "/".join([os.getcwd(),config_catalog['data_preprocessing']['drivers']['vhi']])
        



    # Forecast Engine Parameters
    cores_to_use = config_parameters['global_forecast_engineSettings']['cores_to_use']
    shap_plots = config_parameters['global_forecast_engineSettings']['shap_plots']
    validation_period = config_parameters['global_forecast_engineSettings']['validation_period']
    train_end_date = pd.to_datetime(config_parameters['global_forecast_engineSettings']['train_end_date'])
    forecast_start_date = pd.to_datetime(config_parameters['global_forecast_engineSettings']['forecast_start_date'])
    train_start_date = pd.to_datetime(config_parameters['global_forecast_engineSettings']['train_start_date'])
    try : train_start_date - pd.DateOffset(years=3)
    except: train_start_date = train_end_date - pd.DateOffset(years=3)
    forecast_horizon = config_parameters['global_forecast_engineSettings']['forecast_horizon']

    intermittence_thresh = config_parameters['global_forecast_engineSettings']['intermittence_thresh']
    min_month_thresh = config_parameters['global_forecast_engineSettings']['min_month_thresh']
    min_vol_thresh = config_parameters['global_forecast_engineSettings']['min_vol_thresh']
    season_length = min_month_thresh

    use_external_features = config_parameters['global_forecast_engineSettings']['use_external_features']
    if use_external_features == 1: external_feature_names = config_parameters['global_forecast_engineSettings']['external_feature_names']
    else: external_feature_names = []

    multivariate_models = config_parameters['global_forecast_engineSettings']['multivariate_models']
    univariate_models = config_parameters['global_forecast_engineSettings']['univariate_models']


    hyperparameter_tuning = config_parameters['global_forecast_engineSettings']['hyperparameter_tuning']
    if  hyperparameter_tuning == 1:
        params_xgb = config_parameters['global_forecast_engineSettings']['hyperparameter_tune_space']['params_xgb']
        params_rf = config_parameters['global_forecast_engineSettings']['hyperparameter_tune_space']['params_rf']
        params_lsvr = config_parameters['global_forecast_engineSettings']['hyperparameter_tune_space']['params_lsvr']
        params_histgb = config_parameters['global_forecast_engineSettings']['hyperparameter_tune_space']['params_histgb']
        params_ridge = config_parameters['global_forecast_engineSettings']['hyperparameter_tune_space']['params_ridge']
        params_knn = config_parameters['global_forecast_engineSettings']['hyperparameter_tune_space']['params_knn']
        params_mlp = config_parameters['global_forecast_engineSettings']['hyperparameter_tune_space']['params_mlp']
        


    
    

    def __init__(self,name,df):

        self.name = name
        self.df = df

        thunderbird_global.all[self.name] = self.df 


    
    @classmethod
    def instantiate_inputs(cls):
        
        # Load input files
        sales_data, df, raw_data = preprocess.read_prep_actuals(thunderbird_global.actual_sales,run_level)
        thunderbird_global('raw_data',raw_data)
        wthr_data = preprocess.read_prep_weather(thunderbird_global.weather)
        macro = preprocess.read_macro_files(thunderbird_global.macro, country = thunderbird_global.country)
        agi = preprocess.read_agi_files(thunderbird_global.asi,thunderbird_global.ndvi,thunderbird_global.vhi, country = thunderbird_global.country)
        df_model = preprocess.model_data_combine(df,wthr_data,macro,agi)
        df_model['key'] = df_model['ds'].dt.month.astype(str) + df_model['unique_id'].apply(lambda x : x.split('-')[2])
        df_holi = preprocess.create_holiday(list(raw_data['Country'].unique()))
        df_model = pd.merge(df_model,df_holi[['key','holiday']],on='key',how='left').drop('key',axis=1)
        df_model = df_model[['y','unique_id','ds']+thunderbird_global.external_feature_names]
        logger.logger.info(f" name : raw_data loaded, data shape: {df.shape}")
        logger.logger.info(f" name : wthr_data loaded, data shape: {wthr_data.shape}")
        logger.logger.info(f" name : macro loaded, data shape: {macro.shape}")
        thunderbird_global('sales',df), thunderbird_global('df_model',df_model)
        thunderbird_global('wthr_data',wthr_data), thunderbird_global('macro',macro)
        thunderbird_global('agi',agi), thunderbird_global('sales_data',sales_data)

    
    
    
    @classmethod
    def drop_skus(cls):
        df_model_grp = thunderbird_global.all['df_model'].dropna().groupby('unique_id')['y'].size().reset_index()
        Skus_to_keep = list(df_model_grp[df_model_grp['y']>=thunderbird_global.min_month_thresh]['unique_id'].unique())
        thunderbird_global.all['df_model'] = thunderbird_global.all['df_model'][thunderbird_global.all['df_model']['unique_id'].isin(Skus_to_keep)]
        dropped = pd.DataFrame(list(df_model_grp[df_model_grp['y']<thunderbird_global.min_month_thresh]['unique_id'].unique())).rename(columns={0:'unique_id'})
        dropped['category'] = f'actuals length <{thunderbird_global.min_month_thresh} months'
        thunderbird_global('dropped',dropped)
        logger.logger.info(f" Skus dropped <{thunderbird_global.min_month_thresh} months actuals : {dropped.shape[0]}")

    

    
    @classmethod
    def flag_skus(cls):
        df = thunderbird_global.all['sales']
        dates, act_dates = preprocess.prep_dates_list(df)
        dates = [x for x in df.set_index('Material').columns.to_list() if datetime.datetime.strptime(x, "%Y - %m").date() >= thunderbird_global.train_start_date  and  datetime.datetime.strptime(x, "%Y - %m").date() < thunderbird_global.forecast_start_date] 
        df = df.set_index('Material')[dates].reset_index()
        df['intermittence'] = df.apply(lambda x : preprocess.intermittence(x), axis = 1)
        df['seq - flag'] = df.apply(lambda x : preprocess.seq_flag(x,thunderbird_global.intermittence_thresh,dates,df.columns) , axis = 1)
        SKU_flags = df[['Material','seq - flag']]
        regular_series = list(SKU_flags[SKU_flags['seq - flag'] == 1]['Material'])
        df_local = df.copy()
        df_local.drop(['intermittence'],axis =1,inplace=True)
        df_local.fillna(0,inplace=True)
        thunderbird_global('regular_series',regular_series)
        thunderbird_global('df_local',df_local)
        logger.logger.info(f" sequence flag created, Regular series count : {len(regular_series)}")
    


    
    @classmethod
    def thunder_fuel(cls):
        train = thunderbird_global.all['df_model'].loc[(thunderbird_global.all['df_model']['ds'] >= thunderbird_global.train_start_date)  & 
                (thunderbird_global.all['df_model']['ds'] <= thunderbird_global.train_end_date)].reset_index(drop=True)
        train.fillna(0,inplace=True)
        if thunderbird_global.use_external_features == 0 : 
            train = train[['unique_id', 'ds', 'y']]
            logger.logger.info(f" No external features used")
        logger.logger.info(f" External features used : {thunderbird_global.external_feature_names}")
        valid = thunderbird_global.all['df_model'].loc[(thunderbird_global.all['df_model']['ds'] > thunderbird_global.train_end_date) & 
                 (thunderbird_global.all['df_model']['ds'] <= thunderbird_global.forecast_start_date)].reset_index(drop=True)
        train_uni = train[['y','unique_id','ds']]

        train_grp = train[train["ds"].astype(str).str.contains(str(thunderbird_global.train_end_date.year))].groupby('unique_id')['y'].mean().reset_index()
        sparse_data = list(train_grp[train_grp['y'] < (train_grp['y'].mean()*thunderbird_global.min_vol_thresh)]['unique_id'])
        skus = list(thunderbird_global.all['sales_data'][thunderbird_global.all['sales_data']['year']=='2022']['Material'].unique())
        df_sparse = thunderbird_global.all['df_local'][thunderbird_global.all['df_local']['Material'].isin(sparse_data)]
        df_sparse.rename(columns = {'Material':'index'},inplace=True)
        df_sparse = df_sparse[df_sparse['index'].isin(skus)].reset_index(drop=True)
        thunderbird_global('train_uni',train_uni), thunderbird_global('train',train), thunderbird_global('valid',valid)
        thunderbird_global('df_sparse',df_sparse)
        thunderbird_global('sparse_data',sparse_data)
        logger.logger.info(f" name : univariate train data, data shape: {train_uni.shape}")
        logger.logger.info(f" name : multivariate train data, data shape: {train.shape}")
        logger.logger.info(f" name : validation data, data shape: {valid.shape}")


    

    @classmethod
    def tune_engine(cls):
        if thunderbird_global.hyperparameter_tuning: 
            
            best_params = tune_models.hyper_paramerter_tune(train = thunderbird_global.all['train'], regular_series = thunderbird_global.all['regular_series'], valid = thunderbird_global.all['valid'],
                                                            h = thunderbird_global.validation_period, cores_to_use = thunderbird_global.cores_to_use, static_features = thunderbird_global.external_feature_names)
        
        

            #best hyperparmeters 
            hyper_paramerter_space = {
                                    'params_xgb' : {k:best_params[k] for k,v in thunderbird_global.params_xgb.items()},
                                    'params_rf' : {k:best_params[k] for k,v in thunderbird_global.params_rf.items()},
                                    'params_lsvr' : {k:best_params[k] for k,v in thunderbird_global.params_lsvr.items()},
                                    'params_histgb' : {k:best_params[k] for k,v in thunderbird_global.params_histgb.items()},
                                    'params_ridge' : {k:best_params[k] for k,v in thunderbird_global.params_ridge.items()},
                                    'params_knn' : {k:best_params[k] for k,v in thunderbird_global.params_knn.items()},
                                    'params_mlp' : {k:best_params[k] for k,v in thunderbird_global.params_mlp.items()},
                                    }
            hyper_paramerter_space_df = pd.DataFrame(hyper_paramerter_space)                       
            thunderbird_global('hyper_paramerter_space',hyper_paramerter_space)
            thunderbird_global('hyper_paramerter_space_df',hyper_paramerter_space_df)
            logger.logger.info(f" engine tuning complete : {hyper_paramerter_space}")
        
        else:
            logger.logger.info(f" engine not tuned")
            pass

           
           
    @classmethod
    def fit_models(cls):  
        
        if thunderbird_global.hyperparameter_tuning : 
            ml_, model = ml_forecast_tuned.ml_forecast(thunderbird_global.all['train'], thunderbird_global.all['regular_series'], thunderbird_global.multivariate_models,
                                                hyper_paramerter_space = thunderbird_global.all['hyper_paramerter_space'],
                                                horizon = thunderbird_global.validation_period, processed_data_path = processed_data_path, cores_to_use = thunderbird_global.cores_to_use,
                                                static_features = thunderbird_global.external_feature_names)
        else : 
            ml_, model = ml_forecast.ml_forecast(thunderbird_global.all['train'], thunderbird_global.all['regular_series'], thunderbird_global.multivariate_models,
                                                horizon = thunderbird_global.validation_period, processed_data_path = processed_data_path, cores_to_use = thunderbird_global.cores_to_use,
                                                static_features = thunderbird_global.external_feature_names)
        
        sf_ = sf_forecast.sf_forecast(thunderbird_global.all['train_uni'], thunderbird_global.all['regular_series'], 
                                   univariate_models = thunderbird_global.univariate_models, 
                                   season_length = int((thunderbird_global.season_length)/3), horizon = thunderbird_global.validation_period) 

        final_model_fit = output_prep.output_prep(ml_,sf_, thunderbird_global.all['sales_data'])
        best_model_df, valid_predictions = mape_prep.mape_prep(final_model_fit)
        best_model_df = best_model_df.dropna()
        logger.logger.info(f" regular series validation mape : {best_model_df['validation_mape'].median()}")
        skus = list(best_model_df.dropna()['unique_id'].unique())
        skus_dropped = pd.DataFrame([x for x in thunderbird_global.all['regular_series'] if x not in list(final_model_fit['unique_id'].unique())]).rename(columns={0:'unique_id'})
        skus_dropped['category'] = 'no actuals found for validation period'

        
        
        thunderbird_global.all['dropped'] = pd.concat([thunderbird_global.all['dropped'],skus_dropped])
        thunderbird_global('final_model_fit',final_model_fit), thunderbird_global('best_model_df',best_model_df)
        thunderbird_global('skus', skus), thunderbird_global('model_performance_check', valid_predictions)
        
        

    
    @classmethod
    def predict_oos(cls):  
        thunderbird_global.all['train'] = thunderbird_global.all['df_model'][ (thunderbird_global.all['df_model']['ds'] >= thunderbird_global.train_start_date) &
                        (thunderbird_global.all['df_model']['ds'] < thunderbird_global.forecast_start_date)].reset_index(drop=True)
        thunderbird_global.all['train'].fillna(0,inplace=True)
        if thunderbird_global.use_external_features == 0 : thunderbird_global.all['train'] = thunderbird_global.all['train'][['unique_id', 'ds', 'y']]
        thunderbird_global.all['train_uni'] = thunderbird_global.all['train'][['unique_id','y','ds']]

        if thunderbird_global.hyperparameter_tuning : 
            ml_oos, model = ml_forecast_tuned.ml_forecast(thunderbird_global.all['train'], thunderbird_global.all['regular_series'], thunderbird_global.multivariate_models,
                                                hyper_paramerter_space = thunderbird_global.all['hyper_paramerter_space'],
                                                horizon = thunderbird_global.forecast_horizon, processed_data_path = processed_data_path, cores_to_use = thunderbird_global.cores_to_use,
                                                static_features = thunderbird_global.external_feature_names)

        else :
            ml_oos, model = ml_forecast.ml_forecast(thunderbird_global.all['train'], thunderbird_global.all['regular_series'], thunderbird_global.multivariate_models,
                                            horizon = thunderbird_global.forecast_horizon, processed_data_path = processed_data_path, cores_to_use = thunderbird_global.cores_to_use,
                                            static_features = thunderbird_global.external_feature_names)
        
        sf_oos = sf_forecast.sf_forecast(thunderbird_global.all['train_uni'], thunderbird_global.all['regular_series'],
                                univariate_models = thunderbird_global.univariate_models, 
                                season_length = int((thunderbird_global.season_length)/3), horizon = thunderbird_global.forecast_horizon)
        
        final_output = output_prep.prep_oos(ml_oos, sf_oos, thunderbird_global.all['best_model_df'], thunderbird_global.all['skus'])


        
        thunderbird_global('model',model)
        thunderbird_global('final_output',final_output)
        thunderbird_global.all['final_output']['value'] = thunderbird_global.all['final_output']['value'].apply(lambda x : x if x > 0 else 0)
        final_output_1 = preprocess.post_process(final_output,thunderbird_global.all['best_model_df'],thunderbird_global.forecast_start_date)
        thunderbird_global('final_output_1',final_output_1)






    @classmethod
    def thunder_fuel_local(cls):  
        univariate_models = {}
        multivariate_models = {}
        if globals()['holt_winters'] : univariate_models = {func.__name__:func for func in filter(callable, globals()['holt_winters'].__dict__.values()) if func.__name__ in config_parameters['global_forecast_engineSettings']['low_volume_sparse']}
        if globals()['naive_models'] : univariate_models.update({func.__name__:func for func in filter(callable, globals()['naive_models'].__dict__.values()) if func.__name__ in config_parameters['global_forecast_engineSettings']['low_volume_sparse']})
        config_parameters['global_forecast_engineSettings']['low_volume_sparse'] = [x for x in config_parameters['global_forecast_engineSettings']['low_volume_sparse'] if x not in list(univariate_models.keys())] 

        for i in config_parameters['global_forecast_engineSettings']['low_volume_sparse']:
            univariate_models.update({func.__name__:func  for func in filter(callable, globals()[i].__dict__.values())})
        logger.logger.info(f"Thunderbird sparse_model : loaded models")

        list_of_models = univariate_models.copy()
        thunderbird_global('sparse_models',list_of_models)





    @classmethod
    def thunder_sparse_pipeline(cls):
        lst_error = []
        keys = list(thunderbird_global.all['df_sparse']['index'].unique())
        oos = pd.DataFrame(columns = ['index','DATE','Method','Forecast'])
        models = pd.DataFrame(columns = ['models', 'smape','index'])
        fitted_values = pd.DataFrame(columns = ['date_', 'forecast', 'method','index'])
        ensemble = pd.DataFrame(columns= ['index','date_','forecast','method'])
        for i in keys:
            try:
                df_output = thunderbird_global.all['df_sparse'][thunderbird_global.all['df_sparse']['index'] == i].apply(lambda x: forecast_engine.forecast_engine(x, months_to_forecast = thunderbird_global.forecast_horizon,
                                        debug_flag = 0, debug_models = [], list_of_models = thunderbird_global.all['sparse_models'], single_series = 0,  
                                        all_at_once_flag = 1, validation_window = 3, max_validation = thunderbird_global.validation_period, 
                                        outlier_flag = 1, select_variable_flag = 0),axis = 1)    
                df_output = df_output.ravel()
                output_temp = df_output[0]['output'][['index','DATE','Method','Forecast']]
                oos = pd.concat([oos,output_temp])

                model_list = df_output[0]['dataf']
                model_list['index'] = i
                models = pd.concat([models,model_list])
                
                df_fitted_values_list = df_output[0]['fitted_values_list']
                df_fitted_values_list = df_fitted_values_list[0][model_list['models'][0]]['forecast']
                df_fitted_values_list['index'] = i
                fitted_values = pd.concat([fitted_values,df_fitted_values_list])

                df_ensemble = df_output[0]['df_ensemble'][['index','date_','forecast','method']]
                ensemble = pd.concat([ensemble,df_ensemble])

                
            except:
                lst_error.append(i)
        
        final_output_1_sparse , sap_output_sparse = preprocess.post_process_sparse(oos,models,run_level,thunderbird_global.all['raw_data'],thunderbird_global.forecast_start_date)
        validation_predictions_sparse = preprocess.valid_predictions_sprase_prep(fitted_values,thunderbird_global.all['sales_data'],thunderbird_global.train_end_date,thunderbird_global.forecast_start_date)
        logger.logger.info(f" sparse series validation mape : {final_output_1_sparse['validation_mape'].median()}")
        regular_series = [ x for x in thunderbird_global.all['regular_series'] if x not in thunderbird_global.all['sparse_data']]
        thunderbird_global.all['model_performance_check'] = pd.concat([validation_predictions_sparse,thunderbird_global.all['model_performance_check'][thunderbird_global.all['model_performance_check']['unique_id'].isin(regular_series)]])
        thunderbird_global.all['final_output_1'] = pd.concat([final_output_1_sparse,thunderbird_global.all['final_output_1'][thunderbird_global.all['final_output_1']['unique_id'].isin(regular_series)]])
        sap_output = preprocess.prep_sap_output(thunderbird_global.all['raw_data'],thunderbird_global.all['final_output'][thunderbird_global.all['final_output']['unique_id'].isin(regular_series)],run_level)
        sap_output = pd.concat([sap_output,sap_output_sparse])
        preprocess.plot_validataion(thunderbird_global.all['model_performance_check'],thunderbird_global.forecast_start_date,processed_data_path)
        thunderbird_global('sap_output',sap_output)





    @classmethod
    def generate_results(cls):  
        thunderbird_global.all['df_model'].to_csv(os.path.join(processed_data_path,'df_model.csv'),index=False)
        thunderbird_global.all['dropped'].to_csv(os.path.join(processed_data_path,'dropped_skus.csv'),index=False)
        if thunderbird_global.hyperparameter_tuning: 
            thunderbird_global.all['hyper_paramerter_space_df'].to_csv(os.path.join(processed_data_path,'hyper_paramerter_space.csv'))
        else:
            pass
        thunderbird_global.all['best_model_df'].to_csv(os.path.join(mape_summary_path,'best_model_regular_series.csv'),index=False)
        thunderbird_global.all['model_performance_check'].to_csv(os.path.join(mape_summary_path,'model_performance_check.csv'),index=False)
        thunderbird_global.all['final_output_1'].to_csv(os.path.join(final_results_path,'final_results.csv'),index=False)
        thunderbird_global.all['sap_output'].to_csv(os.path.join(final_results_path,version+'_SAP.csv'),index=False)
        logger.logger.info(" Thunder bird forecast engine run complete.")
        logger.logger.info(f" Forecast Run catalogs : {config_catalog}")
        logger.logger.info(f" Forecast Run parameters : {config_parameters}")

        if thunderbird_global.shap_plots: 
            logger.logger.info(f" Generating SHAP plots for multivariate models...")
            logger.logging.shutdown()
            preprocess.shap_plots(thunderbird_global.all['train'] , thunderbird_global.all['model'], thunderbird_global.all['best_model_df'],
                                                    thunderbird_global.all['regular_series'], processed_data_path)
            logger.logger.info(f" Generating SHAP plots finished")




    def __repr__(self):
        return f"{self.name}, {self.df}"