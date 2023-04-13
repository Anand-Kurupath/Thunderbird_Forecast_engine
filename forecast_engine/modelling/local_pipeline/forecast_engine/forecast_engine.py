import forecast_engine.modelling.local_pipeline.forecast_engine.model_comparison as model_comparison
import pandas as pd
import numpy as np
import forecast_engine.modelling.utils.nodes as utils
from pandas import MultiIndex
import datetime 

def forecast_engine(ti, months_to_forecast,debug_flag,debug_models,list_of_models,single_series,  all_at_once_flag, validation_window, max_validation, outlier_flag, select_variable_flag):
  ts1 = datetime.datetime.now().timestamp()
  if (single_series == 0):
    ti = pd.Series(ti)

  
  series_key = str(ti['index'])
  print(series_key)

  ti1, ti_key, variable_list, series_flag = utils.get_ti_in_shape(ti.copy())
  
  ti = ti1.copy()
  ts2 = datetime.datetime.now().timestamp()

  
  row_counter = ti.shape[0] +1
  fitted_values_list=list()
  
  df_multivariate = pd.DataFrame()
  df_univariate = pd.DataFrame()
  df_ensemble = pd.DataFrame()
  
  oos_length = 1
  if all_at_once_flag == 1:
    oos_length = months_to_forecast
    months_to_forecast = 1
  cols_proper = list(ti.columns)
  if debug_flag==1: print('Run Number '+str(i))
  ti = ti.append(pd.Series(np.nan),ignore_index=True)    
  ti = ti.append(pd.Series(np.nan),ignore_index=True)    
  ti = ti.append(pd.Series(np.nan),ignore_index=True)    
  ti = ti.append(pd.Series(np.nan),ignore_index=True)    
  ti = ti.append(pd.Series(np.nan),ignore_index=True)    
  ti = ti.append(pd.Series(np.nan),ignore_index=True)    
  ti = ti.append(pd.Series(np.nan),ignore_index=True)    
  ti = ti.append(pd.Series(np.nan),ignore_index=True)    
  ti = ti.append(pd.Series(np.nan),ignore_index=True)    
  ti = ti.append(pd.Series(np.nan),ignore_index=True)    
  ti = ti.append(pd.Series(np.nan),ignore_index=True)    
  ti = ti.append(pd.Series(np.nan),ignore_index=True)    

  ti = ti[cols_proper]
  ti['Header'] = pd.to_datetime(ti['Header'],format="%Y - %m")
  ti['Header'] = pd.date_range(start=ti['Header'][0],periods=(ti.shape[0]),freq='MS').strftime('%Y - %m')
  print(ti.tail(20))

  for i in range(0,months_to_forecast):
    
    fts1 = datetime.datetime.now().timestamp()
    op = model_comparison.model_comparison_function(ti.copy(), series_flag, row_counter,  list_of_models, debug_flag, debug_models, validation_window, max_validation, variable_list, single_series, oos_length, outlier_flag, select_variable_flag)
    
    df_univariate=pd.concat([df_univariate,pd.concat([pd.Series([i]*len(op['df_univariate'])),op['df_univariate']],axis=1)],axis=0)
    df_univariate=df_univariate.reset_index(drop=True)
    df_multivariate=pd.concat([df_multivariate,pd.concat([pd.Series([i]*len(op['df_multivariate'])),op['df_multivariate']],axis=1)],axis=0)
    df_multivariate=df_multivariate.reset_index(drop=True)
  
    df_ensemble=pd.concat([df_ensemble,pd.concat([pd.Series([i]*len(op['df_ensemble'])),op['df_ensemble']],axis=1)],axis=0)
    df_ensemble=df_ensemble.reset_index(drop=True)
    
    
    var_coeff = op['var_coeff'].copy()
    var_coeff = var_coeff.reset_index(drop=True)
    dataf = op['dataf'].copy()
    dataf = dataf.reset_index(drop=True)
    all_forecasts = op['all_forecasts'].copy()
    all_forecasts = all_forecasts.reset_index(drop=True)
    oos_forecast = op['oos_forecast'].copy()
    oos_forecast = oos_forecast.reset_index(drop=True)
    op['classifier_train_set']['date_'] = np.nan
    op['classifier_train_set']['date_'].iloc[0] = oos_forecast['date_'].iloc[0]
    
    if(i == 0):
      forecast_df = all_forecasts.copy()
      classifier_train_set = op['classifier_train_set'].copy()
      coeffs = var_coeff.copy()
      
    else:
      forecast_df = pd.concat([forecast_df,pd.DataFrame(all_forecasts.iloc[(all_forecasts.shape[0]-1),:]).T])
      classifier_train_set = pd.concat([classifier_train_set,op['classifier_train_set']],axis=0)
      coeffs = pd.concat([coeffs,op['var_coeff']],axis=0)
      
    fitted_values_list.append(op['fitted_values_list'])
    ti['Value'][(row_counter)-1] = oos_forecast['forecast'][0]
    row_counter = row_counter+1
    
    fts2 = datetime.datetime.now().timestamp()
    forecast_df['timing'] = fts2 - fts1

  ts3 = datetime.datetime.now().timestamp()
  forecast_out = forecast_df.copy()

  forecast_out.columns=['DATE', 'Forecast', 'Method', 'Forecast_accuracy', 'Actual', 'timing']
  forecast_out = forecast_out.reset_index(drop=True)

  classifier_train_set['index'] = ti_key['index']
  ti_key = pd.DataFrame([ti_key])
  

  key_df_univariate = pd.concat([ti_key]*df_univariate.shape[0],axis=0, ignore_index=True)
  df_univariate = pd.concat([key_df_univariate,df_univariate],axis=1)
  key_df_multivariate = pd.concat([ti_key]*df_multivariate.shape[0],axis=0, ignore_index=True)
  df_multivariate = pd.concat([key_df_multivariate,df_multivariate],axis=1)
  key_df_ensemble = pd.concat([ti_key]*df_ensemble.shape[0],axis=0, ignore_index=True)
  df_ensemble = pd.concat([key_df_ensemble,df_ensemble],axis=1)
  
  try:
    key_var = pd.concat([ti_key]*var_coeff.shape[0],axis=0, ignore_index=True)
    var_coeff = pd.concat([key_var,var_coeff],axis=1)
  except: pass
  
  ti_key = pd.concat([ti_key]*forecast_out.shape[0], ignore_index=True)
  output = pd.concat([ti_key,forecast_out],axis=1)
   
  try:
    var_coeff['key'] = var_coeff['index']
  except: pass
  
 
  try:
    ti_imputed = ti[ti['Header'].isin(var_coeff['Date'].unique().tolist())]
    ti_imputed = pd.melt(ti_imputed, id_vars = 'Header')
    ti_imputed.columns = ['Date','Driver','Imputed_Value']
    var_coeff = pd.merge(var_coeff,ti_imputed,on = ['Date','Driver'])
  except:
    var_coeff = pd.DataFrame(columns=['index','randNumCol','Model','Date','Driver','Impact','key','Imputed_Value'])
  
  ts4 = datetime.datetime.now().timestamp()
  output['total'] = ts4 - ts1

  ts4 = datetime.datetime.now().timestamp()
  output['pre_timing'] = ts2 - ts1
  output['mid_timing'] = ts3 - ts2
  output['post_timing'] = ts4 - ts3
  output['total'] = ts4 - ts1
    
  return {'dataf':dataf,'output':output,'classifier_train_set':classifier_train_set,'var_coeff':var_coeff,'df_univariate':df_univariate ,'df_multivariate':df_multivariate, 'df_ensemble':df_ensemble,'fitted_values_list':fitted_values_list}
