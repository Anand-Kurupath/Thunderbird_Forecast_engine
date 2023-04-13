import forecast_engine.modelling.ml_forecast.smape as smape
def mape_prep(final_model_fit):
    """ Used for calculating smapes for the valudation period. Function takes merged model output and actuals data as inputs 
    and creates best model data frame at SKU level and validation prediction data frame
    {final_model_fit: merged model output and actuals data}
    """


    import pandas as pd
    skus_unique = list(final_model_fit['unique_id'].unique())
    models_ml = list(set(list(final_model_fit.columns)) - set(['y','ds','unique_id']))
    model_skus = {}
    best_model_df = pd.DataFrame(columns=['unique_id','model','validation_mape'])
    for i in skus_unique:
        sku_model_lst = []
        df_mape = final_model_fit[final_model_fit['unique_id']==i]
        df_mape = df_mape[~df_mape['y'].astype(float).isna()]
        df_mape = df_mape[df_mape['y'].astype(float)>=0]
        dic = {}
        for m in models_ml:
            mape = smape.smape(df_mape['y'],df_mape[m])
            dic[m] = mape
        min_mape = min(dic.items(), key=lambda x: x[1])
        temp_df = pd.DataFrame(list(min_mape)).T
        temp_df.columns = ['model','validation_mape']
        temp_df['unique_id'] = i
        best_model_df = pd.concat([best_model_df,temp_df])
        if min_mape[0] not in model_skus: model_skus[min_mape[0]] =  [i]
        else : model_skus[min_mape[0]].append(i)
    
    valid_predictions = pd.DataFrame(columns = ['Forecast', 'y', 'ds', 'unique_id', 'method'])
    for model,skus in  model_skus.items():
        temp_df = final_model_fit[[model,'y','ds','unique_id']]
        temp_df = temp_df[temp_df['unique_id'].isin(skus)]
        temp_df.rename(columns={model:'Forecast'},inplace = True)
        temp_df['method'] = model
        temp_df['Forecast'] = temp_df['Forecast'].apply(lambda x : x if x > 0 else 0)
        valid_predictions = pd.concat([valid_predictions,temp_df])


    valid_predictions['year'] = valid_predictions['ds'].astype(str).apply(lambda x:x.split('-')[0])
    valid_predictions['month'] = valid_predictions['ds'].astype(str).apply(lambda x:x.split('-')[1])
    valid_predictions['ds'] = valid_predictions['year'].astype(str)  + " - " +valid_predictions['month'].astype(str)
    valid_predictions['key'] = valid_predictions['unique_id'] + valid_predictions['ds']

    return best_model_df, valid_predictions