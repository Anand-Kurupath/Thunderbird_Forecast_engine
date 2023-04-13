import pandas as pd


def output_prep(ml_,sf_,act):

    """ Used for combining model results of regular series. Function takes multivariate pipeline output, univariate pipeline output, actual data as inputs and merges all three 
    {ml_: multivariate pipeline output, sf_: univariate pipeline output, act: actual data}
    """


    
    
    
    ml_['key'] = ml_['unique_id'] + ml_['ds'].astype(str)
    sf_ = sf_.reset_index()
    sf_['key'] = sf_['unique_id'] + sf_['ds'].astype(str)
    final_model_fit = pd.merge(ml_,sf_.drop(['unique_id','ds'],axis=1),on='key',how='left')

    act = act.groupby(['Material','version'])['y'].sum().reset_index()
    act['key'] = act['Material'] + act['version']
    


    final_model_fit['year'] = final_model_fit['ds'].astype(str).apply(lambda x:x.split('-')[0])
    final_model_fit['month'] = final_model_fit['ds'].astype(str).apply(lambda x:x.split('-')[1])
    final_model_fit['Date'] = final_model_fit['year'].astype(str)  + " - " +final_model_fit['month'].astype(str)
    final_model_fit['key'] = final_model_fit['unique_id'] + final_model_fit['Date']

    final_ff = pd.merge(final_model_fit,act,on='key',how='left')
    final_ff.rename(columns={'Gross Sales Qty ZU':'y'},inplace=True)
    final_ff.fillna(0,inplace=True)
    final_model_fit = final_ff.copy()
    final_model_fit.drop(['version','year','month','Date','Material','key'],axis=1,inplace=True)

    return final_model_fit


def output_prep_sparse(sf_sparse_,act):

    """ Used for combining model results of sparse series. Function takes univariate sparse pipeline output and actual data as inputs and merges all three 
    {sf_sparse_: univariate sparse pipeline output, act: actual data}
    """



    act = act.groupby(['Material','version'])['y'].sum().reset_index()
    act['key'] = act['Material'] + act['version']
    

    final_model_fit = sf_sparse_.reset_index().copy()
    final_model_fit['year'] = final_model_fit['ds'].astype(str).apply(lambda x:x.split('-')[0])
    final_model_fit['month'] = final_model_fit['ds'].astype(str).apply(lambda x:x.split('-')[1])
    final_model_fit['Date'] = final_model_fit['year'].astype(str)  + " - " +final_model_fit['month'].astype(str)
    final_model_fit['key'] = final_model_fit['unique_id'] + final_model_fit['Date']

    final_ff = pd.merge(final_model_fit,act,on='key',how='left')
    final_ff.rename(columns={'Gross Sales Qty ZU':'y'},inplace=True)
    final_ff.fillna(0,inplace=True)
    final_model_fit = final_ff.copy()
    final_model_fit.drop(['version','year','month','Date','Material','key'],axis=1,inplace=True)

    return final_model_fit


def prep_oos(ml_oos, sf_oos, best_model_df, skus):

    """ Used for generating out of sample model results. Function takes multivariate pipeline output, univariate pipeline output and 
    univariate sparse pipeline output data as inputs and merges all three. For each sku best models from training data is chosen
    for generating out of sample results
    {ml_oos: multivariate pipeline output, sf_oos: univariate pipeline output, sf_oos_sparse: univariate sparse pipeline output,
    best_model_df: best model results for regular series, best_model_df_sparse: best model results for sparse series,
    skus: list of SKUs that has data available while calculating validation smapes}
    """




    ml_oos['key'] = ml_oos['unique_id'] + ml_oos['ds'].astype(str)
    sf_oos = sf_oos.reset_index()
    sf_oos['key'] = sf_oos['unique_id'] + sf_oos['ds'].astype(str)
    final_model_fit = pd.merge(ml_oos,sf_oos.drop(['unique_id','ds'],axis=1),on='key',how='left')
    
   
    
    final_model = final_model_fit.copy()
    best_model = best_model_df

    final_model = final_model.melt(id_vars=['unique_id','ds'])
    final_model['key'] = final_model['unique_id'] + final_model['variable']
    best_model['key'] = best_model['unique_id'] + best_model['model']

    final_model = final_model[final_model['key'].isin(list(best_model['key'].unique()))]
    final_model = final_model[final_model['unique_id'].isin(skus)]

    return final_model