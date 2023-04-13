from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import os
import datetime
import statsmodels.api as sm
import shap
import matplotlib.pyplot as plt
import holidays



month_map = {'1':'01', '2':'02', '3':'03', '4':'04', '5':'05', '6':'06', '7':'07', '8':'08', '9':'09'}
month_map_2 = {'JAN':'01', 'FEB':'02', 'MAR':'03', 'APR':'04', 'MAY':'05', 'JUN':'06', 'JUL':'07', 'AUG':'08', 'SEP':'09','OCT':'10','NOV':'11','DEC':'12'}
month_map_3 = {'January':'01', 'February':'02', 'March':'03', 'April':'04', 'May':'05', 'June':'06', 'July':'07', 'August':'08', 'September':'09','October':'10','November':'11','December':'12'}
    

def read_files(filenames, folder_path ,header, skip_rows, sheet = 'Sheet1' ):
    """
    Used for reading excel files. Function takes filenames, folder path as inputs and reads and concats all data in the path
    {filenames: list of all file names in the folder,
    folder_path: specified folder path}
    """

    def read_file(filename,header, skip_rows,sheet = sheet, folder=None ):
        # TODO: add docstring
        if folder is not None:
            filename = os.path.join(folder, filename)
        df = pd.read_excel(filename,sheet_name = sheet,header = header,skiprows = skip_rows )
        df.reset_index(drop=True, inplace=True)
        return df

    merged_df = pd.concat([read_file(f,header,skip_rows,sheet,folder=folder_path ) for f in filenames])
    return merged_df



def read_prep_sales_order(path, year_list = ['2019','2020','2021','2022','2023']):
    """
    Used for reading and preprocessing sales order files. Function takes sales order data path, reads, preprocess and concats all sales order files in the path
    {path: specified folder path}
    """
    
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    sales_data = read_files(onlyfiles,path,header=0,skip_rows=0)
    sales_data = sales_data[sales_data['Requested deliv.date'] != "#"]
    sales_data = sales_data[sales_data['Requested deliv.date'].notna()]
    sales_data['month'] = sales_data['Requested deliv.date'].apply(lambda x : x.split('.')[1])
    sales_data['month'] = sales_data['month'].map(lambda x: month_map.get(x, x))

    sales_data['year'] = sales_data['Requested deliv.date'].apply(lambda x : x.split('.')[2])

    sales_data['version'] = sales_data['year'] + " - " + sales_data['month'] 
    # sales_data['version'] = sales_data['version'].apply(lambda x: datetime.datetime.strptime(x, "%Y - %m").date() )

    sales_data = sales_data[sales_data['Order Qty. (SU)']!="*"]
    sales_data = sales_data[sales_data['Material']!="#"]
    sales_data['Order Qty. (SU)'] = sales_data['Order Qty. (SU)'].astype(float)
    sales_data = sales_data[sales_data['Order Qty. (SU)']>0]

    sales_data['order_value'] = sales_data['Net Price based on SU'] * sales_data['Order Qty. (SU)']
    sales = sales_data.groupby(['Material','version','year','month', 'Sales document']).agg({'Order Qty. (SU)':sum}).reset_index()
    
    sales_data = sales_data.groupby(['Material','version','year','month']).agg({'Order Qty. (SU)':sum}).reset_index()
    lst_filt = [i for e in year_list for i in list(sales_data['version'].unique()) if e in str(i)]
    sales_data = sales_data[sales_data['version'].isin(lst_filt)]
    sales = sales[sales['version'].isin(lst_filt)]
    sales.rename(columns={'Order Qty. (SU)':'y'},inplace=True)

    df = sales_data.pivot(index=["Material"],columns="version", values='Order Qty. (SU)') \
       .reset_index().rename_axis(None, axis=1)

    return sales, df





def read_prep_actuals(path, run_level, year_list = ['2016','2017','2018','2019','2020','2021','2022','2023']):
    """
    Used for reading and preprocessing actual sales file. Function takes actual sales, reads and preprocess actual sales file in the path
    {path: specified folder path}
    """

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    actual_data = read_files(onlyfiles,path,header=0,skip_rows=0)
    raw_data = actual_data.copy()


    actual_data['month'] = actual_data['Fiscal year/period Description'].apply(lambda x:x.split(' ')[0])
    actual_data['year'] = actual_data['Fiscal year/period Description'].apply(lambda x:x.split(' ')[1])
    actual_data['month'] = actual_data['month'].map(lambda x: month_map_3.get(x, x))
    actual_data['version'] = actual_data['year'] + " - " + actual_data['month'] 


    actual_data = actual_data[actual_data['Gross Sales Qty SU']!="*"]
    actual_data = actual_data[actual_data['Material']!="#"]
    actual_data['Gross Sales Qty SU'] = actual_data['Gross Sales Qty SU'].astype(float)
    actual_data = actual_data[actual_data['Gross Sales Qty SU']>0]
    actual_data.rename(columns = {'Gross Sales Qty SU':"y"}, inplace=True)
    actual_data['Material'] = actual_data[run_level].agg('-'.join, axis=1)

    actual_data = actual_data.groupby(['Material','version','year','month']).agg({'y':sum}).reset_index()
    lst_filt = [i for e in year_list for i in list(actual_data['version'].unique()) if e in str(i)]
    actual_data = actual_data[actual_data['version'].isin(lst_filt)]

    df = actual_data.pivot(index=["Material"],columns="version", values='y') \
        .reset_index().rename_axis(None, axis=1)

    return actual_data, df , raw_data




def read_prep_weather(path, country = ['Mexico'], weather_cols = ['Avg Temp (C)','Max Temp (C)','Min Temp (C)','Relative Humidity (%)','Precipitation (mm)']):
    """
    Used for reading and weather data. Function takes weather data path, reads, preprocess and concats all weather data files in the path
    {path: specified folder path}
    """
    wthr_data = pd.read_excel(path,sheet_name = 'Sheet1')
    wthr_data = wthr_data[wthr_data['Country'].isin(country)]
    wthr_data['month'] = wthr_data['Date'].astype(str).apply(lambda x : x.split('-')[1])
    wthr_data['year'] = wthr_data['Date'].astype(str).apply(lambda x : x.split('-')[0])
    wthr_data['version'] = wthr_data['year'] + " - " + wthr_data['month'] 
    # wthr_data['version'] = wthr_data['version'].apply(lambda x: datetime.datetime.strptime(x, "%Y - %m").date() )

    wthr_data = wthr_data.groupby(['version','Country'])[weather_cols].apply(lambda x : x.mean()).reset_index()

    wthr_data = wthr_data.melt(id_vars=['version','Country'], var_name='weather', value_name='value')
    wthr_data['key'] = wthr_data['version'].astype(str) + "_" + wthr_data['weather']
    col_order = ['Country']
    wthr_data = wthr_data.sort_values(['weather','version'],ascending=True)
    col_order.extend(list(wthr_data['key']))
 
    wthr_data =  wthr_data.pivot(index="Country",columns="key", values='value') \
            .reset_index().rename_axis(None, axis=1)
    wthr_data =  wthr_data.reindex(col_order, axis=1)

    return wthr_data


def read_macro_files(path,  country = ['Mexico']):
    """
    Used for reading and preprocessing macro economic data. Function takes weather data path, reads, preprocess and concats all macro economic data files in the path
    {path: specified folder path}
    """

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    def read_file(path,country,filename):
        df = pd.read_csv(os.path.join(path,filename))
        df['Month'] = df['Month'].astype(str)
        df['Month'] = df['Month'].map(lambda x: month_map.get(x, x))
        df['version'] = df['year'].astype(str) + " - " + df['Month'].astype(str)
        # df['version'] = df['version'].apply(lambda x: datetime.datetime.strptime(x, "%Y - %m").date() )
        df['key'] = df['version'].astype(str) + "_" + filename.split('.')[0]
        col_order = ['Country']
        df['Country'] = country[0]
        df = df.sort_values(['version'],ascending=True)
        col_order.extend(list(df['key']))
        df =  df.pivot(index="Country",columns="key", values=df.columns[0]) \
            .reset_index().rename_axis(None, axis=1)
        df =  df.reindex(col_order, axis=1)
        return df

    merged_df = pd.concat([read_file(path,country,f) for f in onlyfiles],axis=1)
    merged_df = merged_df.drop('Country', axis = 1)
    merged_df['Country'] = country[0]
    return merged_df


def process_inventories(df):
    """
    Used for preprocessing inventory data. Function is used inside read_process_inventories function
    {df: inventory data}
    """

    df = df.melt(id_vars=['Profit Center (Mat_Plant)', 'Unnamed: 1', 'Plant', 'Unnamed: 3','Material', 'Cal. Year/Month'], 
                    var_name='Date_inventory_value', value_name='inventory_value')
    df = df[['Material','Date_inventory_value', 'inventory_value']]
    df['date'] = df['Date_inventory_value'].apply(lambda x: x.split('_')[0])
    df['Date_inventory_value'] = df['Date_inventory_value'].apply(lambda x: x.split('_')[1])
    df = pd.pivot_table(df, values = 'inventory_value', index=['Material','date'], columns = 'Date_inventory_value').reset_index().rename_axis(None, axis=1)
    df['year'] = df["date"].apply(lambda x : x.split('.')[1])
    df['month'] = df["date"].apply(lambda x : x.split('.')[0])
    return df


def read_process_inventories(path):
    """
    Used for reading and preprocessing inventory data. Function takes weather data path, reads, preprocess and concats all inventory data files in the path
    {path: specified folder path}
    """

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    cols = ['Profit Center (Mat_Plant)', 'Unnamed: 1', 'Plant', 'Unnamed: 3','Cal. Year/Month']
    df = pd.DataFrame()
    lstq = []
    for i in onlyfiles:
        if len(df) == 0:
            df_t = pd.read_excel(join(path, i), sheet_name = 'Sheet1')
            df = pd.concat([df,process_inventories(df_t)])
            print(f"{i},{df_t.shape}")
            lstq.extend(list(df_t['Material'].unique()))
        else:
            
            df_t = pd.read_excel(join(path, i), sheet_name = 'Sheet1')
            lst = [x for x in list(df_t.columns) if x not in cols]
            df_t = process_inventories(df_t)
            df = pd.concat([df,df_t])
            print(f"{i},{df_t.shape}")
            lstq.extend(list(df_t['Material'].unique()))
    df['version'] = df['year'] + " - " + df['month'] 
    df['key'] = df['Material'] +"_"+ df['version']
    return df





def prep_dates_list(df):
  """
    Used for find all the dates in input data. Function takes input data and creates a sorted list of all available dates
    {df: input data}
    """

  dates_index=[]
  for index, column_header in enumerate(df.columns):
      try:
        pd.to_datetime(column_header)
        p = index
        dates_index.append(p)
      except:
        continue
  dates = df.columns[dates_index]
  act_dates = dates.insert(len(dates),'Material')
  return dates, act_dates




# Intermittence
def intermittence(time_series):
  """
    Used for calculating time series intermittence. Function takes input data and creates intermittence column
    {time_series: input data}
    """
  
  i=time_series.transpose().reset_index(drop=True).first_valid_index()
  time_series=time_series.iloc[i:]
  interm=time_series.isna().sum()/len(time_series)

  
  return interm




# COV
def cov(time_series):
  """
    Used for generating time series coefficient of variation. Function takes input data and creates cov column
    {time_series: input pandas series}
    """

  c = np.nanstd(time_series.astype(float))/np.nanmean(time_series.astype(float))
  if c == np.inf:
    c=0
  elif c == np.nan:
    c=0
  return c





# Time series flags
def seq_flag(df,thresh,dates,columns_save):
    """
    Used for categorizing time series into mainly sparse vs regular. Function takes input data, intermittence threshold and dates list as inputs
    and creates seq - flag column
    
    {df: input data, 
    thresh: intermittence threshold, 
    dates: available dates in the data
    columns_save: all columns in the input data}
    
    {1: Regular Series(series with intermittence less than inetermittence threshold), 
    2: Discontinued Business(series with no data available for last 6 months), 
    3: Sparsely Populated Series(series with intermittence greater than inetermittence threshold), 
    4: Recently Started Business(series with no data available from start but has values for at least last 4 months)}
    """

    df = df.reset_index(drop=True)
    df.index = list(columns_save)
    y = df[df.index.isin(columns_save)]
    y = pd.DataFrame(y)
    y = pd.DataFrame(y.transpose().reset_index(drop=True))

    if y[dates[-6:]].isnull().all(1)[0]:
        return 2
    elif y[dates[-4:]].count(axis=1).eq(4)[0] & y[dates[:-12]].count(axis=1).eq(0)[0]:
        return 4
    elif y.intermittence[0] > thresh:
        return 3
    else:
        return 1




# Time series decompose
def decomposition(df,dates,columns_save):
  """
    Used for decomposing times series into its trend, seasonality and error components. Function takes input series, list of available dates, 
    list of all column names and creates trend, seasonality and error colums.
    {df: time series,
    dates: available dates in the data
    columns_save: all columns in the input data}
    """

  df = df.reset_index(drop=True)
  df.index = list(columns_save)
  
  y = df[df.index.isin(dates)]
  y = y.fillna(0)
  
  decomp = sm.tsa.seasonal_decompose(y, period=12, extrapolate_trend='freq')
  trend = pd.DataFrame(decomp.trend)
  seasonal = pd.DataFrame(decomp.seasonal)
  error = pd.DataFrame(decomp.resid)
  
  list_1 = trend.values.ravel().tolist()
  list_2 = seasonal.values.ravel().tolist()
  list_3 = error.values.ravel().tolist()
  
  return list_2




def melt_pivot(df,i):
    """
    Used for melting and pivoting data. Function takes input data and melt-pivot data
    {df: input data
    i: columns to skip}
    """

    df = df.melt()[i:].reset_index(drop = True)
    df['Date'] = df['variable'].apply(lambda x : x.split('_')[0])
    df['variable'] = df['variable'].apply(lambda x : x.split('_')[1])
    df = df.pivot_table(index='Date', columns='variable', values='value').reset_index().rename_axis(None, axis=1)
    return df




def model_data_combine(df,weather,macro,agi):
    """
    Used for combing input data and external driver data. Function takes input data, weather, macro, and agi data and merges all
    {df: input data,
    weather: weather data,
    macro: macro data
    weather: agi data}
    """


    weather = melt_pivot(weather,1)
    macro = melt_pivot(macro.drop('Country',axis=1),0)
    agi = melt_pivot(agi,1)

    df_melt = df.melt(id_vars=['Material'])
    df_melt.rename(columns = {'variable':'Date'}, inplace =True)

    df_melt_merge = pd.merge(df_melt,agi, on='Date', how = 'left')
    df_melt_merge = pd.merge(df_melt_merge,weather, on='Date', how = 'left')
    df_melt_merge = pd.merge(df_melt_merge,macro, on='Date', how = 'left')
    df_model = df_melt_merge.copy()

    df_model['ds'] = df_model['Date'].apply(lambda x: datetime.datetime.strptime(x, "%Y - %m").date() )
    df_model.drop('Date',axis=1, inplace=True)
    df_model.rename(columns = {'value':'y'}, inplace= True)
    df_model['ds'] = pd.to_datetime(df_model['ds'])
    # df_model.fillna(0,inplace=True)
    df_model.rename(columns={'Material':'unique_id'}, inplace=True)

    return df_model




def read_agi_files(pathasi,pathndvi,pathvhi, country):
    """
    Used for reading and preprocessing agricultural index data. Function takes weather data path, reads, preprocess and concats all agricultural index data files in the path
    {pathasi: agri stress index folder path,
    pathndvi: difference vegitation index folder path,
    pathvhi: vegitation health index folder path}
    """

    asi = pd.read_csv(pathasi)
    ndvi = pd.read_csv(pathndvi)
    vhi = pd.read_csv(pathvhi)


    asi['year'] = asi['Date'].apply(lambda x: x.split('-')[0])
    asi['month'] = asi['Date'].apply(lambda x: x.split('-')[1])
    asi['version'] = asi['year'] + ' - ' +asi['month']

    ndvi['year'] = ndvi['Date'].apply(lambda x: x.split('-')[0])
    ndvi['month'] = ndvi['Date'].apply(lambda x: x.split('-')[1])
    ndvi['version'] = ndvi['year'] + ' - ' +ndvi['month']

    vhi['year'] = vhi['Date'].apply(lambda x: x.split('-')[0])
    vhi['month'] = vhi['Date'].apply(lambda x: x.split('-')[1])
    vhi['version'] = vhi['year'] + ' - ' +vhi['month']


    asi = asi.groupby(['Province','version'])['Data'].mean().reset_index()
    ndvi = ndvi.groupby(['Province','version'])['Data'].mean().reset_index()
    vhi = vhi.groupby(['Province','version'])['Data'].mean().reset_index()

    asi = asi.rename(columns={'Data':'asi'})
    ndvi = ndvi.rename(columns={'Data':'ndvi'})
    vhi = vhi.rename(columns={'Data':'vhi'})


    asi['key'] = asi['Province'].apply(lambda x : x.lower()) + asi['version']
    ndvi['key'] = ndvi['Province'].apply(lambda x : x.lower()) + ndvi['version']
    vhi['key'] = vhi['Province'].apply(lambda x : x.lower()) + vhi['version']

    agi = pd.merge(asi,ndvi[['key','ndvi']],on='key',how='left')
    agi = pd.merge(agi,vhi[['key','vhi']],on='key',how='left')
    agi.rename(columns = {'Province':'State'},inplace=True)

    agi_contr = agi.groupby(['version'])['ndvi','vhi','asi'].apply(lambda x : x.mean()).reset_index()
    col_order = [country]
    agi_contr = agi_contr.melt(id_vars=['version'], var_name='agi', value_name='value')
    for i in list(agi_contr['agi'].unique()):
        col_order.extend([x + '_' + i for x in  sorted(list(agi_contr['version'].unique()))])

    agi_contr['key'] = agi_contr['version'].astype(str) + "_" + agi_contr['agi']
    agi_contr['Country'] = country
    agi_contr = agi_contr.pivot(index="Country",columns="key", values='value') \
            .reset_index().rename_axis(None, axis=1)
    agi_contr =  agi_contr.reindex(col_order, axis=1)

    return agi_contr


def plot_validataion(valid_predictions, forecast_start_date, path):
    """
    Used for plotting the validation mapes. Function takes validation prediction and actual data, reads, preprocess, concats all saves the plot to processed data path
    {valid_predictions: test predictions for regular series,
    valid_predictions_sparse: test predictions for sparse series,
    forecast_start_date: forecast start date,
    path: processed data path}
    """

    import datetime
    import matplotlib.pyplot as plt
    path = os.path.join(path,'validation_plot.pdf')
    plot= valid_predictions.groupby(['ds'])['y','Forecast'].apply(lambda x : x.sum()).reset_index()
    plot['ds'] = plot['ds'].apply(lambda x: datetime.datetime.strptime(x, "%Y - %m").date() )
    plot = plot.loc[plot['ds'] < forecast_start_date].reset_index(drop=True)
    plot.set_index('ds',inplace=True)


    fig = plt.figure(figsize=(12,5))
    plt.xlabel('Actuals vs Forecast validation plot')

    ax1 = plot.y.plot(color='red', grid=False, label='Actual sales')
    ax2 = plot.Forecast.plot(color='blue', grid=False, secondary_y=False, label='Forecast sales')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()


    

    plt.legend(l1+l2, loc=0)
    fig.savefig(path,dpi=200)
    plt.close()
    return 



def post_process(final_output,best_model_df,forecast_start_date):

    best_model_merge = best_model_df
    final_output = pd.merge(final_output,best_model_merge,on='unique_id',how='left')

    final_output = final_output.pivot(index=['unique_id','variable','validation_mape'],columns="ds", values='value') \
        .reset_index().rename_axis(None, axis=1)

    # cols = list(final_output.columns)[:3]
    # cols.extend([forecast_start_date +  pd.DateOffset(month=4)])
    # cols.extend([x for x in list(final_output.columns) if x not in cols])
    # final_output = final_output[cols]
    return final_output



def prep_sap_output(raw_data,final_output,run_level):
    
    
    raw_data['unique_id'] = raw_data[run_level].agg('-'.join, axis=1)
    raw_data_temp = raw_data[['unique_id','Material','Country','Profit Center','Sales Qty Unit_SU']].drop_duplicates()
    
    final_output['year'] = final_output['ds'].astype(str).apply(lambda x: x.split('-')[0])
    final_output['month'] = final_output['ds'].astype(str).apply(lambda x: x.split('-')[1])
    final_output_temp = final_output[['unique_id','value','year','month']].rename(columns={'value':'Net Sales_Qty predicted','year':'Year','month':'Month'})
    sap_output = pd.merge(final_output_temp,raw_data_temp,on='unique_id',how='left')
    sap_output.rename(columns={'Material':"MATERIAL",'Country':'COUNTRY','Profit Center':'PROFIT_CTR','Sales Qty Unit_SU':'Sales_UOM'},inplace=True)
    sap_output['Version'] = np.nan
    sap_output['Terriority'] = np.nan
    sap_output['Area'] = np.nan
    sap_output['Sales Quantity'] = np.nan
    sap_output['Gross Sales Qty predectied'] = np.nan
    sap_output['Weather sales Quantity  predicted'] = np.nan
    sap_output = sap_output[['Version','MATERIAL','Terriority','Area','Year','Month','COUNTRY','PROFIT_CTR','Sales_UOM','Sales Quantity','Net Sales_Qty predicted',	'Gross Sales Qty predectied','Weather sales Quantity  predicted']]

    return sap_output


def shap_plots(train,model,best_model_df,regular_series,processed_data_path):



    shap.initjs()
    data = model.ts._transform(train[train['unique_id'].isin(regular_series)])
    data = pd.merge(data,best_model_df[['model','unique_id']], on = 'unique_id',how='left')
    models = list(model.models_.keys())



    for i in regular_series:
        try:
            data_temp = data[data['unique_id']==i].drop(['unique_id','ds','y'],axis=1).tail(12)
            selected_model = list(data_temp['model'].unique())[0]
            data_temp.drop('model',axis=1,inplace=True)
            
            if selected_model in models:
                model_path =  "/".join([processed_data_path,'shap_plots',selected_model])
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                explainer = shap.KernelExplainer(model.models_[selected_model].predict,data_temp)
                shap_values = explainer.shap_values(data_temp)
                shap.initjs()
                shap.save_html("/".join([model_path,i.replace('/','_')+'.html']), shap.force_plot(explainer.expected_value, shap_values,  data_temp,  plot_cmap="DrDb"))
                f = plt.figure()
                shap.summary_plot(shap_values, data_temp, max_display=20, plot_type='violin', show=False)
                f.savefig("/".join([model_path,i.replace('/','_')+'.png']), format='png', dpi=200, bbox_inches='tight')
                plt.close()

                f = plt.figure()
                sv = explainer.shap_values(data_temp.tail(1))  
                exp = shap.Explanation(sv,explainer.expected_value, data=data_temp.tail(1).values, feature_names=data_temp.columns)
                shap.plots.waterfall(exp[0],show=False)
                f.savefig("/".join([model_path,i.replace('/','_')+'_prior_month.png']), format='png', dpi=200, bbox_inches='tight')
                plt.close()
            else:
                continue
        except:
            continue


def mode_uni(models,stats, sparse):
    univariate_models_temp = [ 'AutoARIMA','AutoETS','AutoTheta','AutoRegressive','MSTL','ARCH(1)','HistoricAverage','Naive','RWD',
    'SeasonalNaive','WindowAverage','SESOpt','SeasESOpt','HoltWinters']
    sparse_models_temp = ['AutoRegressive','Naive','SeasonalNaive','SESOpt','HoltWinters','ADIDA', 'CrostonOptimized','CrostonSBA','IMAPA','TSB']
    
    if sparse == 0:
        set1 = set(univariate_models_temp)
        temp = univariate_models_temp    
    else:
        set1 = set(sparse_models_temp)
        temp = sparse_models_temp 
    
    set2 = set(models)
    missing = list(sorted(set1 - set2))
    
    indices_A = [temp.index(x) for x in missing]
    stats.models = [i for j, i in enumerate(stats.models) if j not in indices_A]
    return stats.models


def create_holiday(lst):
    df_holi = pd.DataFrame(columns = ['index','is_holiday','Country'])
    for i in lst:
        try:
            holiday = holidays.country_holidays(i) 
            range_of_dates = pd.date_range("2023-01-01", "2023-12-31")
            df = pd.DataFrame(
                index=range_of_dates, 
                data={"is_holiday": [date in holiday for date in range_of_dates]}
            )
            df.reset_index(inplace=True)
            df['Country'] = i
            df_holi = pd.concat([df_holi,df])
        except:
            continue

    df_holi['is_holiday'] = df_holi['is_holiday'].astype(int)
    df_holi = df_holi.groupby([(df_holi['index'].dt.month),(df_holi['Country'])])['is_holiday'].sum().reset_index()
    df_holi['key'] = df_holi['index'].astype(str) + df_holi['Country']
    df_holi.rename(columns = {'is_holiday':'holiday'},inplace=True)
    return df_holi


def post_process_sparse(oos,models,run_level,raw_data,forecast_start_date):
    models['key'] = models['models']  + models['index'] 
    oos['key'] = oos['Method']  + oos['index'] 
    oos_mer = pd.merge(oos,models[['smape','key']].drop_duplicates(),on='key',how='left').drop('key',axis=1)
    oos_mer['DATE'] = oos_mer['DATE'] .apply(lambda x: datetime.datetime.strptime(x, "%Y - %m") )
    oos_mer = oos_mer.loc[(oos_mer['DATE'] >= forecast_start_date)].reset_index(drop=True)
    oos_mer.rename(columns = {'index':'unique_id','Method':'variable','smape':'validation_mape','DATE':'ds','Forecast':'value'},inplace=True)
    oos_mer['value'] = oos_mer['value'].apply(lambda x : x if x > 0 else 0)
    final_output_sparse = oos_mer.copy()
    final_output_1_sparse = oos_mer.pivot(index=['unique_id','variable','validation_mape'],columns="ds", values='value').reset_index().rename_axis(None, axis=1)
    sap_output_sparse = prep_sap_output(raw_data,final_output_sparse,run_level)
    return final_output_1_sparse, sap_output_sparse


def valid_predictions_sprase_prep(fitted_values,sales_data,train_end_date,forecast_start_date):
    fitted_values['key'] = fitted_values['index'] + fitted_values['date_']
    sales_data_ = sales_data.copy()
    sales_data_['key'] = sales_data_['Material'] + sales_data_['version']
    validation_predictions_sparse = pd.merge(fitted_values,sales_data_[['key','y']],on='key',how='left').fillna(0)
    validation_predictions_sparse['DATE'] = validation_predictions_sparse['date_'] .apply(lambda x: datetime.datetime.strptime(x, "%Y - %m").date() )
    validation_predictions_sparse = validation_predictions_sparse.loc[(validation_predictions_sparse['DATE'] > train_end_date) & (validation_predictions_sparse['DATE'] < forecast_start_date)].reset_index(drop=True).drop('DATE',axis=1)
    validation_predictions_sparse['forecast'] = validation_predictions_sparse['forecast'].apply(lambda x : x if x > 0 else 0)
    validation_predictions_sparse.rename(columns={'date_':'ds','forecast':'Forecast','index':'unique_id'},inplace = True)
    validation_predictions_sparse['year'] = validation_predictions_sparse['ds'].apply(lambda x : x.split(' - ')[0])
    validation_predictions_sparse['month'] = validation_predictions_sparse['ds'].apply(lambda x : x.split(' - ')[1])
    return validation_predictions_sparse