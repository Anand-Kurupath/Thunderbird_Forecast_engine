def skus_to_exclude(data,month_thresh = 6):
    """
    Used for excluding from modelling. Function takes input data and minumum months threshold and removes SKUs having data less than the threshold
    {data: input data,
    month_thresh: minimum month threshold}
    """

    import numpy as np
    skus_exclude = []
    for i in list(data['unique_id'].unique()):
        df_temp = data[data['unique_id']==i]
        df_temp['count'] = np.where(df_temp['y']>0,1,0)
        if(sum(df_temp['count'])< month_thresh):
            skus_exclude.append(i)
    df_model = data[~data['unique_id'].isin(skus_exclude)]
    return df_model