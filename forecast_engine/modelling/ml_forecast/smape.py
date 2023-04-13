def smape(y_true, y_pred):
    """
    Used for calculating symmetric mapes. Function takes actuals and predicted and calculates smape
    {y_true: actuals,
    y_pred: predicted}
    """
    import numpy as np
    return 100 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
   