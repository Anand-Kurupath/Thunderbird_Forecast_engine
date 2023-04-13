import warnings
warnings.filterwarnings('ignore')
import forecast_engine.modelling.thunderbird_global as thunderbird
import forecast_engine.modelling.utils.logger as logger
import forecast_engine.modelling.utils.config as config
from time import time
import os
import datetime

# Load config
config_catalog = config.load_catalog_params(os.getcwd(), 'conf/catalogs')
config_parameters = config.load_catalog_params(os.getcwd(), 'conf/parameters')
now = str(datetime.datetime.now().replace(second=0, microsecond=0))
version = str(config_catalog['data_preprocessing']['version'])
model_output_path = "/".join([os.getcwd(),config_catalog['output_file_path']['model_output_path'],str(config_catalog['data_preprocessing']['version'])])





def run_pipeline():
    """
    Main pipeline for Running Thunderbird
    """
    
    t1 = time()
    logger.logger.info(f" Thunderbird run started")
    logger.logger.info(" loading data preprocessing pipeline...")
    thunderbird.thunderbird_global.instantiate_inputs()
    thunderbird.thunderbird_global.drop_skus()
    thunderbird.thunderbird_global.flag_skus()
    logger.logger.info(" data preprocessing pipeline complete...")
    
    logger.logger.info(" loading model parameters...")
    thunderbird.thunderbird_global.thunder_fuel()
    logger.logger.info(" model parameters loaded...")
    thunderbird.thunderbird_global.thunder_fuel_local()
    
    logger.logger.info(" hyperparameter tuning started...")
    thunderbird.thunderbird_global.tune_engine()
    logger.logger.info(" hyperparameter tuning complete...")


    logger.logger.info(" fitting models started...")
    thunderbird.thunderbird_global.fit_models()
    logger.logger.info(" fitting models complete...")
    
    logger.logger.info(" oos prediction started...")
    thunderbird.thunderbird_global.predict_oos()
    logger.logger.info(" oos prediction complete...")

    logger.logger.info("Sparse & low volume pipeline running...")
    thunderbird.thunderbird_global.thunder_sparse_pipeline()
    logger.logger.info("Sparse & low volume pipeline run complete...")
    logger.logger.info(" running post processing pipeline...")
    
    logger.logger.info(" post processing pipeline complete...")
    logger.logger.info(" Saving model outputs...")
    thunderbird.thunderbird_global.generate_results()
    logger.logger.info(" Thunder bird forecast engine run complete.")
    t2 = time()
    elapsed = (t2 - t1)/60
    logger.logger.info(" Thunder bird run time is %f minutes." % elapsed)
    os.rename(model_output_path, model_output_path +"_"+now)
    for root, dirs, files in os.walk(model_output_path +"_"+now):
        for name in dirs:
            os.rename(os.path.join(root, name), os.path.join(root, name +"_"+now))
       
if __name__ == "__main__":
    logger.logger.info(" Running main pipeline...")
    run_pipeline()