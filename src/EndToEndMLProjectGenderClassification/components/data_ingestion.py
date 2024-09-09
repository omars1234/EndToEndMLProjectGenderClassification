
import os
import pandas as pd
from EndToEndMLProjectGenderClassification import logger
from EndToEndMLProjectGenderClassification.utils.common import get_size
from EndToEndMLProjectGenderClassification.config.configuration import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            df=pd.read_csv(self.config.source_URL)
            df.to_csv(self.config.local_data_file,index=False,header=True)
            logger.info(df.dtypes) 
            

        logger.info("Ingestion of the data is completed")