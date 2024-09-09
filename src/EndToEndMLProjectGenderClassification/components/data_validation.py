import os
from EndToEndMLProjectGenderClassification import logger
import pandas as pd
from EndToEndMLProjectGenderClassification.config.configuration import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validation_columns(self)-> bool:
        try:
            validation_status=None

            df=df=pd.read_csv(self.config.data_path)

            colms=list(df.columns)
            schemas=self.config.all_schema.keys()

            for col in colms:
                if col not in schemas:
                    validation_status=False
                    with open(self.config.STATUS_FILE,"w") as f :
                        f.write(f"validation status : {validation_status}")

                else:
                    validation_status=True
                    with open(self.config.STATUS_FILE,"w") as f :
                        f.write(f"validation status : {validation_status}")

            return validation_status 

        except Exception as e :
            raise e    