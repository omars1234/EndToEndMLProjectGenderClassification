
from src.EndToEndMLProjectGenderClassification.constants import *
from src.EndToEndMLProjectGenderClassification.utils.common import read_yaml,create_directories
from EndToEndMLProjectGenderClassification.entity.config_entity import (DataIngestionConfig,
                                                                        DataValidationConfig,
                                                                        DataTransfornmationConfig,
                                                                        TrainingConfig,
                                                                        ModelEvaluationConfig)



class ConfigurationManager:
    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH,
                 schema_filepath=SCHEMA_FILE_PATH) -> None:
        
        self.config=read_yaml(config_filepath)
        self.params=read_yaml(params_filepath)
        self.schema=read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self):
        config=self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file
        )

        return data_ingestion_config
    

    def get_data_validation_config(self):
        config=self.config.data_validation
        schema=self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            STATUS_FILE=config.STATUS_FILE,
            all_schema=schema
        )
        
        return data_validation_config
    

    def get_data_transformation_config(self) -> DataTransfornmationConfig:
        config=self.config.data_transformation
        schema=self.schema
    
        create_directories([config.root_dir])
        
        data_transformation_config  = DataTransfornmationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            drop_cols=schema.DROP_COLUMNS,
            lblenc_cols=schema.LABL_ENCODING,
            ordinal_cols=schema.ORDINAL_ENCODING,
            trans_cols=schema.TRANSFORM_FEATURES,
            numerical_cols=schema.NUMERICAL_FEATURES,
            target_cols=schema.TARGET_COLUMN
        )

        return data_transformation_config 
    

    def get_training_config(self)-> TrainingConfig:
        config=self.config.training
        params=self.params.GradientBoostingClassifier
        schema=self.schema
      

        create_directories([config.root_dir])

        training_config = TrainingConfig(
            root_dir=config.root_dir,
            train_data_arr_path=config.train_data_arr_path,
            model_name=config.model_name,
            n_estimators=params.n_estimators,
            min_samples_split=params.min_samples_split,
            min_samples_leaf=params.min_samples_leaf,
            random_state=params.random_state,
            target_column=schema.TARGET_COLUMN

        )

        return training_config
    

    def get_model_evaluation_config(self)-> ModelEvaluationConfig:
        config=self.config.model_evaluation
        params=self.params.GradientBoostingClassifier
        schema=self.schema

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_arr_path=config.test_data_arr_path,
            model_path=config.model_path,
            metrics_file_name=config.metrics_file_name,
            all_params=params,
            target_column=schema.TARGET_COLUMN

        )

        return model_evaluation_config