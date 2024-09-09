
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    data_path: Path
    STATUS_FILE: str
    all_schema: dict
    

@dataclass(frozen=True)
class DataTransfornmationConfig:
    root_dir: Path
    data_path: Path
    drop_cols:str
    lblenc_cols:str
    ordinal_cols:str
    trans_cols:str
    numerical_cols:str
    target_cols:str    


@dataclass(frozen=True)
class TrainingConfig:
    root_dir:Path
    train_data_arr_path:Path
    model_name:str 
    n_estimators: int
    min_samples_split: int
    min_samples_leaf: int
    random_state: int
    target_column:str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir:Path
    test_data_arr_path:Path
    model_path:Path
    metrics_file_name:Path 
    target_column:str
    all_params:dict
    

    