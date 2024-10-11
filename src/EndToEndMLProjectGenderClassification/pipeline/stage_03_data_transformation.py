from EndToEndMLProjectGenderClassification.config.configuration import ConfigurationManager
from EndToEndMLProjectGenderClassification.components.data_transformation import DataTransfornmation
from EndToEndMLProjectGenderClassification.utils.common import logger


STAGE_NAME = "Data Ingestion stage"

class DataTransfornmationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransfornmation(config=data_transformation_config)
        data_transformation.data_preperation()
        data_transformation.train_test_transformation()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransfornmationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e