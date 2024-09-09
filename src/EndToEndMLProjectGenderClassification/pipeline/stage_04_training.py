
from EndToEndMLProjectGenderClassification.config.configuration import ConfigurationManager
from EndToEndMLProjectGenderClassification.components.data_training import Training
from EndToEndMLProjectGenderClassification.utils.common import logger


STAGE_NAME = "Data Ingestion stage"

class TrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.initiate_Training()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = TrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e