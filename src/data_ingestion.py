import os 
import kagglehub
import shutil 
from src.logger import get_logger 
from config.data_ingestion_config import * 
from src.custom_exception import CustomException 
import zipfile 

logger=get_logger(__name__)
class DataIngestion: 
    def __init__(self,dataset_name:str , target_dir:str):
        self.dataset_name=dataset_name 
        
        self.target_dir=target_dir 
        
    def create_raw_dir(self): 
        raw_dir=os.path.join(self.target_dir,"raw")
        if not os.path.exists(raw_dir): 
            try:
                os.makedirs(raw_dir) 
                logger.info(f"Created the {raw_dir}") 
            except Exception as e: 
                logger.error("Error while creating dir")  
                raise CustomException("Fail to create raw dir",e)
        return raw_dir
            
    def extract_images_and_labels(self, path:str, raw_dir:str):
        try: 
            if path.endswith('.zip'):
                logger.info("Extracting zip file") 
                with zipfile.ZipFile(path,'r') as zip_ref: 
                    zip_ref.extractall(path) 
            images_folder=os.path.join(path,"Images")
            labels_folder=os.path.join(path,"Labels") 
            
            if os.path.exists(images_folder): 
                    shutil.move(images_folder,os.path.join(raw_dir,"Images")) 
                    logger.info("Images moved successfully ")
            else: 
                logger.info("Images folder not exist")

            if os.path.exists(labels_folder): 
                    shutil.move(labels_folder,os.path.join(raw_dir,"Labels")) 
                    logger.info("Lables moved successfully ")
            else: 
                logger.info("labels folder not exist")
            print("Images path:", images_folder)
            print("Exists:", os.path.exists(images_folder))

            print("Labels path:", labels_folder)
            print("Exists:", os.path.exists(labels_folder))

        except Exception as e: 
                logger.error("Error while creating images n labels")  
                raise CustomException("Fail to move directories",e)
        
    def download_dataset(self,raw_dir:str): 
        try: 
            path= kagglehub.dataset_download(self.dataset_name)

            print("DOWNLOAD PATH =", path)
            print("PATH EXISTS =", os.path.exists(path))
            print("CONTENTS =", os.listdir(path))
            logger.info(f"downlaod data from {path}") 
            
            self.extract_images_and_labels(path,raw_dir)
        
        
        except Exception as e: 
                logger.error("Error while download data images n labels")  
                raise CustomException("Fail to download data ",e)
            
    def run(self): 
        try: 
            raw_dir = self.create_raw_dir()
            self.download_dataset(raw_dir)
           
        except Exception as e: 
                logger.error("Error while data ingestion")  
                raise CustomException("Fail to data ingestion",e)
            
if __name__=="__main__": 
    data_ingestion=DataIngestion(DATASET_NAME,TARGET_DIR) 
    data_ingestion.run()
    
    
    
    