import sys,os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artofacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)

            return pred
        
        except Exception as e:
            logging.info('Exception occured in prediction')
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,carat,depth,table,x,y,z,cut,color,clarity):
        self.x=x
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut=cut 
        self.color=color
        self.clarity=clarity

    def get_data_as_DataFrame(self):

        try:
            custom_data_input_dict={
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }
            df=pd.DataFrame(custom_data_input_dict)
            logging.info('DataFrame Gathered')
            return df
        
        except Exception as e:
            logging.info("Exception occured in predict pipeline")
            raise CustomException(e,sys)










