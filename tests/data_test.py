from Gemstone.pipeline.predict_pipeline import CustomData,PredictPipeline
from Gemstone.config.logger import logging
from Gemstone.config.exception import CustomException
import sys

class TestResult:
    
    def __init__(self) -> None:
        pass

    def make_prediction(self,carat,depth,table,x,y,z,cut,color,clarity):
        try:
            # calling CustomData method to convert given data to pandas dataframe
            data = CustomData(
                carat = float(carat),
                depth = float(depth),
                table = float(table),
                x = float(x),
                y = float(y),
                z = float(z),
                cut = cut,
                color= color,
                clarity = clarity
            )
            df=data.get_data_as_data_frame()
            predict_pipeline=PredictPipeline()
            results = predict_pipeline.predict(df)
            logging.info(f'Prediction Completed and the result is : {results}')
            return results
        except Exception as e:
            logging.info(
                "Exited the predict method of the PredictPipeline class")
            raise CustomException(e, sys) from e
        
        
if __name__=="__main__":
    obj = TestResult()
    # 1.52,Premium,F,VS2,62.2,58.0,7.27,7.33,4.55
    # 1.11,Premium,D,SI1,60.6,59.0,6.74,6.68,4.06
    carat = 1.11
    cut = "Premium"
    color = "D"
    clarity = "SI1"
    depth = 60.6
    table = 59.0
    x = 6.74
    y = 6.68
    z = 4.06
    result = obj.make_prediction(carat,depth,table,x,y,z,cut,color,clarity)
    print(round(result[0]))