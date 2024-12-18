import sys
import os
import pandas as pd
from dotenv import load_dotenv
from TSForecasting.exception.exception import TSForecastingException
from TSForecasting.logging.logger import logging
from TSForecasting.pipeline.aws_training_pipeline import TrainingPipeline
from TSForecasting.utils.main_utils.utils import load_object
from TSForecasting.utils.ml_utils.model.estimator import TransactionMonitoring
from TSForecasting.constant.training_testing_pipeline import DATA_INGESTION_COLLECTION_NAME
from TSForecasting.constant.training_testing_pipeline import DATA_INGESTION_DATABASE_NAME, DATA_INGESTION_TABLE_NAME
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import snowflake.connector


load_dotenv()

# Snowflake connection details
conn = snowflake.connector.connect(
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    warehouse=DATA_INGESTION_COLLECTION_NAME,
    database=DATA_INGESTION_DATABASE_NAME,
    schema="RAW",  # The schema within the database
    role="transform",  # The role you granted
    ocsp_fail_open=True,
    insecure_mode=True  # Disable SSL verification for debugging

)

# SQL query to fetch data
table_name = DATA_INGESTION_TABLE_NAME
query = f"SELECT * FROM {table_name};"

# Execute query and fetch data
cursor = conn.cursor()
cursor.execute(query)
data = cursor.fetchall()
columns = [col[0] for col in cursor.description]

# Convert to DataFrame
df = pd.DataFrame(data, columns=columns)

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="./templates")

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline=TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise TSForecastingException(e,sys)
    
@app.post("/predict")
async def predict_route(request: Request,file: UploadFile = File(...)):
    try:
        df=pd.read_csv(file.file)
        #print(df)
        preprocesor=load_object("final_model/preprocessor.pkl")
        final_model=load_object("final_model/model.pkl")
        network_model = TransactionMonitoring(preprocessor=preprocesor,model=final_model)
        print(df.iloc[0])
        y_pred = network_model.predict(df)
        print(y_pred)
        df['predicted_column'] = y_pred
        print(df['predicted_column'])
        #df['predicted_column'].replace(-1, 0)
        #return df.to_json()
        df.to_csv('app_output/output.csv')
        table_html = df.to_html(classes='table table-striped')
        #print(table_html)
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
        
    except Exception as e:
            raise TSForecastingException(e,sys)

    
if __name__=="__main__":
    app_run(app,host="0.0.0.0",port=8000)
