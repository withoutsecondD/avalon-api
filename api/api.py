import json
from pathlib import Path
import dotenv
import os
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import requests

from api.utils import construct_response


ROOT_DIR = Path(dotenv.find_dotenv()).parent

# load env variables
dotenv.load_dotenv(f'{ROOT_DIR}/.env', override=True)
TF_SERVING_BASE_URL = os.getenv('TF_SERVING_BASE_URL')
FRONTEND_ADDRESS = os.getenv('FRONTEND_ADDR')
MODEL_NAME = os.getenv('MODEL_NAME')

PATH_TO_SUPERVISED_RESULTS = os.getenv('PATH_TO_SUPERVISED_RESULTS')
PATH_TO_UNSUPERVISED_RESULTS = os.getenv('PATH_TO_UNSUPERVISED_RESULTS')

app = FastAPI()
app.mount("/resources", StaticFiles(directory="../resources"), name="resources")

# allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ADDRESS],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


@app.post('/predict')
async def predict(image:UploadFile=File(...)):
    image_bytes = await image.read()

    print(image_bytes)
    if len(image_bytes) > 10 * 1024 * 1024:
        return RequestValidationError('file too large')

    return construct_response(image_bytes[:100])


@app.get('/results')
def results(model:str):
    if model == 'supervised':
        with open(f'{ROOT_DIR}/{PATH_TO_SUPERVISED_RESULTS}', 'r') as result:
            return construct_response(json.load(result))
    elif model == 'unsupervised':
        with open(f'{ROOT_DIR}/{PATH_TO_UNSUPERVISED_RESULTS}', 'r') as result:
            return construct_response(json.load(result))
    else:
        raise RequestValidationError('unknown model selected')


@app.exception_handler(Exception)
async def exception_handler(req:Request, exc:Exception):
    return JSONResponse(
        status_code=500,
        content=construct_response(None, str(exc)),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(req:Request, exc: RequestValidationError):
    message = "; ".join([f"{err['msg']} at {err['loc']}" for err in exc.errors()])

    return JSONResponse(
        status_code=400,
        content=construct_response(None, message)
    )