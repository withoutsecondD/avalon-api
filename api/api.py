from pathlib import Path
import dotenv
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import requests

from api.utils import construct_response


app = FastAPI()

# load env variables
dotenv.load_dotenv(dotenv.find_dotenv(), override=True)
TF_SERVING_BASE_URL = os.getenv("TF_SERVING_BASE_URL")
MODEL_NAME=os.getenv("MODEL_NAME")

class PredictionRequest(BaseModel):
    instances: List[float]

@app.post('/predict')
def predict(req_data:PredictionRequest):
    payload = {
        'instances': req_data.instances
    }

    resp = requests.post(f'{TF_SERVING_BASE_URL}/v1/models/{MODEL_NAME}:predict', json=payload)
    resp.raise_for_status()

    predictions = resp.json()
    return construct_response(predictions)


@app.exception_handler(Exception)
async def exception_handler(req:Request, exc:Exception):
    return JSONResponse(
        status_code=500,
        content=construct_response(None, str(exc)),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    message = "; ".join([f"{err['msg']} at {err['loc']}" for err in exc.errors()])
    return JSONResponse(
        status_code=400,
        content=construct_response(None, message)
    )