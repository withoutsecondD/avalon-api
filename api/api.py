import io
import json
from pathlib import Path
import dotenv
import os
from tensorflow.keras.preprocessing import image as kerasimg
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import requests

from api.utils import construct_response


# load env variables
dotenv.load_dotenv(override=True)
TF_SERVING_BASE_URL = os.getenv('TF_SERVING_BASE_URL')
FRONTEND_ADDRESS = os.getenv('FRONTEND_ADDR')
MODEL_NAME = os.getenv('MODEL_NAME')

PATH_TO_SUPERVISED_RESULTS = os.getenv('PATH_TO_SUPERVISED_RESULTS')
PATH_TO_UNSUPERVISED_RESULTS = os.getenv('PATH_TO_UNSUPERVISED_RESULTS')

print(TF_SERVING_BASE_URL)
print(FRONTEND_ADDRESS)
print(MODEL_NAME)
print(PATH_TO_SUPERVISED_RESULTS)
print(PATH_TO_UNSUPERVISED_RESULTS)

app = FastAPI()
app.mount("/resources", StaticFiles(directory="resources"), name="resources")

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

    if len(image_bytes) > 10 * 1024 * 1024:
        return RequestValidationError('file too large')

    img = kerasimg.load_img(io.BytesIO(image_bytes), target_size=(224, 224))
    img_arr = kerasimg.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    tensor = img_arr.tolist()

    prediction_resp = requests.post(f'{TF_SERVING_BASE_URL}/v1/models/{MODEL_NAME}:predict', json={"instances": tensor})
    resp = {
        'confidence': f'{(float(prediction_resp.json()['predictions'][0][0]) * 100):.2f}%'
    }

    return construct_response(resp)


@app.get('/results')
def results(model:str):
    if model == 'supervised':
        with open(PATH_TO_SUPERVISED_RESULTS, 'r') as result:
            return construct_response(json.load(result))
    elif model == 'unsupervised':
        with open(PATH_TO_UNSUPERVISED_RESULTS, 'r') as result:
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