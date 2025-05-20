# avalon-api
Avalon API backend application for DS&amp;ML Project \
Includes utilization of supervised, unsupervised algorithms and Computer Vision using Neural Networks to solve Machine Learning tasks

Our project is dedicated for analyzing, building infrastructure and providing comprehensive UI for interaction to client, using those datasets: 

* [Global Development indicators](https://www.kaggle.com/datasets/michaelmatta0/global-development-indicators-2000-2020) for supervised type of analysis
* [UK Road Traffic Collision Dataset](https://www.kaggle.com/datasets/salmankhaliq22/road-traffic-collision-dataset) for unsupervised type of analysis
* [CelebFaces Attributes (CelebA) Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset?select=list_attr_celeba.csv) for training and validating the CNN MobilenetV2 for Computer Vision features extraction.

## Installation
Clone the repository
```
git clone https://github.com/withoutsecondD/avalon-api.git
```

Setup venv and activate it
```
cd avalon-api
python -m venv .venv

source .venv/bin/activate
or
.\.venv\Scripts\activate (Windows)
```

Install requirements
```
pip install -r requirements.txt
```

Create .env file at the root of the project and fill following fields
```
# address of TF Serving container
TF_SERVING_BASE_URL="http://localhost:8501"

# set the same model name as in TF Serving container
MODEL_NAME="smile"

# address of frontend app, only needed for CORS 
FRONTEND_ADDR="http://localhost:3000"

# paths to results of supervised/unsupervised learning
# algorithms results 
PATH_TO_SUPERVISED_RESULTS="model/results/supervised.json"
PATH_TO_UNSUPERVISED_RESULTS="model/results/unsupervised.json"
```


Go to api directory and run the app
```
fastapi dev api.py
```

Application will be available at `localhost:8000`

## Usage
### API

---

**`GET /resources/{file_name}`**

Returns a static file specified by `file_name`

**Example request:**
```
curl localhost:8000/resources/dummy.csv
```

**Example response**
- Response body contains the file itself

---

**`POST /predict`**

After uploading a photo with a person, endpoint returns confidence score which represents how possible is that the person on uploaded photo is smiling 

**Example request:** 
- POST query with image as FormData \
**Ensure that you append the file to FormData with key `image`**

**Example response:**
```
{
  "data": {
    "confidence": "93.31%"
  },
  "success": true,
  "message": ""
}
```

---

**`GET /results?model={model}`**

Returns a results of specified model as json, results includes various metrics, such as accuracy, f2 score, optimal k and so on, response structure may vary depending on choice of the model

**Parameters:**
- `model` - results of which model to return, possible values are `supervised`, `unsupervised`

**Example request:**
```
curl localhost:8000/results?model=supervised
```

**Example response:**

Successfull response
```
{
    "data": <MODEL_RESULTS>,
    "success": true,
    "message": ""
}
```

Unsuccessfull response
```
{
    "data": null,
    "success" false,
    "message" <ERROR_MESSAGE>
}
```

**Note:**

`MODEL_RESULTS` structure depends on choice of the model:

- `supervised:`
```
{
  "regression": {
    "linear": {
      "r2": 0.038026047739232904,
      "rmse": 261.6237722008474,
      "mae": 4.1611325886124755,
      "mse": 261.6237722008474
    },
    "dtree_reg": {
      "d2mae": 0.6506618433481275,
      "cv_results": {
        "d2mae_results": [0.167, 0.187, ..., 0.644],
        "best_maxdepth": 20
      }
    },
    "randomforest_reg": {
      "mae": 3.0555189593971828
    },
    "knn": {
      "mae": 4.842358534082274,
      "cv_results": {
        "mae_results": [5.657, 5.190, ..., 4.873],
        "best_k": 19
      }
    }
  },
  "classification": {
    "logistic": {
      "accuracy": 0.9508098380323935,
      "roc_auc": 0.3973070708625067,
      "log_loss": 0.6220748882638015
    },
    "dtree_clf": {
      "accuracy": 0.978069498069498,
      "cv_results": {
        "accuracy_results": [0.0, 0.794, 0.946, ..., 0.978],
        "best_maxdepth": 5
      }
    },
    "randomforest_clf": {
      "accuracy": 0.9994001199760048
    },
    "naiveB": {
      "accuracy": 0.9070185962807439
    },
    "knn": {
      "accuracy": 0.9524838275194425,
      "cv_results": {
        "accuracy_results": [0.908, 0.945, ..., 0.952],
        "best_k": 9
      }
    },
    "svc": {
      "accuracy": 0.9508098380323935,
      "f1_score": 0.0
    }
  }
}
```

- `unsupervised:`
```
{
    "decomposition": {
        "pca": {
            "explained_variance_ratio": [
                0.17691734853507976,
                0.06470805720522751,
                0.0615497440233952,
                0.05049413014730652,
                0.04790211649338024
            ],
            "noise_variance": 0.0012627757554276168,
            "n_components": 5,
            "accuracy": 0.8977539389875964,
            "f1_score": 0.31537419772713887
        },
        "nmf": {
            "explained_variance_ratio": [
                0.17691734853507976,
                0.06470805720522751,
                0.0615497440233952,
                0.05049413014730652,
                0.04790211649338024
            ],
            "noise_variance": 0.0012627757554276168,
            "n_components": 5,
            "accuracy": 0.8977539389875964,
            "f1_score": 0.31537419772713887
        },
        "isomap": {
            "n_components": 5,
            "accuracy": 0.8977539389875964,
            "f1_score": 0.31537419772713887
        },
        "tsne": {
            "n_components": 5,
            "accuracy": 0.8977539389875964,
            "f1_score": 0.31537419772713887
        }
    },
    "clustering": {
        "inertia_results": [
            58016.19292814988,
            51369.142981542005,
            43204.366726449094,
            38669.96911231872
        ],
        "accuracy": 0.8977539389875964,
        "f1_score": 0.31537419772713887
    }
}
```
