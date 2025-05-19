# avalon-api
Avalon API backend application for DS&amp;ML Project \
Includes utilization of supervised, unsupervised algorithms and Computer Vision using Neural Networks to solve Machine Learning tasks

## Installation
TODO

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

After uploading a photo with a person, endpoint returns a new photo with bounding box that shows where the face is located with label `Result: Smiling` or `Result: Not smiling`

**Example request:** 
- POST query with image as FormData \
**Ensure that you append the file to FormData with key `image`**

**Example response:**
- Reponse body contains binary data of the jpg image

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
    "knn": {
      "metrics": {
        "R2": 0,
        "MAE": 0,
        "D2MAE": 0
      },
      "cv_results": {
        "MAE": [0, 0, 0, 0],
        "best_k": 0
      }
    }
    ...
  },
  "classification": {
    "knn": {
      "metrics": {
        "Accuracy": 0,
        "f1": 0
      },
      "cv_results": {
        "Accuracy": [0, 0, 0, 0],
        "best_k": 0
      }
    }
    ...
  }
}
```

- `unsupervised:`
```
{
  "decomposition": {
    "pca": {
      "explained_variance_ratio": [],
      "noise_variance": 0,
      "n_components": 0,
      "Accuracy": 0,
      "f1_score": 0,
      "roc_curve": {
        "tpr": [0, 0, 0, 0],
        "fpr": [0, 0, 0, 0]
      },
      "roc_auc": 0
    }
  },
  "clustering": {
    "inertia_results": [0, 0, 0, 0],
    "Accuracy": 0,
    "f1_score": 0,
    "roc_curve": {
      "tpr": [0, 0, 0, 0],
      "fpr": [0, 0, 0, 0]
    },
    "roc_auc": 0
  }
  ...
}
```